import itertools
import logging
import os
import re
from collections import namedtuple
from heapq import nlargest
from operator import itemgetter

import nltk
import numpy as np
import spacy
from gutenberg._domain_model.exceptions import UnknownDownloadUriException
from gutenberg.acquire import load_etext
from gutenberg.acquire.text import _TEXT_CACHE
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts, get_metadata
from spacy.tokenizer import Tokenizer
from tabulate import tabulate

from weighwords import ParsimoniousLM
from weighwords.significant_words import SignificantWordsLM

authors = [
    'Carroll, Lewis',
    'Melville, Herman',
    'Doyle, Arthur Conan',
    'Wells, H. G. (Herbert George)',
]
english_works = get_etexts('language', 'en')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

top_k = 20  # How many terms per book to retrieve

en_weights = spacy.load('en_core_web_sm')
nn_tokenizer = Tokenizer(en_weights.vocab)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
token_re = re.compile(r'\w{2,}')

Document = namedtuple('Document', [
    'pk',
    'title',
    'text',
    'nltk_terms',
    'spacy_terms'
])


def spacy_tokenize(text):
    return [
        w.lower_
        for w in nn_tokenizer(text)
        if token_re.search(w.text)
    ]


def nltk_tokenize(text):
    sents = sent_detector.sentences_from_text(text)
    return [
        w.strip().lower()
        for s in sents
        for w in nltk.tokenize.word_tokenize(s)
        if token_re.search(w)
    ]


def read_book(gutenberg_id, fetch_missing=False):
    cached_fp = os.path.join(_TEXT_CACHE, '{0}.txt.gz'.format(gutenberg_id))
    if not (fetch_missing or os.path.exists(cached_fp)):
        return None

    etext = load_etext(gutenberg_id, mirror="http://eremita.di.uminho.pt/gutenberg/")
    return strip_headers(etext).strip()


def get_author_library(author_name):
    logger.info(f'Loading library for {author_name}')
    etext_ids = get_etexts('author', author_name) & english_works
    work_ids, work_texts = [], []
    for pk in etext_ids:
        try:
            text = read_book(pk)
        except UnknownDownloadUriException as e:
            logger.warning(f'Missing text: {e}')
            continue

        if text:
            work_ids.append(pk)
            work_texts.append(text)

    work_titles = [
        next(iter(get_metadata('title', pk)))
        for pk in work_ids
    ]
    nltk_terms = [nltk_tokenize(d) for d in work_texts]
    spacy_terms = [spacy_tokenize(d) for d in work_texts]

    library = {
        pk: Document(
            pk,
            *values
        )
        for pk, *values in zip(
            work_ids,
            work_titles,
            work_texts,
            nltk_terms,
            spacy_terms
        )
    }
    nltk_model = ParsimoniousLM(nltk_terms, w=.01)
    spacy_model = ParsimoniousLM(spacy_terms, w=.01)
    return library, nltk_model, spacy_model


def book_vs_author(library, nltk_model, spacy_model):
    for doc in library.values():
        nltk_termprobs = [
            (term, np.exp(p))
            for term, p in nltk_model.top(top_k, doc.nltk_terms)
        ]
        spacy_termprobs = [
            (term, np.exp(p))
            for term, p in spacy_model.top(top_k, doc.spacy_terms)
        ]
        term_probabilities = [
            nltk_tp + spacy_tp
            for nltk_tp, spacy_tp in zip(nltk_termprobs, spacy_termprobs)
        ]
        print(f'Top {top_k} terms in {doc.title}:')
        print(
            tabulate(term_probabilities, headers=(
                'NLTK term', 'p', 'SpaCy term', 'p'
            )) + "\n"
        )


if __name__ == '__main__':
    # for author in authors:
    #     book_vs_author(*get_author_library(author))

    doc_groups = [
        [
            doc.nltk_terms
            for doc in get_author_library(author)[0].values()
        ][:20]
        for author in authors
    ]
    corpus = itertools.chain(*doc_groups)
    swlm = SignificantWordsLM(corpus, w=0.01)
    for i, author in enumerate(authors):
        k = 40
        group_terms, group_ps = zip(*swlm.group_top(k, doc_groups[i], max_iter=100))
        corpus_terms, corpus_ps = zip(*nlargest(
            k,
            swlm.get_term_probabilities(swlm.p_corpus).items(),
            itemgetter(1)
        ))
        specific_terms, specific_ps = zip(*nlargest(
            k,
            swlm.get_term_probabilities(swlm.p_specific).items(),
            itemgetter(1)
        ))
        rows = [row for row in zip(
            group_terms, group_ps,
            corpus_terms, corpus_ps,
            specific_terms, specific_ps
        )]
        print(f"SWLM for {author}:")
        print(
            tabulate(rows, headers=(
                'Group term', 'Group p',
                'Corpus term', 'Corpus p',
                'Specific terms', 'Specific p'
            )) + "\n"
        )
