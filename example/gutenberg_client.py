import functools
import itertools
import logging
import os
import pickle
import re
from collections import namedtuple
from heapq import nlargest
from operator import itemgetter, attrgetter

import nltk
import numpy as np
import spacy
from gensim.models import Phrases
from gutenberg._domain_model.exceptions import UnknownDownloadUriException
from gutenberg.acquire import load_etext
from gutenberg.acquire.text import _TEXT_CACHE
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts, get_metadata
from spacy.tokenizer import Tokenizer
from tabulate import tabulate

from weighwords import ParsimoniousLM, SignificantWordsLM
from weighwords.specific_term_estimators import me_up_to_40_docs

authors = [
    'Carroll, Lewis',
    'Melville, Herman',
    'Doyle, Arthur Conan',
    'Wells, H. G. (Herbert George)',
]
# authors = [
#     'Couperus, Louis',
#     'Eeden, Frederik van',
#     'Gorter, Herman',
#     'Hildebrand',
#     'Verwey, Albert',
#     'Deyssel, Lodewijk van',
# ]
language = ('en', 'english')
# language = ('nl', 'dutch')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

top_k = 40  # How many terms per book to retrieve

# en_weights = spacy.load('en_core_web_sm')
# nn_tokenizer = Tokenizer(en_weights.vocab)
sent_detector = nltk.data.load(f'tokenizers/punkt/{language[1]}.pickle')
token_re = re.compile(r'\w{2,}')

Document = namedtuple('Document', [
    'pk',
    'title',
    'text',
    'terms'
])


def spacy_tokenize(text):
    return [
        w.lower_
        for w in nn_tokenizer(text)
        if token_re.search(w.text)
    ]


def nltk_tokenize(text):
    sents = sent_detector.sentences_from_text(text)
    strip_chars = ' _*'
    if language[0] == 'nl':
        strip_chars += "'"

    for s in sents:
        yield [
            w.strip(strip_chars).lower()
            for w in nltk.tokenize.word_tokenize(s)
            if token_re.search(w)
        ]


def read_book(gutenberg_id, fetch_missing=True):
    cached_fp = os.path.join(_TEXT_CACHE, '{0}.txt.gz'.format(gutenberg_id))
    if not (fetch_missing or os.path.exists(cached_fp)):
        return None

    etext = load_etext(gutenberg_id, mirror="http://eremita.di.uminho.pt/gutenberg/")
    return strip_headers(etext).strip()


@functools.lru_cache(maxsize=10)
def language_filter(lang_code):
    return get_etexts('language', lang_code)


def get_author_library(author_name, refresh_cache=False, for_plm=False):
    logger.info(f'Loading library for {author_name}')

    cached_library = get_cached_library(author_name)
    if cached_library and not refresh_cache:
        library, work_terms = cached_library
    else:
        work_ids, work_texts = [], []
        etext_ids = get_etexts('author', author_name) & language_filter(language[0])
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
        work_sentences = [list(nltk_tokenize(d)) for d in work_texts]
        bigram_model = Phrases(
            itertools.chain.from_iterable(work_sentences),
            threshold=12
        )
        trigram_model = Phrases(
            bigram_model[itertools.chain.from_iterable(work_sentences)],
            threshold=12
        )
        work_terms = [
            list(
                itertools.chain.from_iterable(
                    trigram_model[bigram_model[work]]
                )
            )
            for work in work_sentences
        ]

        library = {
            pk: Document(
                pk,
                *values
            )
            for pk, *values in zip(
                work_ids,
                work_titles,
                work_texts,
                work_terms,
            )
        }
        cache_library(
            author_name,
            (library, work_terms)
        )

    if for_plm:
        plm = ParsimoniousLM(work_terms, w=.01)
        return library, plm

    return library


def get_cached_library(author_name):
    try:
        with open(f'{author_name}.library', 'rb') as f:
            return pickle.load(f)
    except IOError:
        return None


def cache_library(author_name, library_tuple):
    with open(f'{author_name}.library', 'wb') as f:
        pickle.dump(library_tuple, f)


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
    #     book_vs_author(*get_author_library(author, for_plm=True))

    libraries = {
        author: nlargest(
            20,  # n books with highest term counts
            get_author_library(author, refresh_cache=False).values(),
            key=lambda d: len(d.terms)
        )
        for author in authors
    }
    for author, library in libraries.items():
        titles = set()
        docs = []
        for doc in library:
            if doc.title not in titles:
                docs.append(doc)
                titles.add(doc.title)

            if len(docs) >= 15:
                break

        print(f'Selected {len(docs)} works by {author}')
        libraries[author] = docs

    corpus = itertools.chain.from_iterable(
        map(attrgetter('terms'), library)
        for library in libraries.values()
    )
    swlm = SignificantWordsLM(corpus, lambdas=(.9, .01, .09), thresh=5)
    for author, library in libraries.items():
        doc_group = map(attrgetter('terms'), library)
        group_terms, group_ps = zip(
            *swlm.group_top(
                top_k,
                doc_group,
                max_iter=100,
                fix_lambdas=True,
                parsimonize_specific=False,
                specific_estimator=me_up_to_40_docs
            )
        )
        corpus_terms, corpus_ps = zip(*nlargest(
            top_k,
            swlm.get_term_probabilities(swlm.p_corpus).items(),
            itemgetter(1)
        ))
        specific_terms, specific_ps = zip(*nlargest(
            top_k,
            swlm.get_term_probabilities(swlm.p_specific).items(),
            itemgetter(1)
        ))
        rows = [row for row in zip(
            group_terms, group_ps,
            corpus_terms, corpus_ps,
            specific_terms, specific_ps
        )]
        print(f"SWLM for {author}:", flush=True)
        print(
            tabulate(rows, headers=(
                'Group term', 'Group p',
                'Corpus term', 'Corpus p',
                'Specific terms', 'Specific p'
            )) + "\n",
            flush=True
        )

        lambda_group = np.exp(swlm.lambda_group)
        lambda_corpus = np.exp(swlm.lambda_corpus)
        lambda_specific = np.exp(swlm.lambda_specific)
        if len(lambda_group) == 1:
            lambda_group = len(library) * list(lambda_group)
            lambda_corpus = len(library) * list(lambda_corpus)
            lambda_specific = len(library) * list(lambda_specific)

        doc_lambdas = zip(
            map(attrgetter('pk'), library),
            map(lambda d: d.title[:50], library),
            lambda_group,
            lambda_corpus,
            lambda_specific
        )
        print(
            tabulate(
                sorted(doc_lambdas, key=itemgetter(2), reverse=True),
                headers=(
                    'ID', 'Title', 'Group L', 'Corpus L', 'Specific L'
                )
            ) + "\n",
            flush=True
        )
