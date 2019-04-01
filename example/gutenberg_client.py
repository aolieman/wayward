import logging
import re

import nltk
import numpy as np
import spacy
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from spacy.tokenizer import Tokenizer

from weighwords import ParsimoniousLM

books = [
    ('Oliver Twist',       730),
    ('David Copperfield',  766),
    ('Great Expectations', 1400),
]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

top_k = 20  # How many terms per book to retrieve

en_weights = spacy.load('en_core_web_sm')
nn_tokenizer = Tokenizer(en_weights.vocab)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
token_re = re.compile(r'\w{2,}')


def spacy_tokenize(text):
    return [
        w.lower_
        for w in nn_tokenizer(text)
        if token_re.match(w.text)
    ]


def nltk_tokenize(text):
    sents = sent_detector.sentences_from_text(text)
    return [
        w.strip().lower()
        for s in sents
        for w in nltk.tokenize.word_tokenize(s)
        if token_re.match(w)
    ]


def read_book(gutenberg_id):
    return strip_headers(
        load_etext(gutenberg_id)
    ).strip()


library = {
    title: read_book(pk)
    for title, pk in books
}
spacy_docterms = [spacy_tokenize(d) for d in library.values()]
spacy_model = ParsimoniousLM(spacy_docterms, w=.01)
nltk_docterms = [nltk_tokenize(d) for d in library.values()]
nltk_model = ParsimoniousLM(nltk_docterms, w=.01)


def book_vs_author(model, docterms):
    for i, item in enumerate(library.items()):
        title, _ = item
        print("Top %d words in %s:" % (top_k, title))
        for term, p in model.top(top_k, docterms[i]):
            print("    %s %.4f" % (term, np.exp(p)))
        print("")


if __name__ == "__main__":
    book_vs_author(spacy_model, spacy_docterms)
    book_vs_author(nltk_model, nltk_docterms)
