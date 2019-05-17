#!/usr/bin/env python3

# Find terms that distinguish various novels by Charles Dickens.
# Note: if the w parameter is set wisely, no stop list is needed.
import gzip
import logging
import math
import re
from itertools import zip_longest

import numpy as np

from weighwords import ParsimoniousLM, SignificantWordsLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

top_k = 20  # How many terms per book to retrieve

books = [
    ('Oliver Twist',       '730'),
    ('David Copperfield',  '766'),
    ('Great Expectations', '1400'),
]

startbook = """*** START OF THIS PROJECT GUTENBERG EBOOK """
endbook = """*** END OF THIS PROJECT GUTENBERG EBOOK """


def read_book(title, num):
    """Returns generator over words in book num"""

    logger.info(f"Fetching terms from {title}")
    path = f"{num}.txt.utf8.gz"
    in_book = False
    for ln in gzip.open(path, 'rt', encoding='utf8'):
        if in_book and ln.startswith(endbook):
            break
        elif in_book:
            for w in re.sub(r"[.,:;!?\"'‘’]", " ", ln).lower().split():
                yield w
        elif ln.startswith(startbook):
            in_book = True


def grouper(iterable, n, filler=None):
    """Source: https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=filler)


book_contents = [(title, list(read_book(title, num))) for title, num in books]
corpus = [terms for title, terms in book_contents]

plm = ParsimoniousLM(corpus, w=.01)
swlm = SignificantWordsLM(corpus, lambdas=(.9, .01, .09))

for title, terms in book_contents:
    plm_top = plm.top(top_k, terms)
    swlm_top = swlm.group_top(
        top_k,
        grouper(terms, math.ceil(len(terms) / 10)),
        fix_lambdas=True,
    )
    print(f"\nTop {top_k} words in {title}:")
    print(f"\n{'PLM term':<16} {'PLM p':<12} {'SWLM term':<16} {'SWLM p':<6}")
    for (plm_t, plm_p), (swlm_t, swlm_p) in zip(plm_top, swlm_top):
        print(f"{plm_t:<16} {np.exp(plm_p):<12.4f} {swlm_t:<16} {swlm_p:.4f}")
    print("")



