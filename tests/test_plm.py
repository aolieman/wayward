import numpy as np
import pytest

from weighwords import ParsimoniousLM


def test_document_model(number_corpus, uniform_doc):
    plm = ParsimoniousLM([number_corpus], w=0.1, thresh=3)
    tf, p_term = plm._document_model(uniform_doc)
    assert (tf[:2] == 0).all(), \
        "Terms with a corpus frequency < thresh should not be counted"
    assert tf.sum() == 3, f"Expected tf.sum() to be 3, got {tf.sum()} instead"
    linear_p_term = np.exp(p_term)
    assert (linear_p_term[2:].sum() - 1) < 1e-10, \
        f"All probability mass should be on the last 3 terms, got {linear_p_term} instead"


def test_document_model_out_of_vocabulary(number_corpus):
    plm = ParsimoniousLM([number_corpus], w=0.1)
    doc = ['two', 'or', 'three', 'unseen', 'words']
    tf, p_term = plm._document_model(doc)
    assert tf.sum() == 2, f"Unseen words should be ignored, got {tf} instead"


@pytest.fixture(scope="module")
def uniform_doc():
    return ['one', 'two', 'three', 'four', 'five']


@pytest.fixture(scope="module")
def number_corpus():
    return [
        'one',
        'two', 'two',
        'three', 'three', 'three',
        'four', 'four', 'four', 'four',
        'five', 'five', 'five', 'five', 'five'
    ]
