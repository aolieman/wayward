import logging

from weighwords import SignificantWordsLM

logging.basicConfig(level=logging.INFO)


def test_model_fit_fixed(number_corpus, uniform_doc):
    swlm = SignificantWordsLM([uniform_doc], lambdas=(1/3, 1/3, 1/3))
    doc_group = [l + r for l, r in zip(number_corpus, reversed(number_corpus))]
    term_probs = swlm.fit_parsimonious_group(doc_group, fix_lambdas=True)
    expected_probs = {
        "one": 0.0,
        "two": 0.12373,
        "three": 2e-5,
        "four": 0.50303,
        "five": 0.37322,
    }
    for term, p in expected_probs.items():
        diff = abs(term_probs[term] - p)
        assert diff < 1e-5, f"P({term}) != {p} with difference {diff}"


def test_model_fit_shifty(number_corpus, uniform_doc):
    swlm = SignificantWordsLM([uniform_doc], lambdas=(1/3, 1/3, 1/3))
    doc_group = [l + r for l, r in zip(number_corpus, reversed(number_corpus))]
    term_probs = swlm.fit_parsimonious_group(doc_group, fix_lambdas=False)
    expected_probs = {
        "one": 0.0,
        "two": 0.33322,
        "three": 0.0,
        "four": 0.66678,
        "five": 0.0,
    }
    for term, p in expected_probs.items():
        diff = abs(term_probs[term] - p)
        assert diff < 1e-5, f"P({term}) != {p} with difference {diff}"
