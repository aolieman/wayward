import re
from itertools import chain

import pytest

from weighwords import ParsimoniousLM, SignificantWordsLM
from weighwords.logsum import logsum


def test_model_equivalence(shakespeare_quotes):
    weight = .1
    plm = ParsimoniousLM(shakespeare_quotes, w=weight)
    # initialize SWLM with weights that make it equivalent to PLM
    swlm = SignificantWordsLM(
        shakespeare_quotes,
        lambdas=(1 - weight, weight, 0.)
    )
    plm_terms, swlm_terms = fit_models(plm, swlm, shakespeare_quotes)

    assert plm_terms == swlm_terms, 'PLM and SWLM are not functionally equivalent'


def test_model_non_equivalence(shakespeare_quotes):
    weight = .1
    plm = ParsimoniousLM(shakespeare_quotes, w=weight)
    # initialize SWLM with weights that make it non-equivalent to PLM
    swlm = SignificantWordsLM(
        shakespeare_quotes,
        lambdas=(1 - 2 * weight, weight, weight)
    )
    plm_terms, swlm_terms = fit_models(plm, swlm, shakespeare_quotes)

    assert plm_terms != swlm_terms, 'PLM and SWLM should not be functionally equivalent'


@pytest.fixture(scope="module")
def shakespeare_quotes():
    quotes = [
        "Love all, trust a few, Do wrong to none",
        "But love that comes too late, "
        "Like a remorseful pardon slowly carried, "
        "To the great sender turns a sour offence.",
        "If thou remember'st not the slightest folly "
        "That ever love did make thee run into, "
        "Thou hast not lov'd.",
        "We that are true lovers run into strange capers; "
        "but as all is mortal in nature, "
        "so is all nature in love mortal in folly.",
        "But are you so much in love as your rhymes speak? "
        "Neither rhyme nor reason can express how much.",
        "A lover's eyes will gaze an eagle blind. "
        "A lover's ear will hear the lowest sound.",
    ]
    return [
        re.sub(r"[.,:;!?\"‘’]|'s\b", " ", quote).lower().split()
        for quote in quotes
    ]


def get_p_corpus(language_model):
    p_corpus = language_model.p_corpus.copy()
    vocab = language_model.vocab
    term_tiers = [
        (1.5, ['love', 'folly', "lov'd", 'lovers', 'lover']),
        (1.3, ['trust', 'remorseful', 'sour', 'offence', 'gaze']),
        (1.1, ["remember'st", 'capers', 'rhyme', 'rhymes', 'eagle']),
    ]
    for multiplier, terms in term_tiers:
        for t in terms:
            p_corpus[vocab[t]] *= multiplier

    return p_corpus - logsum(p_corpus)


def fit_models(plm, swlm, docs):
    # artificially reduce the corpus probability of selected terms
    plm.p_corpus = swlm.p_corpus = get_p_corpus(plm)

    top_k = 15
    plm_top = plm.top(top_k, chain(*docs))
    swlm_top = swlm.group_top(top_k, docs, fix_lambdas=True)
    plm_terms = [term for term, log_prob in plm_top]
    swlm_terms = [term for term, prob in swlm_top]

    return plm_terms, swlm_terms
