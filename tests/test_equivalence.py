from itertools import chain

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
