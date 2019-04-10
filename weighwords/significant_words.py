import logging
from heapq import nlargest
from operator import itemgetter

import numpy as np

from weighwords import ParsimoniousLM
from weighwords.logsum import logsum

logger = logging.getLogger(__name__)


class SignificantWordsLM(ParsimoniousLM):
    """Language model for a set of documents.

    Constructing an object of this class fits a background model. The top
    method can then be used to fit document-specific models, also for unseen
    documents (with the same vocabulary as the background corpus).

    Parameters
    ----------
    documents : iterable over iterable over terms
    w : float
        Weight of document model (1 - weight of corpus model)
    thresh : int
        Don't include words that occur < thresh times

    Attributes
    ----------
    vocab : dict of term -> int
        Mapping of terms to numeric indices
    p_corpus : array of float
        Log prob of terms
    """
    def __init__(self, documents, w, thresh=0):
        super().__init__(documents, w, thresh=thresh)
        self.lambda_corpus = None
        self.lambda_group = None
        self.lambda_specific = None
        self.p_group = None
        self.p_specific = None

    def group_top(self, k, document_group, **kwargs):
        term_probabilities = self.fit_parsimonious_group(document_group, **kwargs)
        return nlargest(k, term_probabilities.items(), itemgetter(1))

    def fit_parsimonious_group(self, document_group, max_iter=50, eps=1e-5, w=None):
        if w is None:
            w = self.w
        assert 0 < w < 1, f"invalid w={w}; `w` needs a value between 0.0 and 1.0"

        document_models = [
            self._document_model(doc)
            for doc in document_group
        ]
        doc_term_frequencies = [tf for tf, _ in document_models]
        group_tf, p_group = self._group_model(
            doc_term_frequencies
        )
        try:
            old_error_settings = np.seterr(divide='ignore')
            doc_term_probs = [
                np.log(tf) - np.log(np.sum(tf))
                for tf in doc_term_frequencies
            ]
        finally:
            np.seterr(**old_error_settings)

        p_specific = self._specific_model(doc_term_probs)

        general_w = specific_w = np.log(0.5 * (1 - w))
        group_w = np.log(w)
        weights_shape = len(document_group)
        self.lambda_corpus = np.full(weights_shape, general_w, dtype=np.float)
        self.lambda_specific = np.full(weights_shape, specific_w, dtype=np.float)
        self.lambda_group = np.full(weights_shape, group_w, dtype=np.float)
        logger.info(
            f'Lambdas initialized to: Corpus={np.exp(general_w)}, '
            f'Group={w}, Specific={np.exp(specific_w)}'
        )
        self.p_group = self._estimate(p_group, p_specific, doc_term_frequencies, max_iter, eps)
        self.p_specific = p_specific

        exp_p_group = np.exp(p_group)

        return {t: exp_p_group[i] for t, i in self.vocab.items()}

    def _estimate(self, p_group, p_specific, doc_tf, max_iter, eps):
        try:
            old_error_settings = np.seterr(divide='ignore')
            log_doc_tf = np.log(doc_tf)
            for i in range(1, 1 + max_iter):
                expectation = self._e_step(p_group, p_specific)
                new_p_group = self._m_step(expectation, log_doc_tf)

                diff = new_p_group - p_group
                p_group = new_p_group
                if (diff < eps).all():
                    logger.info(f'EM: convergence reached after {i} iterations')
                    break
        finally:
            np.seterr(**old_error_settings)

        return p_group

    def _e_step(self, p_group, p_specific):
        corpus_numerator = np.add.outer(self.lambda_corpus, self.p_corpus)
        specific_numerator = np.add.outer(self.lambda_specific, p_specific)
        group_numerator = np.add.outer(self.lambda_group, p_group)
        denominator = [
            logsum(np.asarray([sp_corpus, sp_corpus, sp_specific]))
            for sp_corpus, sp_corpus, sp_specific in zip(
                corpus_numerator,
                specific_numerator,
                group_numerator
            )
        ]
        return {
            'corpus': corpus_numerator - denominator,
            'specific': specific_numerator - denominator,
            'group': group_numerator - denominator
        }

    def _m_step(self, expectation, log_doc_tf):
        group_numerator = logsum(log_doc_tf + expectation['group'])
        p_group = group_numerator - logsum(group_numerator)
        # TODO: estimate lambdas
        return p_group

    @staticmethod
    def _group_model(document_term_frequencies):
        group_tf = np.array(document_term_frequencies).sum(axis=0)

        try:
            old_error_settings = np.seterr(divide='ignore')
            p_group = np.log(group_tf) - np.log(np.sum(group_tf))
        finally:
            np.seterr(**old_error_settings)

        return group_tf, p_group

    @staticmethod
    def _specific_model(document_term_probabilities):
        # complement events: 1 - p
        complements = [
            np.log1p(-np.exp(p_doc))
            for p_doc in document_term_probabilities
        ]
        # probability of term to be important in one doc, and not others
        complement_products = np.array([
            document_term_probabilities[i] + complement
            for i, dlm in enumerate(document_term_probabilities)
            for j, complement in enumerate(complements)
            if i != j
        ])

        try:
            old_error_settings = np.seterr(divide='ignore')
            # marginalize over all documents
            p_specific = (
                logsum(complement_products)
                - np.log(
                    np.count_nonzero(complement_products > np.log(0), axis=0)
                )
            )
        finally:
            np.seterr(**old_error_settings)

        return p_specific
