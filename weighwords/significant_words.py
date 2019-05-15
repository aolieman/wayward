#!/usr/bin/env python3

# Copyright 2019 TinQwise Stamkracht, University of Amsterdam
# Author: Alex Olieman

from __future__ import annotations
# TODO: remove redundant typing imports once PEP 585 is finalized

import logging
from heapq import nlargest
from operator import itemgetter
from typing import Iterable, Optional, Sequence, Tuple, List, Dict

import numpy as np

from weighwords import ParsimoniousLM
from weighwords.logsum import logsum

logger = logging.getLogger(__name__)

InitialLambdas = Tuple[np.floating, np.floating, np.floating]


class SignificantWordsLM(ParsimoniousLM):
    """
    Language model that consists of three sub-models:

    - Corpus model: represents term probabilities in a (large) background collection;
    - Group model: parsimonious term probabilities in a group of documents;
    - Specific model: represents the same group, but is biased towards terms that
      occur with a high frequency in single docs, and a low frequency in others.

    Parameters
    ----------
    documents : iterable over iterable over terms
        All documents that should be included in the corpus model.
    lambdas : 3-tuple of floats
        Weight of corpus, group, and specific models. Will be normalized
        if the weights in the tuple don't sum to one.
    thresh : int
        Don't include words that occur fewer than `thresh` times.

    Attributes
    ----------
    vocab : dict of term -> int
        Mapping of terms to numeric indices
    p_corpus : array of float
        Log probability of terms in background model (indexed by `vocab`)
    p_group : array of float
        Log probability of terms in background model (indexed by `vocab`)
    p_specific : array of float
        Log probability of terms in background model (indexed by `vocab`)
    lambda_corpus : array of float
        Log probability (weight) of corpus model for documents
    lambda_group : array of float
        Log probability (weight) of group model for documents
    lambda_specific : array of float
        Log probability (weight) of specific model for documents

    Methods
    -------
    fit_parsimonious_group(document_group, ...)
        Estimates a document group model, parsimonized against the corpus
        and specific models. The documents may be unseen, but terms that
        are not in the vocabulary will be ignored.
    group_top(k, document_group, ...)
        Shortcut to fit the group model and retrieve the top `k` terms.
    get_term_probabilities(log_prob_distribution)
        Aligns a term distribution with the vocabulary, and transforms
        the term log probabilities to linear probabilities.

    See Also
    --------
    parsimonious.ParsimoniousLM : one-sided parsimonious model
    """

    def __init__(
            self,
            documents: Iterable[Iterable[str]],
            lambdas: InitialLambdas,
            thresh: int = 0
    ):
        """Collect the vocabulary and fit the background model."""
        self.initial_lambdas = self.normalize_lambdas(lambdas)
        super().__init__(documents, self.initial_lambdas[1], thresh=thresh)
        self.lambda_corpus: Optional[np.ndarray] = None
        self.lambda_group: Optional[np.ndarray] = None
        self.lambda_specific: Optional[np.ndarray] = None
        self.p_group: Optional[np.ndarray] = None
        self.p_specific: Optional[np.ndarray] = None
        self.fix_lambdas = False

    def group_top(
            self,
            k: int,
            document_group: Iterable[Iterable[str]],
            **kwargs
    ) -> List[Tuple[str, float]]:
        term_probabilities = self.fit_parsimonious_group(document_group, **kwargs)
        return nlargest(k, term_probabilities.items(), itemgetter(1))

    def fit_parsimonious_group(
            self,
            document_group: Iterable[Iterable[str]],
            max_iter: int = 50,
            eps: float = 1e-5,
            lambdas: Optional[InitialLambdas] = None,
            fix_lambdas: bool = False,
            parsimonize_specific: bool = False,
            post_parsimonize: bool = False
    ) -> Dict[str, float]:
        if lambdas is None:
            lambdas = self.initial_lambdas
        else:
            lambdas = self.normalize_lambdas(lambdas)

        self.fix_lambdas = fix_lambdas

        document_models = [
            self._document_model(doc)
            for doc in document_group
        ]
        del document_group

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
        if parsimonize_specific:
            p_specific = self._EM(group_tf, p_specific, self.w, max_iter, eps)

        self.p_specific = p_specific

        weights_shape = len(document_models)
        general_w, group_w, specific_w = np.log(lambdas)
        self.lambda_corpus = np.full(weights_shape, general_w, dtype=np.double)
        self.lambda_specific = np.full(weights_shape, specific_w, dtype=np.double)
        self.lambda_group = np.full(weights_shape, group_w, dtype=np.double)
        logger.info(
            f'Lambdas initialized to: Corpus={lambdas[0]}, '
            f'Group={lambdas[1]}, Specific={lambdas[2]}'
        )
        self.p_group = self._estimate(
            p_group, p_specific, doc_term_frequencies, max_iter, eps
        )
        if post_parsimonize:
            self.p_group = self._EM(group_tf, self.p_group, self.w, max_iter, eps)

        if self.fix_lambdas is False:
            logger.info(
                f'Final lambdas (mean): Corpus={np.mean(np.exp(self.lambda_corpus))}, '
                f'Group={np.mean(np.exp(self.lambda_group))}, '
                f'Specific={np.mean(np.exp(self.lambda_specific))}'
            )
        return self.get_term_probabilities(self.p_group)

    def get_term_probabilities(
            self,
            log_prob_distribution: np.ndarray
    ) -> Dict[str, float]:
        probabilities = np.exp(log_prob_distribution)
        probabilities[np.isnan(probabilities)] = 0.
        return {t: probabilities[i] for t, i in self.vocab.items()}

    def _estimate(
            self,
            p_group: np.ndarray,
            p_specific: np.ndarray,
            doc_tf: Sequence[np.ndarray],
            max_iter: int,
            eps: float
    ) -> np.ndarray:
        try:
            old_error_settings = np.seterr(divide='ignore')
            log_doc_tf = np.log(doc_tf)
            for i in range(1, 1 + max_iter):
                expectation = self._e_step(p_group, p_specific)
                new_p_group = self._m_step(expectation, log_doc_tf)

                diff = new_p_group - p_group
                p_group = new_p_group
                if (diff[np.isfinite(diff)] < eps).all():
                    logger.info(f'EM: convergence reached after {i} iterations')
                    break
        finally:
            np.seterr(**old_error_settings)

        return p_group

    def _e_step(
            self,
            p_group: np.ndarray,
            p_specific: np.ndarray
    ) -> Dict[str, np.ndarray]:
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
        out = {
            'corpus': corpus_numerator - denominator,
            'specific': specific_numerator - denominator,
            'group': group_numerator - denominator
        }
        # prevent NaNs from causing downstream errors
        for v in out.values():
            v[np.isnan(v)] = np.NINF

        return out

    def _m_step(
            self,
            expectation: Dict[str, np.ndarray],
            log_doc_tf: Sequence[np.ndarray]
    ) -> np.ndarray:
        term_weighted_group = log_doc_tf + expectation['group']
        group_numerator = logsum(term_weighted_group)
        p_group = group_numerator - logsum(group_numerator)

        if self.fix_lambdas is False:
            # estimate lambdas
            corpus_numerator = logsum(
                np.transpose(log_doc_tf + expectation['corpus'])
            )
            specific_numerator = logsum(
                np.transpose(log_doc_tf + expectation['specific'])
            )
            group_numerator = logsum(np.transpose(term_weighted_group))
            denominator = logsum(
                np.asarray([corpus_numerator, specific_numerator, group_numerator])
            )
            self.lambda_corpus = corpus_numerator - denominator
            self.lambda_specific = specific_numerator - denominator
            self.lambda_group = group_numerator - denominator

        return p_group

    @staticmethod
    def _group_model(
            document_term_frequencies: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        group_tf = np.array(document_term_frequencies).sum(axis=0)

        try:
            old_error_settings = np.seterr(divide='ignore')
            p_group = np.log(group_tf) - np.log(np.sum(group_tf))
        finally:
            np.seterr(**old_error_settings)

        return group_tf, p_group

    @staticmethod
    def _specific_model(
            document_term_probabilities: Sequence[np.ndarray]
    ) -> np.ndarray:
        # complement events: 1 - p
        complements = [
            np.log1p(-np.exp(p_doc))
            for p_doc in document_term_probabilities
        ]
        # probability of term to be important in one doc, and not others
        complement_products = np.array([
            dlm + complement
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
                    np.count_nonzero(complement_products > np.NINF, axis=0)
                )
            )
            # prevent NaNs from causing downstream errors
            p_specific[np.isnan(p_specific)] = np.NINF
        finally:
            np.seterr(**old_error_settings)

        return p_specific

    @staticmethod
    def normalize_lambdas(lambdas: InitialLambdas) -> InitialLambdas:
        assert len(lambdas) == 3, f'lambdas should be a 3-tuple, not {lambdas}'
        lambda_sum = sum(lambdas)
        if abs(lambda_sum - 1) > 1e-10:
            lambdas = tuple(
                w / lambda_sum
                for w in lambdas
            )
        return lambdas
