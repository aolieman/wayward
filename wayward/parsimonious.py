#!/usr/bin/env python3

# Copyright 2011-2019 University of Amsterdam
# Author: Lars Buitinck

from __future__ import annotations
# TODO: remove redundant typing imports once PEP 585 is finalized

from collections import defaultdict
from heapq import nlargest
import logging
from operator import itemgetter
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np

from wayward.logsum import logsum


logger = logging.getLogger(__name__)


class ParsimoniousLM:
    """
    Language model for a set of documents.

    Constructing an object of this class fits a background model. The top
    method can then be used to fit document-specific models, also for unseen
    documents (with the same vocabulary as the background corpus).

    Parameters
    ----------
    documents : iterable over iterable of str terms
        All documents that should be included in the corpus model.
    w : float
        Weight of document model (1 - weight of corpus model).
    thresh : int
        Don't include words that occur fewer than `thresh` times.

    Attributes
    ----------
    vocab : dict of term -> int
        Mapping of terms to numeric indices
    p_corpus : array of float
        Log probability of terms in background model (indexed by `vocab`)
    p_document : array of float
        Log probability of terms in the last processed document model
        (indexed by `vocab`)

    Methods
    -------
    top(k, document, ...)
        Fit the document model and retrieve the top `k` terms.
    get_term_probabilities(log_prob_distribution)
        Aligns a term distribution with the vocabulary, and transforms
        the term log probabilities to linear probabilities.

    References
    ----------
    D. Hiemstra, S. Robertson, and H. Zaragoza (2004).
    `Parsimonious Language Models for Information Retrieval
    <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.5806>`_.
    Proc. SIGIR'04.
    """

    def __init__(
            self,
            documents: Iterable[Iterable[str]],
            w: np.floating,
            thresh: int = 0
    ):
        """Collect the vocabulary and fit the background model."""
        logger.info('Building corpus model')

        self.w = w
        self.p_document: Optional[np.ndarray] = None
        # Vocabulary: maps terms to numeric indices
        vocab: Dict[str, int]
        self.vocab = vocab = {}
        # Corpus frequency
        count: Dict[int, int] = defaultdict(int)

        for d in documents:
            for tok in d:
                i = vocab.setdefault(tok, len(vocab))
                count[i] += 1

        cf = np.empty(len(count), dtype=np.double)
        for i, f in count.items():
            cf[i] = f
        rare = (cf < thresh)
        cf -= rare * cf

        try:
            old_error_settings = np.seterr(divide='ignore')

            # lg P(t|C)
            self.p_corpus: np.ndarray = np.log(cf) - np.log(np.sum(cf))
        finally:
            np.seterr(**old_error_settings)

    def top(
            self,
            k: int,
            d: Iterable[str],
            max_iter: int = 50,
            eps: float = 1e-5,
            w: Optional[np.floating] = None
    ) -> List[Tuple[str, float]]:
        """
        Get the top `k` terms of a document `d` and their log probabilities.

        Uses the Expectation Maximization (EM) algorithm to estimate term
        probabilities.

        Parameters
        ----------
        k : int
            Number of top terms to return.
        d : iterable of str terms
            Terms that make up the document.
        max_iter : int, optional
            Maximum number of iterations of EM algorithm to run.
        eps : float, optional
            Epsilon: convergence threshold for EM algorithm.
        w : float, optional
            Weight of document model; overrides value given to __init__

        Returns
        -------
        t_p : list of (str, float)
            Terms and their probabilities in the parsimonious model.
        """

        tf, p_document = self._document_model(d)
        self.p_document = self._EM(tf, p_document, w, max_iter, eps)
        term_probabilities = self.get_term_probabilities(self.p_document)
        return nlargest(k, term_probabilities.items(), itemgetter(1))

    def get_term_probabilities(
            self,
            log_prob_distribution: np.ndarray
    ) -> Dict[str, float]:
        """
        Align a term distribution with the vocabulary, and transform
        the term log probabilities to linear probabilities.

        Parameters
        ----------
        log_prob_distribution : array of float
            Log probability of terms which is indexed by the vocabulary.

        Returns
        -------
        t_p_map : dict of term -> float
            Dictionary of terms and their probabilities in the (sub-)model.
        """
        probabilities = np.exp(log_prob_distribution)
        probabilities[np.isnan(probabilities)] = 0.
        return {t: probabilities[i] for t, i in self.vocab.items()}

    def _document_model(self, d: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build document model.

        Parameters
        ----------
        d : iterable of str terms

        Returns
        -------
        tf : array of float
            Term frequencies
        p_term : array of float
            Term log probabilities

        Initial p_term is 1/n_distinct for terms with non-zero tf,
        0 for terms with 0 tf.
        """

        logger.info('Gathering term probabilities')

        tf = np.zeros(len(self.vocab), dtype=np.double)  # Term frequency

        for tok in d:
            term_id = self.vocab.get(tok)
            if term_id:
                tf[term_id] += 1

        # ignore counts of terms with zero corpus probability
        tf *= np.isfinite(self.p_corpus)

        n_distinct = (tf > 0).sum()

        try:
            old_error_settings = np.seterr(divide='ignore')
            p_term = np.log(tf > 0) - np.log(n_distinct)
        finally:
            np.seterr(**old_error_settings)

        return tf, p_term

    def _EM(
            self,
            tf: np.ndarray,
            p_term: np.ndarray,
            w: Optional[np.floating],
            max_iter: int,
            eps: float
    ) -> np.ndarray:
        """
        Expectation maximization.

        Parameters
        ----------
        tf : array of float
            Term frequencies, as returned by document_model
        p_term : array of float
            Term probabilities, as returned by document_model
        max_iter : int
            Number of iterations to run.
        eps : float
            Epsilon: convergence threshold for EM algorithm.

        Returns
        -------
        p_term : array of float
            A posteriori term probabilities.
        """

        logger.info(f'EM with max_iter={max_iter}, eps={eps}')

        if w is None:
            w = self.w
        w_ = np.log(1 - w)
        w = np.log(w)

        p_corpus = self.p_corpus + w_
        tf = np.log(tf)

        try:
            old_error_settings = np.seterr(divide='ignore')
            p_term = np.asarray(p_term)
            for i in range(1, max_iter + 1):
                # E-step
                p_term += w
                E = tf + p_term - np.logaddexp(p_corpus, p_term)

                # M-step
                new_p_term = E - logsum(E)

                diff = new_p_term - p_term
                p_term = new_p_term
                if (diff[np.isfinite(diff)] < eps).all():
                    logger.info(f'EM: convergence reached after {i} iterations')
                    break
        finally:
            np.seterr(**old_error_settings)

        return p_term
