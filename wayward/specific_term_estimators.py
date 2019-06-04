#!/usr/bin/env python3

# Copyright 2019 TinQwise Stamkracht, University of Amsterdam
# Author: Alex Olieman

from __future__ import annotations
# TODO: remove redundant typing imports once PEP 585 is finalized

import functools
import logging
from typing import Sequence, Callable

import numpy as np
from wayward.logsum import logsum

logger = logging.getLogger(__name__)

SpecificTermEstimator = Callable[[Sequence[np.ndarray]], np.ndarray]


class RequiresMultipleDocuments(Exception):
    pass


def requires_multiple_docs(estimator_func: SpecificTermEstimator):
    """
    Do not let the decorated function be called with fewer than two docs.

    Parameters
    ----------
    estimator_func : SpecificTermEstimator

    Raises
    ------
    RequiresMultipleDocuments

    Returns
    -------
    decorated_func : SpecificTermEstimator
    """
    @functools.wraps(estimator_func)
    def wrapper_func(document_term_frequencies):
        if len(document_term_frequencies) < 2:
            raise RequiresMultipleDocuments

        return estimator_func(document_term_frequencies)

    return wrapper_func


@requires_multiple_docs
def mutual_exclusion(
        document_term_frequencies: Sequence[np.ndarray]
) -> np.ndarray:
    """Estimate the fixed specific model with the mutual exclusion method."""
    doc_term_probs = [
        np.log(tf) - np.log(np.sum(tf))
        for tf in document_term_frequencies
    ]
    # complement events: 1 - p
    complements = [
        np.log1p(-np.exp(p_doc))
        for p_doc in doc_term_probs
    ]
    # probability of term to be important in one doc, and not others
    complement_products = np.array([
        dlm + complement
        for i, dlm in enumerate(doc_term_probs)
        for j, complement in enumerate(complements)
        if i != j
    ])
    # marginalize over all documents
    p_specific = (
        logsum(complement_products)
        - np.log(
            np.count_nonzero(complement_products > np.NINF, axis=0)
        )
    )
    # prevent NaNs from causing downstream errors
    p_specific[np.isnan(p_specific)] = np.NINF

    return p_specific


@requires_multiple_docs
def inverse_doc_frequency(
        document_term_frequencies: Sequence[np.ndarray]
) -> np.ndarray:
    """Estimate the fixed specific model with the inverse doc frequency method."""
    idf = 1 / np.count_nonzero(document_term_frequencies, axis=0)
    idf[~np.isfinite(idf)] = 0.

    # calculate normalized idf as log-probabilities
    p_specific = np.log(idf) - np.log(np.sum(idf))

    return p_specific


def idf_fallback_for_many_docs(
        document_term_frequencies: Sequence[np.ndarray],
        primary_estimator: SpecificTermEstimator,
        fallback_thresh: int
):
    if len(document_term_frequencies) < fallback_thresh:
        estimator_func = primary_estimator
    else:
        estimator_func = inverse_doc_frequency
        logger.warning(
            f'Estimator got more than {fallback_thresh} docs:'
            ' falling back to IDF for the current doc group.'
        )

    return estimator_func(document_term_frequencies)


me_up_to_40_docs = functools.partial(
    idf_fallback_for_many_docs,
    primary_estimator=mutual_exclusion,
    fallback_thresh=40
)
