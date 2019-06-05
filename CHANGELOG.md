# Changelog
All notable changes to this project should be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2019-06-05

### Added

- This changelog.

### Changed

- Explicitly specified the readme in `pyproject.toml`.
- Updated install instructions for Poetry.


## [0.3.0] - 2019-06-04

### Added

- Significant Words Language Model.
- Pluggable specific terms estimator.
- Tests for PLM document model.
- Tests for SWLM model fit.
- Tests for model (non-)equivalence between PLM and SWLM.
- SWLM example in `exmaple/dickens.py`.
- Usage examples in README.
- Type hints in function annotations.

### Changed

- Renamed package to Wayward.
- Replaced `setup.py` with `pyproject.toml`.
- `ParsimoniousLM.top()` now returns linear probabilities instead of log-probabilities.

### Removed

- Dropped python 2.7 compatibility in favor of ^3.7.

### Fixed

- `KeyError` when out-of-vocabulary terms occurred in a document.

## [0.2.x] - 2011-11-13 to 2013-04-18

The WeighWords version from which Wayward was forked.

Some commits have been put on the master branch after bumping the version to 0.2.
Since there is no git tag to pin down what's part of 0.2, I've mentioned both the
version bump date, and the date of the latest commit that we use here.
