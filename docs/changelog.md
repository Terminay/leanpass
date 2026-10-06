---
id: changelog
title: Changelog
sidebar_label: Changelog
---

# Changelog

All notable changes to LeanPass are documented here. Release versions and dates
follow the published releases on [PyPI](https://pypi.org/project/leanpass/) and
[GitHub](https://github.com/Terminay/leanpass/releases).

The format is based on [keep-a-changelog.org](https://keepachangelog.org/) and uses semantic versioning.

## [0.1.13] - 2026-10-05

### Changed

- Updated documentation-site JavaScript dependencies (`qs` and `sharp`); no Python API changes

## [0.1.12] - 2026-09-20

### Fixed

- Added numerical-stability guards to logarithm, exponential, sigmoid, and softmax operations to avoid domain errors, overflow, and invalid normalization

## [0.1.11] - 2026-09-19

### Changed

- Updated documentation-site JavaScript dependencies; no Python API changes

## [0.1.10] - 2026-09-10

### Changed

- Updated documentation-site JavaScript dependencies; no Python API changes

## [0.1.9] - 2026-08-28

### Added

- Usable operation registry for registering, decorating, and retrieving built-in or plugin operations

## [0.1.8] - 2026-08-25

### Added

- Explicit `ZeroDivisionError` when dividing a tensor by zero
- Contributor acknowledgments in the README

## [0.1.7] - 2026-08-09

### Added

- `optim.SGD` now supports momentum and L2 weight decay
- `optim.Adam` now supports optional L2 weight decay
- `nn.Dropout` layer for training-time regularization
- `Tensor.clip(a_min, a_max)` / `Tensor.clamp(min, max)` value clamping
- Basic indexing support via `Tensor[...]` with autodiff propagation

## [0.1.6] - 2026-08-09

### Added

- Issue-management CI workflow
- GitHub Actions workflow for bumping package versions and creating tags

### Fixed

- Store reduction parents as tuples in `sum()` and `mean()`, fixing graph evaluation failures

## [0.1.5] - 2026-07-30

### Added

- Cloudflare Workers configuration for deploying the documentation site
- Setuptools build configuration for distributing the Python package

### Fixed

- Repaired the incomplete `0.1.4` distribution on PyPI

## [0.1.4] - 2026-07-28

### Added

- Complete documentation site with Docusaurus
- XOR classification guide
- Computation graph visualization assets

### Changed

- Fixed documentation image paths for production
- Completed documentation placeholders

## [0.1.3] - 2026-07-25

### Added

- `Tensor.tanh()` activation
- `Tensor.leaky_relu()` activation
- `Tensor.gelu()` activation

## [0.1.2] - 2026-07-25

### Added

- `cross_entropy_loss` for categorical classification training
- `binary_cross_entropy_loss` for binary label training

## [0.1.1] - 2026-07-25

### Added

- Manually triggerable `Publish to PyPI` GitHub Actions workflow

## [0.1.0] - 2026-07-23

### Added

- Initial PyPI release of LeanPass, including the `Tensor` autodiff core, neural-network layers, and optimizers
- GitHub Actions workflows for publishing releases to GitHub and PyPI
- Initial documentation, issue templates, and contribution files

### Fixed

- Fixed README rendering and workflow details
