---
id: changelog
title: Changelog
sidebar_label: Changelog
---

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [keep-a-changelog.org](https://keepachangelog.org/) and uses semantic versioning.

## [0.1.7] - 2026-08-09

### Added

- `optim.SGD` now supports momentum and L2 weight decay
- `optim.Adam` now supports optional L2 weight decay
- `nn.Dropout` layer for training-time regularization
- `Tensor.clip(a_min, a_max)` / `Tensor.clamp(min, max)` value clamping
- Basic indexing support via `Tensor[...]` with autodiff propagation

### Changed

- Updated documentation and README to reflect the new optimizer and tensor APIs

## [0.1.6] - 2026-08-09

### Added

- new CI workflow for issues
- GitHub Actions workflow to bump version

### Fixed

- Store `_prev` as a tuple in sum() and mean()
- Bump the npm_and_yarn group across 1 directory with 4 updates

## [0.1.5] - 2026-07-30

### Added

- Cloudflare Workers configuration by @cloudflare-workers-and-pages[bot]

### Fixed

- Repaired `0.1.4` version on PyPI not having all files 

## [0.1.4] - 2026-07-28

### Added

- Complete documentation site with Docusaurus
- XOR classification guide
- Computation graph visualization assets

### Changed

- Updated package version to `0.1.4`
- Fixed documentation image paths for production
- Completed all documentation placeholders

## [0.1.3] - 2026-07-25

### Added

- `Tensor.tanh()` activation
- `Tensor.leaky_relu()` activation
- `Tensor.gelu()` activation

### Changed

- Updated package version to `0.1.3`

## [0.1.2] - 2026-07-25

## Added
- `cross_entropy_loss` for categorical classification training
- `binary_cross_entropy_loss` for binary label training

### Changed
- Updated package version to `0.1.2`

## [0.1.1] - 2026-07-24

### Added

- Manual `Publish to PyPI` workflow trigger in GitHub Actions
- `CHANGELOG.md` to track release notes

### Changed

- Bumped package version from `0.1.0` to `0.1.1`
- Updated release automation and GitHub Pages site content

## [0.1.0] - 2026-07-24

### Added

- Initial package release automation for GitHub Releases
- GitHub Actions workflow for `Publish to PyPI`
- Basic package documentation, issue templates, and contribution files
- Static GitHub Pages site under `docs/`

### Fixed

- Repair README rendering and workflow details
