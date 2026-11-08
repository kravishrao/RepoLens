# RepoLens Python Repository Evaluation Dataset

## 1. Overview

This dataset contains five Python repositories selected to evaluate
RepoLens's repository-level code understanding and question-answering
capabilities.

The repositories provide different software architectures, module
structures, dependencies, classes, functions and test suites.

## 2. Dataset Contents

| Repository | Source | Commit |
|---|---|---|
| Flask | https://github.com/pallets/flask | 49943b5a94413a530761350f04e896f76e8d620e |
| Requests | https://github.com/psf/requests | 3c7baaa05ab4aa49cc819ce3a819d260de19c157 |
| FastAPI | https://github.com/fastapi/fastapi | 8f9f47de69982705d96e4165a9f03c6537e8c9fe |
| Click | https://github.com/pallets/click | 8abb77a8432e03ed14a95f972b44f905279415a7 |
| Pydantic | https://github.com/pydantic/pydantic | 57ffc4a517ddbd16ce697b4600d24a4fa1e33801 |

Exact repository metadata is maintained in `metadata/repositories.json`.

## 3. Repository Selection Criteria

Repositories were selected based on:

- Python as the primary implementation language.
- Availability of public GitHub source code.
- Sufficient module and package structure.
- Presence of classes, functions and imports.
- Availability of dependency and configuration files.
- Availability of meaningful tests and examples.
- Diversity of software application domains.

## 4. Repository Characteristics

- Flask: Python web framework.
- Requests: HTTP client library.
- FastAPI: API development framework.
- Click: Command-line interface framework.
- Pydantic: Data validation and settings library.

These repositories provide different structural patterns for evaluating
repository-level code understanding.

## 5. Snapshot Preparation

Each repository was cloned from its public GitHub source and exported
using `git archive` at its recorded commit identifier.

This ensures that snapshots correspond to specific repository versions.

The exported snapshots do not contain Git history.

## 6. Included Content

The dataset retains relevant tracked repository content, including:

- Python source files.
- Test suites and fixtures.
- Dependency manifests.
- Project configuration files.
- README and documentation files.
- License and attribution files.
- Relevant examples.

## 7. Excluded Content

The preparation process excludes:

- Git metadata and commit history.
- Python bytecode and cache files.
- Virtual environments.
- Generated build directories.
- Test and lint caches.
- Other specified generated artifacts.

## 8. Reproducibility

Each repository is identified by:

- Repository name.
- Original GitHub URL.
- Exact Git commit identifier.
- Local snapshot path.

To reproduce a snapshot, clone its source repository and export the
recorded commit using `git archive`.

Example:

    git clone https://github.com/pallets/flask.git
    git -C flask archive --format=tar \
      49943b5a94413a530761350f04e896f76e8d620e \
      | tar -xf - -C flask-snapshot

## 9. Limitations

- The dataset contains only five repositories.
- Repository sizes and architectural complexity vary.
- The selected repositories are public open-source projects.
- The dataset represents fixed snapshots, not continuously updated code.
- Repository selection may introduce domain and project-size bias.
- Dependency manifests are preserved, but dependencies are not installed
  as part of dataset preparation.
- Snapshot availability depends on the continued availability of
  the original source repositories.

## 10. Licensing and Attribution

The original repository licenses and attribution files are retained
where available in each snapshot.

Each repository remains subject to its original license.

Users of this dataset should review the individual repository licenses
before redistribution or other reuse.

## 11. Directory Structure

    dataset/
    ├── README.md
    ├── metadata/
    │   └── repositories.json
    └── repositories/
        ├── flask/
        ├── requests/
        ├── fastapi/
        ├── click/
        └── pydantic/
