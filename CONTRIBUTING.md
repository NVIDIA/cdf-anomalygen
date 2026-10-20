# Contributing

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Setup](#setup)
  - [Virtual environment](#virtual-environment)
  - [Docker](#docker)
- [Updating the cosmos-framework pin](#updating-the-cosmos-framework-pin)
- [Regenerating the license file](#regenerating-the-license-file)
- [Test](#test)
  - [Run Linting and Formatting](#run-linting-and-formatting)
  - [Run Tests](#run-tests)
  - [Test Coverage](#test-coverage)
  - [Run a Single Test](#run-a-single-test)
- [Code Reviews](#code-reviews)
- [Signing Off Your Work](#signing-off-your-work)

______________________________________________________________________

<!--TOC-->

We'd love to receive your patches and contributions. Please keep your PRs as draft until such time that you would like us to review them.

## Setup

### Virtual environment

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) as the virtual environment manager.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Build the venv:

```shell
# MAX_JOBS: parallel compile jobs (default 16)
# NVCC_THREADS: threads per nvcc invocation (default 1)
bash scripts/env_setup.sh [MAX_JOBS] [NVCC_THREADS]
```

Activate the environment and install the development tooling:

```shell
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

Install the pre-commit hooks so they run automatically on `git commit`:

```shell
pre-commit install
```

### Docker

The Docker `develop` image already bundles this dev tooling (`pytest`, `ruff`,
`pre-commit`) on top of the runtime, so you can develop in a container instead of the
venv. See [docker/README.md](docker/README.md).

## Updating the cosmos-framework pin

This repo installs [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) as a pip
package from a specific upstream commit, pinned in `requirements-nodeps.txt`. It is installed
with `--no-deps` because one of its pins contradicts ours: it declares
`transformers>=4.57.1,<5.0.0`, so a normal resolve would drag transformers back to 4.x. That
is the only conflict among its base dependencies — every other one it declares is already
satisfied by `requirements.txt`, which is where the subset this repo actually uses is pinned.
Its optional `train` extra, which we do not request, also pins `multi-storage-client==0.44.0`
against the 1.1.0 we ship; see the note in `requirements-nodeps.txt` for the API surface that
covers.

Bumping the pin means editing the `@<sha>` in `requirements-nodeps.txt`:

```shell
# 1. Resolve the target commit (or pick a specific <commit-sha>).
git ls-remote https://github.com/NVIDIA/cosmos-framework.git main

# 2. Edit the `@<sha>` suffix in requirements-nodeps.txt, then reinstall into the active venv.
uv pip install -r requirements-nodeps.txt --no-deps --reinstall-package cosmos-framework

# 3. Re-check that requirements.txt still covers the deps the new commit needs, then commit.
git add requirements-nodeps.txt
git commit -s -m "Bump cosmos-framework to <short-sha>"
```

## Regenerating the license file

`LICENSE-3rd-party.txt` is generated, not hand-edited. It reads license facts from the *installed*
distributions, so **generate it from the `develop` Docker image, not a local venv**

`develop` rather than `product` because `develop` is the image this project releases.

```shell
# build the develop image first (see docker/README.md), then:
docker run --rm --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
    -v "$PWD:/workspace/cdf-anomalygen" -w /workspace/cdf-anomalygen \
    "$IMAGE" /opt/venv/bin/python scripts/generate_third_party_licenses.py
```

It covers every third-party package that ends up on disk, transitive dependencies included.
Regenerate it whenever dependencies change, review the diff, then commit the result.

A distribution whose wheel ships no license text is a hard error rather than a silent omission —
add it to `UPSTREAM_LICENSE_URLS` (citing the canonical upstream file) or, if no text exists
anywhere, to `NO_UPSTREAM_TEXT`.

## Test

### Run Linting and Formatting

```shell
pre-commit run -a
```

This runs the configured hooks over all files, including Python lint + autofix (`ruff check --fix`), Python formatting (`ruff format`), and Markdown formatting (`rumdl-fmt`). We recommend that you commit your changes first.

### Run Tests

```shell
pytest
```

Tests live under `tests/`, mirroring the `anomalygen/` package layout. The suite is CPU-only by
default: any test that needs a CUDA GPU is marked `@pytest.mark.gpu` and auto-skips when no GPU is
present (see `tests/conftest.py`).

### Test Coverage

`pytest-cov` reports coverage for the `anomalygen` package:

```shell
pytest --cov=anomalygen --cov-report=term-missing   # console, with uncovered lines
pytest --cov=anomalygen --cov-report=html           # browsable report in htmlcov/
```

Coverage settings live in `pyproject.toml` under `[tool.coverage.*]`;
`tests/` and `anomalygen/scripts/` are excluded. The CI `test` job reports the total coverage.

### Run a Single Test

```shell
pytest path/to/test_file.py::test_name [--pdb]
```

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

<!-- rumdl-disable MD004 MD014 MD031 MD034 -->

## Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

<!-- rumdl-enable -->
