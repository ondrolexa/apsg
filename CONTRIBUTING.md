# Contributing

Contributions are welcome! Report bugs or propose features at
<https://github.com/ondrolexa/apsg/issues>.

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

```sh
git clone https://github.com/ondrolexa/apsg.git
cd apsg
uv sync --all-extras --dev
```

Install the git hooks, which run ruff and the test suite on every commit:

```sh
uvx pre-commit install
```

## Running tests and checks

```sh
uv run pytest            # full suite, parallel by default
uv run pytest -n0        # single process, for debugging with --pdb or breakpoints
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Before submitting a pull request

- Format and lint with `ruff format src/ tests/` and `ruff check src/ tests/`
  (line length 88, configured in `pyproject.toml`).
- Make sure the test suite passes with `uv run pytest`.
- Add or update docstrings for new functionality.
- Add an entry under the current version in `CHANGELOG.md` for user-visible changes.
  Mark breaking changes with **BREAKING:**.
- If adding dependencies, add them to `pyproject.toml` (for example with `uv add`).
