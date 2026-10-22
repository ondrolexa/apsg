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

## Building the documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Pages live in
`docs/` as Markdown, the API reference is generated from docstrings by
[mkdocstrings](https://mkdocstrings.github.io/), and the tutorial notebooks in
`docs/notebooks/` are executed on every build, so commit them without outputs.

```sh
uv run mkdocs serve            # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict   # full build into site/, warnings are errors
```

## Documentation conventions

- Docstrings follow the Google style already used in the package (`Args:`,
  `Keyword Args:`, `Returns:`, `Examples:`). They state the public contract:
  what the function takes and returns, units (degrees) and conventions such
  as the sign of a sense of movement or a stress tensor.
- Every method implementing a published scientific method gets a
  `References:` section with the full citation, and its docstring names the
  assumptions the method makes (e.g. what a statistic assumes about the data).
- Inline comments explain *why*, not *what*: non-obvious thresholds,
  tolerances, sign flips and numerical workarounds. If a constant has no
  literature source, say so instead of leaving it unexplained.
- Vendored code (such as `KentDistribution`) keeps its upstream style; mark
  it as vendored in its docstring.
- `CLAUDE.md` holds the architecture overview. Explanations of how a single
  function works belong in that function's docstring or comments, not only
  there.

## Before submitting a pull request

- Format and lint with `ruff format src/ tests/` and `ruff check src/ tests/`
  (line length 88, configured in `pyproject.toml`).
- Make sure the test suite passes with `uv run pytest`.
- Add or update docstrings for new functionality.
- Add an entry under the current version in `CHANGELOG.md` for user-visible changes.
  Mark breaking changes with **BREAKING:**.
- If adding dependencies, add them to `pyproject.toml` (for example with `uv add`).
