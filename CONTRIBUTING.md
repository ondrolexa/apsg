# Contributing

Contributions are welcome! Report bugs or propose features at
<https://github.com/ondrolexa/apsg/issues>.

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.10.

```sh
git clone https://github.com/ondrolexa/apsg.git
cd apsg
uv sync --all-extras --dev
```

## Before submitting a pull request

- Format with `ruff format src/` (line-length 88, configured in `pyproject.toml`).
- Make sure your code works with Python >= 3.10.
- Add or update docstrings for new functionality.
- If adding dependencies, update `pyproject.toml` accordingly.
