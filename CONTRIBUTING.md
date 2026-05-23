# Contributing

Contributions are welcome! Report bugs or propose features at
[https://github.com/ondrolexa/apsg/issues](https://github.com/ondrolexa/apsg/issues).

## Development setup

Requires Python >= 3.10.

```sh
git clone https://github.com/ondrolexa/apsg.git
cd apsg
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before submitting a pull request

- Format with `black src/` (line-length 88, configured in `pyproject.toml`).
- Make sure your code works with Python >= 3.10.
- Add or update docstrings for new functionality.
- If adding dependencies, update `pyproject.toml` accordingly.
