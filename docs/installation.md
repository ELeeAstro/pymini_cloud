# Installation

`pymini_cloud` requires Python 3.14 or newer.

## Install from a local checkout

Clone the repository and install it in editable mode:

```console
git clone https://github.com/ELeeAstro/pymini_cloud.git
cd pymini_cloud
python -m pip install -e .
```

## Install documentation tools

To build this website locally, install the optional documentation dependencies:

```console
python -m pip install -e ".[docs]"
```

Then build the HTML documentation:

```console
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser to view the result.
