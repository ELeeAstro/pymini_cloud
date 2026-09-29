# Contributing

Documentation lives in the `docs/` directory and is written primarily in
MyST Markdown.

Before submitting documentation changes, build the site with warnings treated
as errors:

```console
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Changes pushed to the main branch are built and published automatically by the
GitHub Pages workflow.
