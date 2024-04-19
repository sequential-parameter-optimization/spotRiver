#!/bin/sh
rm -f dist/spotriver*; python -m build; python -m pip install dist/spotriver*.tar.gz
python -m mkdocs build
pytest
