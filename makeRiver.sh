#!/bin/zsh
cd ~/workspace/spotRiver
pytest
rm -f dist/spotRiver*; python -m build; python -m pip install dist/spotRiver*.tar.gz
python -m mkdocs build
