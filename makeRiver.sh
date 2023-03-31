#!/bin/zsh
cd ~/workspace/spotRiver
rm -f dist/spotRiver*; python -m build; python -m pip install dist/spotRiver*.tar.gz
