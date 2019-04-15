#!/usr/bin/env bash
echo "======== REFRESH dist DIRECTORY ========"
rm -rf dist build

echo "======== COMPILE ========"
python3.6 setup.py sdist

echo "======== UPLOAD ========"
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

text="
_______________________________________________________________________________

To install this package, please follow the command below
$ pip install --index-url https://test.pypi.org/simple/ --no-deps TF_RL
_______________________________________________________________________________
"
echo "$text"