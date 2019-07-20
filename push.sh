#!/usr/bin/env bash
echo Hello, what is the comment for this commit?

read comment

git add . && git commit -m "$comment" && git push -u origin master