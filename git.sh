#!/bin/bash
git add .
git commit -m "$1"
git push 
echo "pushed with commit message: $1"