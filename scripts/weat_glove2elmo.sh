#!/bin/bash
# ./weat_glove2elmo.sh weat1
# Convert test files into elmo-compatible format
set -e
if [ $# -ne 1 ]
then
    echo "Usage: $0 <test-name>" >&2
    echo "Example: $0 weat1" >&2
    exit 1
fi
mkdir -p ../elmo
grep '^targ1' ../tests/$1.txt | cut -f 3- | tr "\t" "\n" > ../elmo/$1.X.txt
grep '^targ2' ../tests/$1.txt | cut -f 3- | tr "\t" "\n" > ../elmo/$1.Y.txt
grep '^attr1' ../tests/$1.txt | cut -f 3- | tr "\t" "\n" > ../elmo/$1.A.txt
grep '^attr2' ../tests/$1.txt | cut -f 3- | tr "\t" "\n" > ../elmo/$1.B.txt
