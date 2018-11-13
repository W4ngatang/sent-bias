# ./weat_glove2elmo.sh weat1
# Convert the weat files (for glove) into elmo-compatible format
head -1 ../tests/$1.txt | tail -1 | tr "," "\n" > ../elmo/$1.X.txt
head -2 ../tests/$1.txt | tail -1 | tr "," "\n" > ../elmo/$1.Y.txt
head -3 ../tests/$1.txt | tail -1 | tr "," "\n" > ../elmo/$1.A.txt
head -4 ../tests/$1.txt | tail -1 | tr "," "\n" > ../elmo/$1.B.txt
