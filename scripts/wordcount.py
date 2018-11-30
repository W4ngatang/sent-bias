"""code to read words in weat file"""

import os
import json


main_path = '/scratch/sb6416/'
datasets = ['word_count_wikitext-103', '/scratch/sb6416/word_count_TBC']
word_data = ["weat1", "weat2", "weat3", "weat4"]

data = []

fullpath = '/home/sb6416/sentbias/tests'


def read_weat_words():
    for word in word_data:
        sent_file = os.path.join(fullpath, word + '.txt')
        with open(sent_file, 'r') as sent_fh:

            for row in sent_fh:
                _, examples = row.strip().split(':')
                for x in examples.split(','):
                    data.append(x)
    return data


data = read_weat_words()
for dataset in datasets:
    vocab = {}
    word_counts = {}
    output = '/home/sb6416/sentbias/data'
    output_path = os.path.join(output, 'word_count_' + dataset[-3:] + '.son')
    print(output_path)
    for root, dirs, files in os.walk(os.path.join(main_path, dataset)):

        root = os.path.abspath(root)
        for fname in files:
            txt_path = os.path.join(root, fname)
            with open(txt_path, 'r') as f:
                for line in f:

                    word = line.split(",")[0][1:]
                    freq = line.split(",")[1][:-2]
                    if word in data:
                        # print(line)
                        # print(word, freq)
                        try:

                            word_counts[word] = int(freq)

                        except BaseException:
                            print("Except: " + word)

    y = []
    for x in data:
        if x not in word_counts:
            y.append(x)
            # print(x+',')# +  ' not present in ' + dataset)
    # print(y)

    with open(output_path, 'w') as f:
            json.dumps(word_counts)
    # sorted(((v, k) for k, v in word_counts.items()), key=lambda p:p[1], reverse=True)
    print(word_counts)
