import os
import json


def convert_file(filename):
    print("Converting %s..." % filename)
    data = {}
    with open("%s.txt" % filename) as filehandle:
        for raw in filehandle:
            if raw[0] == "#":
                continue
            row = raw.strip().split('\t')
            data[row[0]] = {"category": row[1], "examples": row[2:]}

    with open("%s.jsonl" % filename, 'w') as out_fh:
        json.dump(data, out_fh, indent=2)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info('Note: this script should be called from the "tests" directory')

    files = []
    for entry in os.listdir('.'):
        if not entry.startswith('.') and entry.endswith('.txt'):
            files.append(entry[:-len('.txt')])
    for f in files:
        convert_file(f)
