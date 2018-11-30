import json

FILES = ['angry_black_woman_stereotype_b', 'angry_black_woman_stereotype',
	 'heilman_double_bind_ambiguous_1,3-', 'heilman_double_bind_ambiguous_1-',
	 'heilman_double_bind_ambiguous_1', 'heilman_double_bind_clear_1,3-',
	 'heilman_double_bind_clear_1-', 'heilman_double_bind_clear_1',
	 'project_implicit_arab-muslim', 'project_implicit_disability',
	 'project_implicit_native', 'project_implicit_religion',
	 'project_implicit_sexuality', 'project_implicit_skin-tone',
	 'project_implicit_weapons', 'project_implicit_weight',
	 'sent-weat1', 'sent-weat2', 'sent-weat3', 'sent-weat4',
	 'weat1', 'weat2', 'weat3b', 'weat3', 'weat4',
	 'weat5b', 'weat5', 'weat6b', 'weat6',
	 'weat7b', 'weat7', 'weat8b', 'weat8',
	 'weat9', 'weat10']

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
    for file in FILES:
        convert_file(file)
