import re
import os
import pickle
import json
import sys


def parse(text_data):
    lines = text_data.splitlines()
    pats = {
        "n": (re.compile(r"(?<=n\=)\d+"), int),
        "t": (re.compile(r"(?<=t\=)[\d\.]+"), float),
        "m1": (re.compile(r"(?<=m1\=)[\d\.]+"), float),
        "rms": (re.compile(r"(?<=rms\=)[\d\.nan]+"), float),
        "ttc": (re.compile(r"(?<=ttc\=)[\d\.nan]+"), float),
        "irq": (re.compile(r"(?<=irq\=)[\d\.]+"), float),
        "pts_num": (re.compile(r"(?<=pts_num\=)[\d\.nan]+"), int)
    }

    data = {
        "detector": {"n": list(), "t": list(), "m1": list(), "rms": list()},
        "descriptor": {"n": list(), "t": list()},
        "matcher":  {"n": list(), "t": list()},
        "Lidar": {"ttc": list(), "irq": list()},
        "Camera": {"ttc": list(), "pts_num": list(), "irq": list()},
    }
    for line in lines:
        for k in data:
            if k in line:
                for pat in pats:
                    m = pats[pat][0].findall(line)
                    if len(m) != 0:
                        data[k][pat].append(pats[pat][1](m[0]))
    return data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "./log/"
    parsed_data = list()
    for file in os.listdir(log_dir):
        if "log" in file.lower() and os.path.isfile(os.path.join(log_dir, file)):
            file = os.path.join(log_dir, file)
            with open(file) as fid:
                text = fid.read()
                file = os.path.splitext(os.path.basename(file))[0]
                parsed_data.append((os.path.splitext(file)[0], parse(text)))
    for i in parsed_data:
        print(i)
    store_file = open("logs.pk", "wb")
    pickle.dump(parsed_data, store_file)
    store_file.close()

    store_file = open("logs.json", "w")
    json.dump(parsed_data, store_file)
    store_file.close()
