import shutil
from subprocess import Popen
from subprocess import PIPE
import os
import re
from queue import Queue
import queue
from threading import Lock
from threading import Thread
import pickle
import json
from parse_logs import parse

output_log_dir = "./log/"


def run_tracker(t_id, lock, tools, parsed_data):
    to_continue = True
    exe = os.path.join("..", "build", "3D_object_tracking")
    lock.acquire()
    print(f"Thread {t_id} starts")
    lock.release()
    while to_continue:
        try:
            case = tools.get(block=False)
            os.makedirs(f"./log/img/{case[0]}_{case[1]}")
            command_line = [exe] + f"-det {case[0]} -des {case[1]} -dir ./log/img/{case[0]}_{case[1]} -sel SEL_NN".split()
            proc = Popen(command_line, stderr=PIPE, stdout=PIPE)
            out_bin, err_bin = proc.communicate()
            if proc.returncode != 0:
                lock.acquire()
                print(case)
                lock.release()
            if len(err_bin):
                parsed_data.put((f"{case[0]}_{case[1]}", None))
            else:
                parsed_data.put((f"{case[0]}_{case[1]}", parse(out_bin.decode())))
                with open(os.path.join(output_log_dir, f"{case[0]}_{case[1]}.log"), "wb") as fid:
                    fid.write(out_bin)
            tools.task_done()
        except queue.Empty:
            to_continue = False
    lock.acquire()
    print(f"Thread {t_id} finishes")
    lock.release()


def main():
    thread_number = 8
    detectors = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
    descriptors = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]

    test_cases = Queue()
    parsed_data = Queue()
    lock = Lock()
    for det in detectors:
        for des in descriptors:
            test_cases.put((det, des))

    if os.path.exists(output_log_dir):
        shutil.rmtree(output_log_dir)
    os.mkdir(output_log_dir)

    my_threads = list()
    for i in range(thread_number):
        t = Thread(target=run_tracker, args=(i, lock, test_cases, parsed_data))
        t.start()
        my_threads.append(t)

    for t in my_threads:
        t.join()

    data_to_store = list()
    while parsed_data.qsize() != 0:
        d = parsed_data.get()
        data_to_store.append(d)
        print(d)

    store_file = open("logs.pk", "wb")
    pickle.dump(data_to_store, store_file)
    store_file.close()

    store_file = open("logs.json", "w")
    json.dump(data_to_store, store_file)
    store_file.close()


if __name__ == "__main__":
    main()
