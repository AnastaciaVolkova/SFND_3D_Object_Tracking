import pickle
import pandas as pd
import numpy as np
import sys
import csv


def key_points_info(data):
    data_x = dict()
    detectors = list()
    for record in data:
        if record[1] is not None:
            det, des = record[0].split("_")
            if det not in detectors:
                for metric, column in record[1]["detector"].items():
                    data_x[(det, metric)] = column
    data_det = pd.DataFrame.from_dict(data_x)
    data_det.to_csv("detectors.csv", index=False)


def matchers_info(data):
    data_x = list()
    for record in data:
        if record[1] is not None:
            det, des = record[0].split("_")
            for metric, column in record[1]["matcher"].items():
                data_x.append(((det, des, metric), column))
    data_x.sort(key=lambda x: x[0][0])
    data_x = dict((x, y) for x, y in data_x)
    data_det = pd.DataFrame.from_dict(data_x)
    data_det.to_csv("matchers.csv", index=False)


def det_des_time_info(data):
    data_x = dict()
    for record in data:
        if record[1] is not None:
            det, des = record[0].split("_")
            data_x[(record[0], f"detector {det}, ms")] = record[1]["detector"]["t"]
            data_x[(record[0], f"descriptor {des}, ms")] = record[1]["descriptor"]["t"]

    data_tms = pd.DataFrame.from_dict(data_x)
    print(data_tms)
    data_tms.to_csv("timing.csv", index=False)

    det_des_timing = list()
    data_tms_cols = list(set(x[0] for x in data_tms.columns))
    for det_des in data_tms_cols:
        mx = data_tms[det_des].to_numpy()
        det_des_timing.append((det_des, np.mean(mx.sum(axis=1))))
    det_des_timing.sort(key=lambda x: x[1])
    for comb, t in det_des_timing:
        print(f"{comb},{t}")


def det_des_ttc_info(data):
    data_x = dict()
    data_p_m = dict()
    data_p_v = dict()
    dets = set()
    dess = set()

    for record in data:
        if record[1] is not None:
            x = np.array([abs(float(x)-float(y))/float(x) for x, y in zip(record[1]["Camera"]["ttc"], record[1]["Lidar"]["ttc"])])
            data_x[record[0]] = x
            det, des = record[0].split("_")

            x = x[~np.isnan(x)]
            if det not in data_p_m:
                data_p_m[det] = dict()
            data_p_m[det][des] = np.average(x)

            if det not in data_p_v:
                data_p_v[det] = dict()
            data_p_v[det][des] = np.var(x)
            dets.add(det)
            dess.add(des)

    data_ttc = pd.DataFrame.from_dict(data_x)
    print(data_ttc)
    data_ttc.to_csv("ttc.csv", index=False)

    dets = list(dets)
    dess = list(dess)
    dets.sort()
    dess.sort()

    datum = (data_p_m, data_p_v)
    for data in datum:
        line = " ," + ",".join(dess) + "\n"
        for det in dets:
            line += det + ","
            vs = [str(data[det][x]) for x in dess]
            line += ",".join(vs) + "\n"
        print(line)
        print("\n")


def det_des_pts_num_info(data):
    data_x = dict()
    data_p_m = dict()
    data_p_v = dict()
    dets = set()
    dess = set()

    for record in data:
        if record[1] is not None:
            x = record[1]["Camera"]["pts_num"]
            data_x[record[0]] = x
            det, des = record[0].split("_")

            x = np.array(x)
            x = x[~np.isnan(x)]
            if det not in data_p_m:
                data_p_m[det] = dict()
            data_p_m[det][des] = np.average(x)

            if det not in data_p_v:
                data_p_v[det] = dict()
            data_p_v[det][des] = np.var(x)
            dets.add(det)
            dess.add(des)

    # data_ttc = pd.DataFrame.from_dict(data_x)
    # print(data_ttc)
    # data_ttc.to_csv("pts.csv", index=False)

    dets = list(dets)
    dess = list(dess)
    dets.sort()
    dess.sort()

    datum = (data_p_m, data_p_v)
    for data in datum:
        line = " ," + ",".join(dess) + "\n"
        for det in dets:
            line += det + ","
            vs = [str(data[det][x]) for x in dess]
            line += ",".join(vs) + "\n"
        print(line)
        print("\n")


def ttc_camera_lidar(data):
    with open("camera_lidar_ttc.csv", mode="w") as fid:
        writer = csv.writer(fid, delimiter=',')
        i = 1
        for x in zip(data[0][1]["Camera"]["ttc"], data[0][1]["Lidar"]["ttc"], data[0][1]["Lidar"]["irq"]):
            writer.writerow((i,)+x)
            i = i + 1


def main():
    log_pk = sys.argv[1] if len(sys.argv) > 1 else "logs.pk"
    with open(log_pk, "rb") as fid:
        data = pickle.load(fid)

    # key_points_info(data)
    # matchers_info(data)
    # det_des_time_info(data)
    det_des_ttc_info(data)
    # ttc_camera_lidar(data)
    # det_des_pts_num_info(data)


if __name__ == "__main__":
    main()
