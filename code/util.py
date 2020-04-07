import math
import datetime

from collections import defaultdict

R_EARTH = 6371000


def dist2angle(dist):

    return dist * 180.0 / math.pi / R_EARTH


def get_ordered_history(imh):
    history_order_dict = dict()
    for tid in imh['trajectories']:
        ts = datetime.datetime.fromtimestamp(imh['trajectories'][tid].start_point()[2] / 1000.0)
        history_order_dict[tid] = ts
    return history_order_dict


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d
