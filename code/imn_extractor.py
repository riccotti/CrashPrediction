import os
import sys
sys.path.append('/home/riccardo/Documenti/PhD/TrackAndKnow/code/imn')

import json
import gzip
import datetime
import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict
from networkx.readwrite import json_graph

import trajectory
import database_io
import individual_mobility_network


__author__ = 'Riccardo Guidotti'


def key2str(k):
    if isinstance(k, tuple):
        return str(k)
    elif isinstance(k, datetime.time):
        return str(k)
    elif isinstance(k, np.int64):
        return str(k)
    elif isinstance(k, np.float64):
        return str(k)
    return k


def clear_tuples4json(o):
    if isinstance(o, dict):
        return {key2str(k): clear_tuples4json(o[k]) for k in o}
    return o


def agenda_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()
    elif isinstance(o, datetime.timedelta):
        return o.__str__()
    elif isinstance(o, trajectory.Trajectory):
        return o.to_json()
    elif isinstance(o, nx.DiGraph):
        return json_graph.node_link_data(o, {'link': 'edges', 'source': 'from', 'target': 'to'})
    else:
        return o.__str__()


def start_time_map(t):
    m = t.month
    if m == 1:
        return '01-02', None
    elif m == 12:
        return '11-12', None
    else:
        return '%02d-%02d' % (m-1, m), '%02d-%02d' % (m, m+1)


def imn_extract(filename, path, type_user, traj_table, evnt_table,
                min_traj_nbr, min_length, min_duration, area, overwrite=False, users_filename_crash=None):

    output_filename = path + '%s_imn_%s.json.gz' % (area, type_user)

    con = database_io.get_connection()
    cur = con.cursor()

    users_list = pd.read_csv(filename).values[:, 0].tolist()
    users_list = sorted(users_list)

    if users_filename_crash is not None:
        users_list_crash = pd.read_csv(users_filename_crash).values[:, 0].tolist()
        users_list_crash = sorted(users_list_crash)
        users_list = [uid for uid in users_list if uid not in users_list_crash]

    nbr_users = len(users_list)
    print(nbr_users, len(users_list))
    if os.path.isfile(output_filename) and not overwrite:
        processed_users = list()
        fout = gzip.GzipFile(output_filename, 'r')
        # count = 0
        for row in fout:
            customer_obj = json.loads(row)
            processed_users.append(customer_obj['uid'])
            # print(customer_obj['uid'])
            # if count == 100:
            #     break
            # count += 1
        fout.close()
        users_list = [uid for uid in users_list if uid not in processed_users]

    print(nbr_users, len(users_list))
    # from_perc = 95
    # to_perc = 100
    for i, uid in enumerate(users_list):

        # if not from_perc < i / len(users_list) * 100.0 <= to_perc:
        #     continue

        if i % 1 == 0:
            print(datetime.datetime.now(), '%s %s %s [%s/%s] - %.2f' % (
                traj_table, area, type_user, i,  nbr_users, i / nbr_users * 100.0))
            # print(datetime.datetime.now(), '%s %s %s %.2f' % (
            # traj_table, area, type_user, i / len(users_list) * 100.0), from_perc, to_perc)

        imh = database_io.load_individual_mobility_history(cur, uid, traj_table, min_length, min_duration)
        events = database_io.load_individual_event_history(cur, uid, evnt_table) if evnt_table is not None else None

        if len(imh['trajectories']) < min_traj_nbr:
            # print('len trajectories]) < min_traj_nbr', len(imh['trajectories']), min_traj_nbr)
            continue

        # print(len(events))
        # print(list(events.keys()))

        wimh_dict = dict()
        wevents_dict = dict()
        for tid, traj in imh['trajectories'].items():
            st = traj.start_time()
            stk_list = start_time_map(st)
            for stk in stk_list:
                if stk is None:
                    continue
                if stk not in wimh_dict:
                    wimh_dict[stk] = {'uid': uid, 'trajectories': dict()}
                    wevents_dict[stk] = dict()
                wimh_dict[stk]['trajectories'][tid] = traj
                if tid in events:
                    wevents_dict[stk][tid] = events[tid]

        customer_obj = {'uid': uid}
        for stk in wimh_dict:
            wimh = wimh_dict[stk]
            wevents = wevents_dict[stk]
            # print(stk, len(wimh['trajectories']), len(wevents))
            if len(wimh['trajectories']) < min_traj_nbr // 12:
                continue

            imn = individual_mobility_network.build_imn(wimh, reg_loc=True, events=wevents, verbose=False)
            customer_obj[stk] = imn

        json_str = '%s\n' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)
        json_bytes = json_str.encode('utf-8')
        with gzip.GzipFile(output_filename, 'a') as fout:
            fout.write(json_bytes)
        # with gzip.GzipFile(output_filename.replace('.json.gz', '_%s_%s.json.gz' % (from_perc, to_perc)), 'a') as fout:
        #     fout.write(json_bytes)

    cur.close()
    con.close()


def main():

    area = sys.argv[1]       # 'rome' 'tuscany' 'london'
    type_user = sys.argv[2]  # 'crash' 'nocrash'
    overwrite = False

    country = 'uk' if area == 'london' else 'italy'

    path = './'
    path_dataset = path + 'dataset/'
    path_imn = path + 'imn_new/'

    traj_table = 'tak.%s_traj' % country
    evnt_table = 'tak.%s_evnt' % country
    if area == 'london' and type_user == 'nocrash':
        users_filename = path_dataset + '%s_%s_users_list.csv' % (area, 'all')
        users_filename_crash = path_dataset + '%s_%s_users_list.csv' % (area, 'crash')
    else:
        users_filename = path_dataset + '%s_%s_users_list.csv' % (area, type_user)
        users_filename_crash = None

    min_traj_nbr = 300
    min_length = 1.0
    min_duration = 60.0

    imn_extract(users_filename, path_imn, type_user, traj_table, evnt_table,
                min_traj_nbr, min_length, min_duration, area, overwrite, users_filename_crash)


if __name__ == "__main__":
    main()
