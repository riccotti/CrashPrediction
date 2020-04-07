import os
import sys

import json
import gzip
import datetime
import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta

import database_io
import feature_extractor


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():

    area = sys.argv[1]      # 'rome' 'tuscany' 'london'
    type_user = sys.argv[2]  # 'crash' 'nocrash'
    overwrite = int(sys.argv[3])
    country = 'uk' if area == 'london' else 'italy'

    min_length = 1.0
    min_duration = 60.0

    print(datetime.datetime.now(), 'Crash Prediction - Train Test Partitioner')
    if not overwrite:
        print(datetime.datetime.now(), '(restart)')

    path = './'
    path_imn = path + 'imn_new/'
    path_dataset = path + 'dataset/'
    path_traintest = path + 'traintest/'
    path_quadtree = path + 'quadtree/'

    traj_table = 'tak.%s_traj' % country
    evnt_table = 'tak.%s_evnt' % country
    crash_table = 'tak.%s_crash' % country

    if area == 'london' and type_user == 'nocrash':
        users_filename = path_dataset + '%s_%s_users_list.csv' % (area, 'all')
        users_filename_crash = path_dataset + '%s_%s_users_list.csv' % (area, 'crash')
    else:
        users_filename = path_dataset + '%s_%s_users_list.csv' % (area, type_user)
        users_filename_crash = None

    users_list = pd.read_csv(users_filename).values[:, 0].tolist()
    users_list = sorted(users_list)

    if users_filename_crash is not None:
        users_list_crash = pd.read_csv(users_filename_crash).values[:, 0].tolist()
        users_list_crash = sorted(users_list_crash)
        users_list = [uid for uid in users_list if uid not in users_list_crash]

    nbr_users = len(users_list)

    print(datetime.datetime.now(), 'Reading quadtree')
    quadtree_poi_filename = path_quadtree + '%s_personal_osm_poi_lv17.json.gz' % area
    fout = gzip.GzipFile(quadtree_poi_filename, 'r')
    quadtree = json.loads(fout.readline())
    fout.close()

    print(datetime.datetime.now(), 'Reading quadtree features')
    quadtree_features_filename = path_quadtree + '%s_quadtree_features.json.gz' % area
    fout = gzip.GzipFile(quadtree_features_filename, 'r')
    quadtrees_features_str = json.loads(fout.readline())
    quadtrees_features = {int(k): v for k, v in quadtrees_features_str.items()}
    fout.close()

    processed_users = set()
    if overwrite:
        for index in range(0, 7):
            output_filename = path_traintest + '%s_%s_traintest_%s.json.gz' % (area, type_user, index)
            if os.path.exists(output_filename):
                os.remove(output_filename)
    else:
        processed_users = set()
        for index in range(0, 7):
            output_filename = path_traintest + '%s_%s_traintest_%s.json.gz' % (area, type_user, index)
            if os.path.isfile(output_filename):
                fout = gzip.GzipFile(output_filename, 'r')
                for row in fout:
                    customer_obj = json.loads(row)
                    processed_users.add(customer_obj['uid'])
                fout.close()

    window = 4
    datetime_from = datetime.datetime.strptime('2017-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    datetime_to = datetime.datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

    print(datetime.datetime.now(), 'Generating month boundaries')
    months = pd.date_range(start=datetime_from, end=datetime_to, freq='MS')
    boundaries = [[lm, um] for lm, um in zip(months[:-window], months[window:])]
    training_months = list()
    test_months = list()
    for i in range(len(boundaries)-1):
        training_months.append(boundaries[i])
        test_months.append(boundaries[i+1])

    index = 0
    tr_data_map = dict()
    ts_data_map = dict()
    for tr_months, ts_months in zip(training_months, test_months):
        tr_data_map[tuple(tr_months)] = index
        ts_data_map[tuple(ts_months)] = index
        index += 1

    print(datetime.datetime.now(), 'Initializing quadtree features')
    tr_quadtree_features = dict()
    for m in quadtrees_features:
        for lu, index in tr_data_map.items():
            if lu[0].month <= m < lu[1].month:
                if index not in tr_quadtree_features:
                    tr_quadtree_features[index] = dict()
                for path in quadtrees_features[m]:
                    if path not in tr_quadtree_features[index]:
                        tr_quadtree_features[index][path] = {
                            'nbr_traj_start': 0,
                            'nbr_traj_stop': 0,
                            'nbr_traj_move': 0,
                            'traj_speed_sum': 0,
                            'traj_speed_count': 0,
                            'nbr_evnt_A': 0,
                            'nbr_evnt_B': 0,
                            'nbr_evnt_C': 0,
                            'nbr_evnt_Q': 0,
                            'nbr_evnt_start': 0,
                            'nbr_evnt_stop': 0,
                            'speed_A_sum': 0,
                            'max_acc_A_sum': 0,
                            'avg_acc_A_sum': 0,
                            'speed_B_sum': 0,
                            'max_acc_B_sum': 0,
                            'avg_acc_B_sum': 0,
                            'speed_C_sum': 0,
                            'max_acc_C_sum': 0,
                            'avg_acc_C_sum': 0,
                            'speed_Q_sum': 0,
                            'max_acc_Q_sum': 0,
                            'avg_acc_Q_sum': 0,
                            'nbr_crash': 0,
                        }
                    for k, v in quadtrees_features[m][path].items():
                        tr_quadtree_features[index][path][k] += v

    ts_quadtree_features = dict()
    for m in quadtrees_features:
        for lu, index in tr_data_map.items():
            if lu[0].month <= m < lu[1].month:
                if index not in ts_quadtree_features:
                    ts_quadtree_features[index] = dict()
                for path in quadtrees_features[m]:
                    if path not in ts_quadtree_features[index]:
                        ts_quadtree_features[index][path] = {
                            'nbr_traj_start': 0,
                            'nbr_traj_stop': 0,
                            'nbr_traj_move': 0,
                            'traj_speed_sum': 0,
                            'traj_speed_count': 0,
                            'nbr_evnt_A': 0,
                            'nbr_evnt_B': 0,
                            'nbr_evnt_C': 0,
                            'nbr_evnt_Q': 0,
                            'nbr_evnt_start': 0,
                            'nbr_evnt_stop': 0,
                            'speed_A_sum': 0,
                            'max_acc_A_sum': 0,
                            'avg_acc_A_sum': 0,
                            'speed_B_sum': 0,
                            'max_acc_B_sum': 0,
                            'avg_acc_B_sum': 0,
                            'speed_C_sum': 0,
                            'max_acc_C_sum': 0,
                            'avg_acc_C_sum': 0,
                            'speed_Q_sum': 0,
                            'max_acc_Q_sum': 0,
                            'avg_acc_Q_sum': 0,
                            'nbr_crash': 0,
                        }
                    for k, v in quadtrees_features[m][path].items():
                        ts_quadtree_features[index][path][k] += v

    print(datetime.datetime.now(), 'Connecting to database')
    con = database_io.get_connection()
    cur = con.cursor()

    count = 0
    imn_filedata = gzip.GzipFile(path_imn + '%s_imn_%s.json.gz' % (area, type_user), 'r')

    print(datetime.datetime.now(), 'Calculating features and partitioning dataset')
    for row in imn_filedata:
        if len(row) <= 1:
            print('new file started ;-)')
            continue

        user_obj = json.loads(row)
        uid = user_obj['uid']
        count += 1
        if uid in processed_users:
            continue

        if count % 10 == 0:
            print(datetime.datetime.now(), 'train test partition %s %s [%s/%s] - %.2f' % (
                area, type_user, count, nbr_users, 100*count/nbr_users))

        imh = database_io.load_individual_mobility_history(cur, uid, traj_table, min_length, min_duration)
        events = database_io.load_individual_event_history(cur, uid, evnt_table) if evnt_table is not None else None
        trajectories = imh['trajectories']

        tr_data = dict()
        ts_data = dict()

        # partitioning imn for train and test
        for imn_months in user_obj:
            if imn_months == 'uid':
                continue

            # print(imn_months)
            m0 = int(imn_months.split('-')[0])
            m1 = int(imn_months.split('-')[1])
            for lu, index in tr_data_map.items():
                if lu[0].month <= m0 < m1 < lu[1].month:
                    if index not in tr_data:
                        tr_data[index] = {'uid': uid, 'crash': False, 'trajectories': dict(),
                                          'imns': dict(), 'events': dict(),}
                    tr_data[index]['imns'][imn_months] = user_obj[imn_months]

            for lu, index in ts_data_map.items():
                if lu[0].month <= m0 < lu[1].month:
                    if index not in ts_data:
                        ts_data[index] = {'uid': uid, 'crash': False, 'trajectories': dict(),
                                          'imns': dict(), 'events': dict(),}
                    ts_data[index]['imns'][imn_months] = user_obj[imn_months]

        # partitioning trajectories for train and test
        for tid, traj in trajectories.items():
            for lu, index in tr_data_map.items():
                if lu[0] <= traj.start_time() < lu[1] and index in tr_data:
                    tr_data[index]['trajectories'][tid] = traj
            for lu, index in ts_data_map.items():
                if lu[0] <= traj.start_time() < lu[1] and index in ts_data:
                    ts_data[index]['trajectories'][tid] = traj

        # partitioning events for train and test
        for eid, evnt in events.items():
            # print(evnt)
            for lu, index in tr_data_map.items():
                if lu[0] <= evnt[0]['date'] < lu[1] and index in tr_data:
                    tr_data[index]['events'][eid] = evnt[0]
            for lu, index in ts_data_map.items():
                if lu[0] <= evnt[0]['date'] < lu[1] and index in ts_data:
                    ts_data[index]['events'][eid] = evnt[0]

        # get has crash next month
        for lu, index in tr_data_map.items():
            if index not in tr_data:
                continue
            query = """SELECT * FROM %s WHERE uid = '%s' 
                        AND date >= TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS') 
                        AND date < TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS')""" % (
                crash_table, uid, str(lu[1]), str(lu[1] + relativedelta(months=1)))
            cur.execute(query)
            rows = cur.fetchall()
            has_crash_next_month = len(rows) > 0
            tr_data[index]['crash'] = has_crash_next_month

        for lu, index in ts_data_map.items():
            if index not in ts_data:
                continue
            query = """SELECT * FROM %s WHERE uid = '%s' 
                        AND date >= TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS') 
                        AND date < TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS')""" % (
                crash_table, uid, str(lu[1]), str(lu[1] + relativedelta(months=1)))
            cur.execute(query)
            rows = cur.fetchall()
            has_crash_next_month = len(rows) > 0
            ts_data[index]['crash'] = has_crash_next_month

        tr_features, ts_features = feature_extractor.extract_features(uid, tr_data, ts_data, quadtree,
                                                                      tr_quadtree_features, ts_quadtree_features)

        for index in tr_features:
            if index in ts_features:
                output_filename = path_traintest + '%s_%s_traintest_%s.json.gz' % (area, type_user, index)
                store_obj = {'uid': uid, 'train': tr_features[index], 'test': ts_features[index]}
                feature_extractor.store_features(output_filename, store_obj)


    imn_filedata.close()


if __name__ == "__main__":
    main()
