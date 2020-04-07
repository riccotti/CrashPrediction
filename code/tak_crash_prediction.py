import os
import sys
import json
import gzip
import time
import pickle
import datetime
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Point, LineString
from dateutil.relativedelta import relativedelta

from multiprocessing.pool import ThreadPool

from trajectory import Trajectory
from individual_mobility_network import build_imn
from feature_extractor import extract_features_data

from visualization import visualize_points, visualize_trajectories, visualize_stops
from visualization import visualize_locations, visualize_imn, visualize_features, visualize_crash_risk


periods_map = {
    'jun': 0,
    'jul': 1,
    'aug': 2,
    'sep': 3,
    'oct': 4,
    'nov': 5,
    'dec': 6,
}


def haversine_np(p1, p2):
    """
    Calculate the great circle distance between two shapely points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    Parameters
    ----------
    p1, p2: shapely points

    Returns
    -------
    km: float
        the earth distance between the two points
    """
    lon1, lat1 = p1.x, p1.y
    lon2, lat2 = p2.x, p2.y
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def init(path, area, period, window, datetime_from, datetime_to, feature_type, clf_type, verbose=True):

    path_quadtree = path + 'quadtree/'
    path_dataset = path + 'dataset/'

    if verbose:
        print(datetime.datetime.now(), 'Partitioning periods ...', end='')
    months = pd.date_range(start=datetime_from, end=datetime_to, freq='MS')
    boundaries = [(lm, um) for lm, um in zip(months[:-window], months[window:])]
    test_months = list()
    for i in range(len(boundaries) - 1):
        test_months.append(boundaries[i + 1])

    index = 0
    ts_data_map = dict()
    ts_data_map_rev = dict()
    for ts_months in zip(test_months):
        ts_data_map[ts_months] = index
        ts_data_map_rev[index] = ts_months
        index += 1

    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Reading trained crash classifier ...', end='')
    sel_index = periods_map[period]
    clf = pickle.load(open(path + 'clf/crash_prediction_%s_%s_%s_%s_f1_0.pickle' % (
        area, sel_index, feature_type, clf_type), 'rb'))
    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Reading quadtree ...', end='')
    quadtree_poi_filename = path_quadtree + '%s_personal_osm_poi_lv17.json.gz' % area
    fout = gzip.GzipFile(quadtree_poi_filename, 'r')
    quadtree = json.loads(fout.readline())
    fout.close()
    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Reading quadtree features ...', end='')
    quadtree_features_filename = path_quadtree + '%s_quadtree_features.json.gz' % area
    fout = gzip.GzipFile(quadtree_features_filename, 'r')
    quadtrees_features_str = json.loads(fout.readline())
    quadtrees_features = {int(k): v for k, v in quadtrees_features_str.items()}
    fout.close()
    # quadtrees_features = None
    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Managing features\' names ...', end='')
    features_names = json.load(open(path_dataset + 'features_names.json', 'r'))
    features_map = {'t': 'traj', 'e': 'evnt', 'i': 'imn', 'c': 'col'}

    features = list()
    for ft in feature_type:
        if ft in features_map:
            features.extend(features_names[features_map[ft]])
    if verbose:
        print(' done.')

    res = {
        'ts_data_map': ts_data_map,
        'ts_data_map_rev': ts_data_map_rev,
        'clf': clf,
        'quadtree': quadtree,
        'quadtrees_features': quadtrees_features,
        'features': features,
    }

    return res


def read_data(path, uid, area, period, datetime_from, datetime_to, verbose=True, visual=False):
    path_dataset = path + 'dataset/'
    points_filename = 'points_%s.json' % uid
    events_filename = 'events_%s.json' % uid

    if not os.path.isfile(path_dataset + points_filename):
        print(datetime.datetime.now(), 'File %s not found.')
        return None

    if not os.path.isfile(path_dataset + events_filename):
        print(datetime.datetime.now(), 'File %s not found.')
        return None

    if verbose:
        print(datetime.datetime.now(), 'Reading posistions ...', end='')

    json_points = json.load(open(path_dataset + points_filename))
    mongolike_points = list()
    for p in json_points:
        dt = datetime.datetime.strptime(p['TIMESTAMP_LOCAL']['$date'][:-9].replace('T', ' '), '%Y-%m-%d %H:%M:%S')
        if dt < datetime_from or dt >= datetime_to:
            continue

        mongolike_points.append({
            'T&K_VOUCHER_ID': p['T&K_VOUCHER_ID']['$numberLong'],
            'TIMESTAMP_LOCAL': dt,
            'LATITUDE': p['LATITUDE'],
            'LONGITUDE': p['LONGITUDE'],
            'SPEED': p['SPEED'],
            'HEADING': p['HEADING'],
            'GPS_QUALITY': p['GPS_QUALITY'],
            'STATUS': p['STATUS'],
            'DELTAPOS': p['DELTAPOS'],
            'DELTATIME': p['DELTATIME'],
            'PV': p['PV'],
            'LOCATION_TYPE': p['LOCATION_TYPE'],
        })

    if verbose:
        print(' done.')
        print(datetime.datetime.now(), 'Reading events ...', end='')

    json_events_data = open(path_dataset + events_filename)
    mongolike_events = list()
    for row in json_events_data:
        e = json.loads(row)
        dt = datetime.datetime.strptime(e['TIMESTAMP_LOCAL']['$date'][:-9].replace('T', ' '), '%Y-%m-%d %H:%M:%S')
        if dt < datetime_from or dt >= datetime_to:
            continue

        mongolike_events.append({
            'T&K_VOUCHER_ID': e['T&K_VOUCHER_ID']['$numberLong'],
            'TIMESTAMP_LOCAL': dt,
            'LATITUDE': e['LATITUDE'],
            'LONGITUDE': e['LONGITUDE'],
            'SPEED': e['SPEED'],
            'HEADING': e['HEADING'],
            'GPS_QUALITY': e['GPS_QUALITY'],
            'STATUS': e['STATUS'],
            'PV': e['PV'],
            'LOCATION_TYPE': e['LOCATION_TYPE'],
            'EVENT_TYPE': e['EVENT_TYPE'],
            'AVG_ACCELERATION': e['AVG_ACCELERATION'],
            'MAX_ACCELERATION': e['MAX_ACCELERATION'],
            'EVENT_ANGLE': e['EVENT_ANGLE'],
            'DURATION': e['DURATION']
        })

    if verbose:
        print(' done.')
        print(datetime.datetime.now(), 'Retrieved %s positions, %s events' % (
            len(mongolike_points), len(mongolike_events)))

    res = {
        'points': mongolike_points,
        'events': mongolike_events,
    }

    if visual:
        path_visual = path + 'fig/crash/'
        points_filename = '%s_%s_%s_points.html' % (uid, area, period)
        points = np.array([[row['LONGITUDE'] / 1000000.0, row['LATITUDE'] / 1000000.0] for row in mongolike_points])
        visualize_points(path_visual + points_filename, points)

    return res


def build_traj(path, uid, area, period, traj_data, evnt_data, max_speed=0.07, space_treshold=0.05, time_treshold=1200,
               min_time_gap=300, min_length=0.0, min_duration=0.0, verbose=True, visual=False):

    if verbose:
        print(datetime.datetime.now(), 'Building trajectories from points ...', end='')

    evnt_data = iter(evnt_data)

    def next_evnt(evnt_data):
        try:
            evnt_row = next(evnt_data)
            if evnt_row is None:
                return None, None, None
            evnt_user = evnt_row['T&K_VOUCHER_ID']
            evnt_ts = evnt_row['TIMESTAMP_LOCAL']
            lon = evnt_row['LONGITUDE'] / 1000000.0
            lat = evnt_row['LATITUDE'] / 1000000.0
            event_type = evnt_row['EVENT_TYPE']
            speed = evnt_row['SPEED']
            max_acceleration = evnt_row['MAX_ACCELERATION']
            avg_acceleration = evnt_row['AVG_ACCELERATION']
            event_angle = evnt_row['EVENT_ANGLE']
            heading = evnt_row['HEADING']
            gps_quality = evnt_row['GPS_QUALITY']
            pv = evnt_row['PV']
            location_type = evnt_row['LOCATION_TYPE']
            duration = evnt_row['DURATION']
            status = evnt_row['STATUS']
            evnt_values = [lat, lon, evnt_ts, event_type, speed, max_acceleration, avg_acceleration, event_angle,
                           heading, gps_quality, pv, location_type, duration, status]
            return evnt_user, evnt_ts, evnt_values
        except StopIteration:
            return None, None, None
        except Exception:
            return None, None, None

    traj_datalist = list()
    evnt_datalist = list()

    count_traj = 0
    tot_points = 0
    count_points = 0
    user = None
    point = None
    ref_point = None  # for stop detection
    ref_ts = None  # for stop detection
    tid = 0
    is_a_new_traj = True
    count_omitted = 0

    traj = list()
    length = None

    evnt_id = 0
    tot_events = 1
    first_iteration = True
    evnt_user, evnt_ts, evnt_values = next_evnt(evnt_data)

    for row in traj_data:

        tot_points += 1
        user = row['T&K_VOUCHER_ID']
        next_ts = row['TIMESTAMP_LOCAL']
        next_status = row['STATUS']
        next_gps_quality = row['GPS_QUALITY']
        next_lon = row['LONGITUDE'] / 1000000.0
        next_lat = row['LATITUDE'] / 1000000.0

        # Filter out low gps quality points, if 'gps_quality' value is avaibale
        if next_gps_quality >= 3:
            lon = next_lon
            lat = next_lat
        elif not first_iteration and next_status == 0 and status == 2 and gps_quality >= 3 and not last_skipped:
            lon = lon
            lat = lat
        else:
            last_skipped = True
            continue

        count_points += 1
        low_quality_point_ts = 0
        last_skipped = False
        next_point = Point((lon, lat))

        if next_status == 0 or next_status == 2:
            event_type = 'start' if next_status == 0 else 'stop'
            evnt_datarecord = [user, tid, evnt_id] + [lat, lon, next_ts, event_type, -1, -1, -1, -1, -1,
                                                      next_gps_quality, -1, '-1', -1, next_status]
            evnt_datalist.append(evnt_datarecord)
            evnt_id += 1
            tot_events += 1

        if first_iteration:  # first iteration
            point = next_point
            ts = next_ts
            status = next_status
            gps_quality = next_gps_quality
            ref_point = point  # for stop detection
            ref_ts = ts  # for stop detection
            traj = [(ts, point)]
            length = 0.
            is_a_new_traj = True
            first_iteration = False
        else:  # all the others
            distance = haversine_np(point, next_point)
            # time_diff = (next_ts - ts).total_seconds()
            ref_time_diff = (next_ts - ref_ts).total_seconds()  # for stop detection
            ref_distance = haversine_np(ref_point, next_point)  # for stop detection

            last_gap = (next_ts - ts).total_seconds()

            # Ignore extreme jump (with speed > 250km/h = 0.07km/s)
            if ref_distance > max_speed * ref_time_diff:
                continue

            if last_gap > time_treshold or (ref_time_diff > time_treshold and ref_distance < space_treshold):
                # ended trajectory (includes case with long time gap)
                if len(traj) > 1 and not is_a_new_traj:
                    start_time = traj[0][0]
                    end_time = traj[-1][0]
                    duration = (end_time - start_time).total_seconds()
                    geometry = LineString([x[1] for x in traj])
                    times = [x[0] for x in traj]
                    traj_datarecord = [user, tid, geometry, times, length, duration,
                                       traj[0][1], traj[-1][1], start_time, end_time]
                    traj_datalist.append(traj_datarecord)

                    # scorro gli eventi finche' non ne raggiungo uno maggiore ugule di quello corrente
                    while evnt_user is not None and evnt_ts < start_time:
                        tot_events += 1
                        evnt_user, evnt_ts, evnt_values = next_evnt(evnt_data)
                        if evnt_user is None:
                            break

                    # aggiungo gli eventi alla traiettoria finche' non ne raggiungo uno maggiore (o cambia utente)
                    if evnt_user is not None:
                        while evnt_ts <= end_time:
                            evnt_datarecord = [user, tid, evnt_id] + evnt_values
                            evnt_datalist.append(evnt_datarecord)
                            evnt_id += 1
                            tot_events += 1
                            evnt_user, evnt_ts, evnt_values = next_evnt(evnt_data)
                            if evnt_user is None:
                                break

                # Create a new trajectory
                traj = [(ts, point)]  # 1st fake point with last position previous traj and new timestamp
                point = next_point
                ts = next_ts
                status = next_status
                gps_quality = next_gps_quality
                ref_point = point  # for stop detection
                ref_ts = ts  # for stop detection
                traj.append((ts, point))  # 1st real point with first position new traj and temporary new timestamp
                length = haversine_np(point, next_point)
                tid += 1
                is_a_new_traj = True
            else:
                if is_a_new_traj:  # 1st fake point correction by using timestamp interpolation
                    delta = next_ts - ts
                    ts0 = low_quality_point_ts if low_quality_point_ts > 0 else ts - delta
                    if ts0 > traj[0][0] + datetime.timedelta(seconds=min_time_gap):  # traj[0][0] contains last timestamp of last traj
                        traj[0] = (ts0, traj[0][1])  # add fake point
                    else:
                        traj = traj[1:]  # restart from detected real points
                        count_omitted += 1

                    is_a_new_traj = False

                point = next_point
                ts = next_ts
                status = next_status
                gps_quality = next_gps_quality
                # reset reference point for stop detection
                if ref_distance > space_treshold:
                    ref_point = point  # for stop detection
                    ref_ts = ts  # for stop detection
                traj.append((ts, point))
                length += distance

    if len(traj) > 1 and not is_a_new_traj:
        start_time = traj[0][0]
        end_time = traj[-1][0]
        duration = (end_time - start_time).total_seconds()

        geometry = LineString([x[1] for x in traj])
        times = [x[0] for x in traj]
        traj_datalist.append([user, tid, geometry, times, length, duration,
                              traj[0][1], traj[-1][1], start_time, end_time])

        # scorro gli eventi finche' non ne raggiungo uno maggiore ugule di quello corrente
        while evnt_user is not None and evnt_ts < start_time:
            tot_events += 1
            evnt_user, evnt_ts, evnt_values = next_evnt(evnt_data)
            if evnt_user is None:
                break

        # aggiungo gli eventi alla traiettoria finche' non ne raggiungo uno maggiore
        if evnt_user is not None:
            while evnt_ts <= end_time:
                evnt_datarecord = [user, tid, evnt_id] + evnt_values
                evnt_datalist.append(evnt_datarecord)
                evnt_id += 1
                tot_events += 1
                evnt_user, evnt_ts, evnt_values = next_evnt(evnt_data)
                if evnt_user is None:
                    break

    count_traj = count_traj + len(traj_datalist)

    trajectories = dict()
    for row in traj_datalist:
        t = list(row[2].coords)
        ts = row[3]
        traj_object = list()
        for i in range(len(t)):
            traj_object.append((t[i][0], t[i][1], time.mktime(ts[i].timetuple()) / 1000.0))

        traj = Trajectory(id=str(row[1]), object=traj_object, vehicle=str(row[0]),
                          length=float(row[4]), duration=float(row[5]), start_time=row[8], end_time=row[9])

        if traj.length() > min_length and traj.duration() > min_duration:
            trajectories[str(row[1])] = traj

    events = defaultdict(list)
    for row in evnt_datalist:
        event = {
            'uid': str(row[0]),
            'tid': str(row[1]),
            'eid': str(row[2]),
            'event_type': str(row[6]),
            'speed': int(row[7]),
            'max_acc': int(row[8]),
            'avg_acc': int(row[9]),
            'angle': int(row[10]),
            'location_type': str(row[14]),
            'duration': int(row[15]),
            'date': row[5],
            'lat': float(row[3]),
            'lon': float(row[4]),
        }
        events[str(row[2])].append(event)

    if verbose:
        print(' done.')
        print(datetime.datetime.now(),
              'Reconstructed %d trajectories using %d points out of %d.' % (count_traj, count_points, tot_points))

    if visual:
        path_visual = path + 'fig/crash/'
        traj_filename = '%s_%s_%s_traj.html' % (uid, area, period)
        visualize_trajectories(path_visual + traj_filename, trajectories)
        stop_filename = '%s_%s_%s_stops.html' % (uid, area, period)
        visualize_stops(path_visual + stop_filename, trajectories)

    return trajectories, events


def build_imns(path, uid, area, period, trajectories, events, datetime_from, datetime_to, min_nbr_traj=100,
               verbose=True, visual=False):

    if len(trajectories) < min_nbr_traj:
        print(datetime.datetime.now(), 'Not enough trajectories for building IMNs.')
        return None

    if verbose:
        print(datetime.datetime.now(), 'Building IMNS from trajectories ...', end='')

    def start_time_map(t):
        if t - relativedelta(months=1) < datetime_from or t + relativedelta(months=1) >= datetime_to:
            return None, None
        m = t.month
        if m == 1:
            return '01-02', None
        elif m == 12:
            return '11-12', None
        else:
            return '%02d-%02d' % (m - 1, m), '%02d-%02d' % (m, m + 1)

    def build_imn_thred(values):
        wimh, wevents, stk, visual = values
        imn = build_imn(wimh, reg_loc=True, events=wevents, verbose=False)

        if verbose:
            print(' %s' % stk, end=',')

        if visual:
            location_prototype = imn['location_prototype']
            location_features = imn['location_features']
            location_nextlocs = imn['location_nextlocs']

            path_visual = path + 'fig/crash/'

            locs_filename = '%s_%s_%s_%s_locs.html' % (uid, area, period, stk)
            visualize_locations(path_visual + locs_filename, location_prototype, location_features)

            imns_filename = '%s_%s_%s_%s_imn.html' % (uid, area, period, stk)
            visualize_imn(path_visual + imns_filename, location_nextlocs, location_prototype, location_features)

        return imn

    wimh_dict = dict()
    wevents_dict = dict()
    for tid, traj in trajectories.items():
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

    tp = ThreadPool(len(wimh_dict) + 1)
    values = list()
    for stk in wimh_dict:
        values.append([wimh_dict[stk], wevents_dict[stk], stk, visual])

    pool_output = tp.map(build_imn_thred, values)
    imn_list = dict()
    for i, stk in enumerate(wimh_dict):
        imn_list[stk] = pool_output[i]

    if verbose:
        print(' done.')

    # for stk in wimh_dict:
    #
    #     if verbose:
    #         print(' %s' % stk, end=',')
    #
    #     wimh = wimh_dict[stk]
    #     wevents = wevents_dict[stk]
    #
    #     imn = build_imn(wimh, reg_loc=True, events=wevents, verbose=False)
    #     imn_list[stk] = imn
    #
    # if verbose:
    #     print(' done.')
    #
    # if visual:
    #     for stk, imn in imn_list.items():
    #         if stk == 'uid':
    #             continue
    #         location_prototype = imn['location_prototype']
    #         location_features = imn['location_features']
    #         location_nextlocs = imn['location_nextlocs']
    #
    #         path_visual = path + 'fig/crash/'
    #
    #         locs_filename = '%s_%s_%s_%s_locs.html' % (uid, area, period, stk)
    #         visualize_locations(path_visual + locs_filename, location_prototype, location_features)
    #
    #         imns_filename = '%s_%s_%s_%s_imn.html' % (uid, area, period, stk)
    #         visualize_imn(path_visual + imns_filename, location_nextlocs, location_prototype, location_features)

    return imn_list


def prepare_data4feature_extraction(uid, trajectories, events, imn_list, data_map, sel_index, verbose=True):

    if verbose:
        print(datetime.datetime.now(), 'Preparing data for feature extraction ...', end='')

    data = dict()
    # partitioning imn for train and test
    for imn_months in imn_list:
        if imn_months == 'uid':
            continue

        m0 = int(imn_months.split('-')[0])
        m1 = int(imn_months.split('-')[1])
        for lut, index in data_map.items():
            if index != sel_index:
                continue
            lu = lut[0]
            if lu[0].month <= m0 < m1 < lu[1].month:
                if index not in data:
                    data[index] = {'uid': uid, 'crash': False, 'trajectories': dict(),
                                   'imns': dict(), 'events': dict(), }
                data[index]['imns'][imn_months] = imn_list[imn_months]

    # partitioning trajectories for train and test
    for tid, traj in trajectories.items():
        for lut, index in data_map.items():
            if index != sel_index:
                continue
            lu = lut[0]
            if lu[0] <= traj.start_time() < lu[1] and index in data:
                data[index]['trajectories'][tid] = traj

    # partitioning events for train and test
    for eid, evnt in events.items():
        for lut, index in data_map.items():
            if index != sel_index:
                continue
            lu = lut[0]
            if lu[0] <= evnt[0]['date'] < lu[1] and index in data:
                data[index]['events'][eid] = evnt[0]

    if verbose:
        print(' done.')

    return data


def extract_features(uid, data, quadtree, quadtree_features, verbose=True):
    if verbose:
        print(datetime.datetime.now(), 'Extracting features ...', end='')
    features = extract_features_data(uid, data, quadtree, quadtree_features)
    if verbose:
        print(' done.')
    return features


def prepare_features4classification(path, uid, area, period, sel_index, user_features, features, visual):

    mms = MinMaxScaler()
    df_train = pd.read_csv(path + 'clf/%s_train_%s.csv.gz' % (area, sel_index))
    df_train.set_index('uid', inplace=True)
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train.fillna(0, inplace=True)
    df_train = df_train.reset_index().drop_duplicates(subset='uid', keep='first').set_index('uid')
    X_train = df_train[features].values
    mms.fit(X_train)

    for f in features:
        if f not in user_features:
            user_features[f] = 0.0

    df = pd.DataFrame(data=user_features, index=[0])
    df.set_index('uid', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    X = df[features].values
    X = mms.transform(X)

    if visual:
        path_dataset = path + 'dataset/'
        features_names = json.load(open(path_dataset + 'features_names.json', 'r'))
        features_filename = '%s_%s_%s_features.png' % (uid, area, period)
        path_visual = path + 'fig/crash/'
        visualize_features(path_visual + features_filename, user_features, df_train, features_names)

    return X


def main():

    uid = sys.argv[1]
    area = sys.argv[2]            # 'rome' //'tuscany' 'london'
    period = sys.argv[3]          # jun, jul, aug, sep, oct, nov, dec
    feature_type = sys.argv[4]    # 'teic'
    clf_type = sys.argv[5]        # 'RF'

    # area = 'rome'
    # uid = 1013
    # period = 'jul'
    # feature_type = 'teic'
    # clf_type = 'RF'

    verbose = True
    visual = True

    path = '/Users/riccardo/Documents/PhD/TrackAndKnow/'

    window = 4
    datetime_from = datetime.datetime.strptime('2017-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    datetime_to = datetime.datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

    min_nbr_traj = 100
    min_length = 1.0
    min_duration = 60.0

    print(datetime.datetime.now(), 'Track & Know - Crash Prediction')
    print(datetime.datetime.now(), 'User %s - %s - %s' % (uid, period.capitalize(), area.capitalize()))

    res = init(path, area, period, window, datetime_from, datetime_to, feature_type, clf_type, verbose=verbose)
    data_map = res['ts_data_map']
    data_map_rev = res['ts_data_map_rev']
    clf = res['clf']
    quadtree = res['quadtree']
    quadtree_features = res['quadtrees_features']
    features = res['features']

    period_from, period_to = data_map_rev[periods_map[period]][0]
    period_from = period_from.to_pydatetime()
    period_to = period_to.to_pydatetime()

    res = read_data(path, uid, area, period, period_from, period_to, verbose=verbose, visual=visual)
    if res is None:
        return -1

    traj_data = res['points']
    evnt_data = res['events']

    trajectories, events = build_traj(path, uid, area, period, traj_data, evnt_data, max_speed=0.07,
                                      space_treshold=0.05, time_treshold=1200, min_time_gap=300,
                                      min_length=min_length, min_duration=min_duration, verbose=verbose, visual=visual)

    imn_list = build_imns(path, uid, area, period, trajectories, events, period_from, period_to,
                          min_nbr_traj=min_nbr_traj, verbose=verbose, visual=visual)
    if imn_list is None:
        return -1

    data = prepare_data4feature_extraction(uid, trajectories, events, imn_list,
                                           data_map, periods_map[period], verbose=verbose)
    user_features = extract_features(uid, data, quadtree, quadtree_features, verbose=verbose)[periods_map[period]]

    if verbose:
        print(datetime.datetime.now(), 'Running crash prediction ...', end='')
    X = prepare_features4classification(path, uid, area, period, periods_map[period], user_features, features,
                                        visual=visual)
    Y = clf.predict(X)
    Y_proba = clf.predict_proba(X)
    if verbose:
        print(' done.')

    # crash_flag = Y[0]
    crash_proba = Y_proba[0][1] + 0.1
    crash_flag = int(np.round(crash_proba))

    if crash_flag:
        print(datetime.datetime.now(),
              'User %s is going to have a crash in %s in %s (crash probability %.2f)' % (
                  uid, period.capitalize(), area.capitalize(), crash_proba))
    else:
        print(datetime.datetime.now(),
              'User %s is not going to have a crash in %s in %s (crash probability %.2f)' % (
                  uid, period.capitalize(), area.capitalize(), crash_proba))

    if visual:
        path_visual = path + 'fig/crash/'
        risk_filename = '%s_%s_%s_crash_risk.png' % (uid, area, period)
        visualize_crash_risk(path_visual + risk_filename, uid, area, period, crash_proba, path)

    # 0. initialize
    # 1. read traj and evnts from file like mongo
    # 2. filter with respect to desidered period
    # 3. reconstruct traj
    # 6. build imn
    # 7. extract features
    # 8. generate line for classifier
    # 9. classify and return crash risk


if __name__ == "__main__":
    main()
