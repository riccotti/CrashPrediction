import gzip
import json
import datetime
import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from networkx.readwrite import json_graph

from individual_mobility_network import entropy as calculate_entropy
from mobility_distance_functions import spherical_distance
from tak_quadtree import lon_lat_to_quadtree_path


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


def get_timeday(time):
    morning_from = datetime.datetime.strptime('06', '%H').time()     # 6 hours
    afternoon_from = datetime.datetime.strptime('12', '%H').time()   # 6 hours
    evening_from = datetime.datetime.strptime('18', '%H').time()     # 4 hours
    night_from = datetime.datetime.strptime('22', '%H').time()       # 8 hours
    if morning_from <= time < afternoon_from:
        return 0
    elif afternoon_from <= time < evening_from:
        return 1
    elif evening_from <= time < night_from:
        return 2
    else:
        return 3


def get_trjectory_features(trajectories):

    km_list = list()
    traveltime_list = list()
    speed_list = list()

    traj_per_day = defaultdict(int)
    km_per_day = defaultdict(float)
    traveltime_per_day = defaultdict(float)
    max_speed_per_day = dict()
    min_speed_per_day = dict()

    traj_per_timeday = defaultdict(int)
    km_per_timeday = defaultdict(float)
    traveltime_per_timeday = defaultdict(float)
    speed_per_timeday = defaultdict(list)

    for traj in trajectories.values():
        km = traj.length()
        duration = traj.duration()
        speed = km / (duration/3600)
        day = traj.start_time().date()
        time = traj.start_time().time()
        timeday = get_timeday(time)

        km_list.append(km)
        traveltime_list.append(duration)
        speed_list.append(speed)

        traj_per_day[day] += 1
        km_per_day[day] += km
        traveltime_per_day[day] += duration
        max_speed_per_day[day] = max(speed, max_speed_per_day[day]) if day in max_speed_per_day else speed
        min_speed_per_day[day] = min(speed, min_speed_per_day[day]) if day in min_speed_per_day else speed

        traj_per_timeday[timeday] += 1
        km_per_timeday[timeday] += km
        traveltime_per_timeday[timeday] += duration
        speed_per_timeday[timeday].append(speed)

    features = {
        'tot_traj': len(trajectories),
        'tot_km': np.sum(km_list),
        'tot_traveltime': np.sum(traveltime_list),
        'avg_km': np.mean(km_list),
        'avg_traveltime': np.mean(traveltime_list),
        'avg_speed': np.mean(speed_list),
        'std_km': np.std(km_list),
        'std_traveltime': np.std(traveltime_list),
        'std_speed': np.std(speed_list),

        'avg_traj_per_day': np.mean(list(traj_per_day.values())),
        'avg_km_per_day': np.mean(list(km_per_day.values())),
        'avg_traveltime_per_day': np.mean(list(traveltime_per_day.values())),
        'avg_max_speed_per_day': np.mean(list(max_speed_per_day.values())),
        'avg_min_speed_per_day': np.mean(list(min_speed_per_day.values())),
        'std_traj_per_day': np.std(list(traj_per_day.values())),
        'std_km_per_day': np.std(list(km_per_day.values())),
        'std_traveltime_per_day': np.std(list(traveltime_per_day.values())),
        'std_max_speed_per_day': np.std(list(max_speed_per_day.values())),
        'std_min_speed_per_day': np.std(list(min_speed_per_day.values())),

        'morning_traj': traj_per_timeday.get(0, 0),
        'afternoon_traj': traj_per_timeday.get(1, 0),
        'evening_traj': traj_per_timeday.get(2, 0),
        'night_traj': traj_per_timeday.get(3, 0),
        'morning_ntraj': traj_per_timeday.get(0, 0) / len(trajectories),
        'afternoon_ntraj': traj_per_timeday.get(1, 0) / len(trajectories),
        'evening_ntraj': traj_per_timeday.get(2, 0) / len(trajectories),
        'night_ntraj': traj_per_timeday.get(3, 0) / len(trajectories),

        'morning_km': km_per_timeday.get(0, 0.0),
        'afternoon_km': km_per_timeday.get(1, 0.0),
        'evening_km': km_per_timeday.get(2, 0.0),
        'night_km': km_per_timeday.get(3, 0.0),
        'morning_nkm': km_per_timeday.get(0, 0.0) / np.sum(km_list),
        'afternoon_nkm': km_per_timeday.get(1, 0.0) / np.sum(km_list),
        'evening_nkm': km_per_timeday.get(2, 0.0) / np.sum(km_list),
        'night_nkm': km_per_timeday.get(3, 0.0) / np.sum(km_list),

        'morning_ttime': traveltime_per_timeday.get(0, 0.0),
        'afternoon_ttime': traveltime_per_timeday.get(1, 0.0),
        'evening_ttime': traveltime_per_timeday.get(2, 0.0),
        'night_ttime': traveltime_per_timeday.get(3, 0.0),
        'morning_nttime': traveltime_per_timeday.get(0, 0.0) / np.sum(traveltime_list),
        'afternoon_nttime': traveltime_per_timeday.get(1, 0.0) / np.sum(traveltime_list),
        'evening_nttime': traveltime_per_timeday.get(2, 0.0) / np.sum(traveltime_list),
        'night_nttime': traveltime_per_timeday.get(3, 0.0) / np.sum(traveltime_list),

        'morning_avg_speed': np.mean(speed_per_timeday[0]) if 0 in speed_per_timeday else -1,
        'afternoon_avg_speed': np.mean(speed_per_timeday[1]) if 1 in speed_per_timeday else -1,
        'evening_avg_speed': np.mean(speed_per_timeday[2]) if 2 in speed_per_timeday else -1,
        'night_avg_speed': np.mean(speed_per_timeday[3]) if 3 in speed_per_timeday else -1,
        'morning_std_speed': np.std(speed_per_timeday[0]) if 0 in speed_per_timeday else -1,
        'afternoon_std_speed': np.std(speed_per_timeday[1]) if 1 in speed_per_timeday else -1,
        'evening_std_speed': np.std(speed_per_timeday[2]) if 2 in speed_per_timeday else -1,
        'night_std_speed': np.std(speed_per_timeday[3]) if 3 in speed_per_timeday else -1,

    }

    return features


def get_events_features(events):
    nbr_events_per_day = defaultdict(int)
    nbr_events_per_time = defaultdict(int)

    nbr_events_type = defaultdict(int)
    nbr_event_type_per_day = defaultdict(lambda: defaultdict(int))
    nbr_event_type_per_time = defaultdict(lambda: defaultdict(int))

    nbr_event_location = defaultdict(int)
    nbr_event_location_per_day = defaultdict(lambda: defaultdict(int))
    nbr_event_location_per_time = defaultdict(lambda: defaultdict(int))

    nbr_events_type_locations = defaultdict(lambda: defaultdict(int))

    durations_list = list()
    durations_event_type = defaultdict(list)
    durations_event_location = defaultdict(list)

    avg_acc_list = list()
    avg_acc_event_type = defaultdict(list)
    avg_acc_event_location = defaultdict(list)

    max_acc_list = list()
    max_acc_event_type = defaultdict(list)
    max_acc_event_location = defaultdict(list)

    angle_list = list()
    angle_event_type = defaultdict(list)
    angle_event_location = defaultdict(list)

    for event in events.values():
        day = event['date'].date()
        time = event['date'].time()
        event_type = event['event_type']
        duration = event['duration']
        location = event['location_type']
        avg_acc = event['avg_acc']
        max_acc = event['max_acc']
        angle = event['angle']

        nbr_events_per_day[day] += 1
        nbr_events_per_time[time] += 1

        nbr_events_type[event_type] += 1
        nbr_event_type_per_day[event_type][day] += 1
        nbr_event_type_per_time[event_type][time] += 1

        nbr_event_location[location] += 1
        nbr_event_location_per_day[location][day] += 1
        nbr_event_location_per_time[location][time] += 1

        nbr_events_type_locations[event_type][location] += 1

        durations_list.append(duration)
        durations_event_type[event_type].append(duration)
        durations_event_location[location].append(duration)

        avg_acc_list.append(avg_acc)
        avg_acc_event_type[event_type].append(avg_acc)
        avg_acc_event_location[location].append(avg_acc)

        max_acc_list.append(max_acc)
        max_acc_event_type[event_type].append(max_acc)
        max_acc_event_location[location].append(max_acc)

        angle_list.append(angle)
        angle_event_type[event_type].append(angle)
        angle_event_location[location].append(angle)

    features = {
        'tot_events': len(events),
        'avg_events_per_day': np.mean(list(nbr_events_per_day.values())),
        'std_events_per_day': np.std(list(nbr_events_per_day.values())),
        'avg_events_per_time': np.mean(list(nbr_events_per_time.values())),
        'std_events_per_time': np.std(list(nbr_events_per_time.values())),
        'tot_duration': np.sum(durations_list),
        'avg_duration': np.mean(durations_list),
        'std_duration': np.std(durations_list),
        'avg_avg_acc': np.mean(avg_acc_list),
        'std_avg_acc': np.std(avg_acc_list),
        'avg_max_acc': np.mean(max_acc_list),
        'std_max_acc': np.std(max_acc_list),
        'avg_angle': np.mean(angle_list),
        'std_angle': np.std(angle_list),
    }

    for event_type in ['Q', 'B', 'A', 'C', 'stop', 'start']:
        features['tot_events_%s' % event_type] = nbr_events_type.get(event_type, 0)
        features['tot_events_p%s' % event_type] = \
            nbr_events_type.get(event_type, 0) / len(events) if len(events) > 0 else 0.0
        features['avg_events_per_day_%s' % event_type] = np.mean(
            list(nbr_event_type_per_day[event_type].values())) if event_type in nbr_event_type_per_day else -1
        features['avg_events_per_time_%s' % event_type] = np.mean(
            list(nbr_event_type_per_time[event_type].values())) if event_type in nbr_event_type_per_time else -1
        features['std_events_per_day_%s' % event_type] = np.std(
            list(nbr_event_type_per_day[event_type].values())) if event_type in nbr_event_type_per_day else -1
        features['std_events_per_time_%s' % event_type] = np.std(
            list(nbr_event_type_per_time[event_type].values())) if event_type in nbr_event_type_per_time else -1

        features['tot_duration_%s' % event_type] = np.sum(
            durations_event_type[event_type]) if event_type in durations_event_type else 0
        features['tot_duration_p%s' % event_type] = np.sum(
            durations_event_type[event_type]) / np.sum(durations_list) if event_type in durations_event_type else 0
        features['avg_duration_%s' % event_type] = np.mean(
            durations_event_type[event_type]) if event_type in durations_event_type else -1
        features['std_duration_%s' % event_type] = np.std(
            durations_event_type[event_type]) if event_type in durations_event_type else -1

        features['avg_avg_acc_%s' % event_type] = np.mean(
            avg_acc_event_type[event_type]) if event_type in avg_acc_event_type else -1
        features['std_avg_acc_%s' % event_type] = np.std(
            avg_acc_event_type[event_type]) if event_type in avg_acc_event_type else -1

        features['avg_max_acc_%s' % event_type] = np.mean(
            max_acc_event_type[event_type]) if event_type in max_acc_event_type else -1
        features['std_max_acc_%s' % event_type] = np.std(
            max_acc_event_type[event_type]) if event_type in max_acc_event_type else -1

        features['avg_angle_%s' % event_type] = np.mean(
            angle_event_type[event_type]) if event_type in angle_event_type else -1
        features['std_angle_%s' % event_type] = np.std(
            angle_event_type[event_type]) if event_type in angle_event_type else -1

    for location in ['0', '1', '2']:
        features['tot_events_loc%s' % location] = nbr_event_location.get(location, 0)
        features['tot_events_ploc%s' % location] = \
            nbr_event_location.get(location, 0) / len(events) if len(events) > 0 else 0.0
        features['avg_events_per_day_loc%s' % location] = np.mean(
            list(nbr_event_location_per_day[location].values())) if location in nbr_event_location_per_day else -1
        features['avg_events_per_time_loc%s' % location] = np.mean(
            list(nbr_event_location_per_time[location].values())) if location in nbr_event_location_per_time else -1
        features['std_events_per_day_loc%s' % location] = np.std(
            list(nbr_event_location_per_day[location].values())) if location in nbr_event_location_per_day else -1
        features['std_events_per_time_loc%s' % location] = np.std(
            list(nbr_event_location_per_time[location].values())) if location in nbr_event_location_per_time else -1

        features['tot_duration_loc%s' % location] = np.sum(
            durations_event_location[location]) if location in durations_event_location else 0
        features['tot_duration_ploc%s' % location] = np.sum(
            durations_event_location[location]) / np.sum(durations_list) if location in durations_event_location else 0
        features['avg_duration_loc%s' % location] = np.mean(
            durations_event_location[location]) if location in durations_event_location else -1
        features['std_duration_loc%s' % location] = np.std(
            durations_event_location[location]) if location in durations_event_location else -1

        features['avg_avg_acc_loc%s' % location] = np.mean(
            avg_acc_event_location[location]) if location in avg_acc_event_location else -1
        features['std_avg_acc_loc%s' % location] = np.std(
            avg_acc_event_location[location]) if location in avg_acc_event_location else -1

        features['avg_max_acc_loc%s' % location] = np.mean(
            max_acc_event_location[location]) if location in max_acc_event_location else -1
        features['std_max_acc_loc%s' % location] = np.std(
            max_acc_event_location[location]) if location in max_acc_event_location else -1

        features['avg_angle_loc%s' % location] = np.mean(
            angle_event_location[location]) if location in angle_event_location else -1
        features['std_angle_loc%s' % location] = np.std(
            angle_event_location[location]) if location in angle_event_location else -1

    for event_type in ['Q', 'B', 'A', 'C', 'stop', 'start']:
        if event_type in nbr_events_type_locations:
            for location in ['0', '1', '2']:
                if location in nbr_events_type_locations[event_type]:
                    features['tot_events_type_%s_loc%s' % (event_type, location)] = \
                        nbr_events_type_locations[event_type][location]
                    features['ptot_events_type_%s_loc%s' % (event_type, location)] = \
                        nbr_events_type_locations[event_type][location] / len(events)
                else:
                    features['tot_events_type_%s_loc%s' % (event_type, location)] = 0
                    features['ptot_events_type_%s_loc%s' % (event_type, location)] = 0.0
        else:
            for location in ['0', '1', '2']:
                features['tot_events_type_%s_loc%s' % (event_type, location)] = 0
                features['ptot_events_type_%s_loc%s' % (event_type, location)] = 0.0

    return features


def string2timedelta(s):
    if isinstance(s, str):
        t = datetime.datetime.strptime(s, '%H:%M:%S.%f') if '.' in s else datetime.datetime.strptime(s, '%H:%M:%S')
        return datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    else:
        return s


def get_imn_temporal_features(imn_list, loc_dist_thr=100):

    imn_keys_sorted = sorted(list(imn_list.keys()))

    imn_key0 = imn_keys_sorted[0]
    imn_key1 = imn_keys_sorted[-1]
    imn0 = imn_list[imn_key0]
    imn1 = imn_list[imn_key1]
    if imn0 is None or imn1 is None:
        features = {
            'delta_locations': np.nan,
            'delta_movements': np.nan,
            'delta_reg_locations': np.nan,
            'delta_rg': np.nan,
            'jaccard': np.nan,
            'cosine': np.nan,
            'jaccard_mov': np.nan,
            'cosine_mov': np.nan,
        }

        return features

    delta_locations = imn1['n_locs'] - imn0['n_locs']
    delta_movements = imn1['n_movs'] - imn0['n_movs']
    delta_reg_locations = imn1['n_reg_locs'] - imn0['n_reg_locs']
    delta_rg = imn1['rg'] - imn0['rg']

    imn0loc = np.array([np.array(v) for v in imn0['location_prototype'].values()])
    imn1loc = np.array([np.array(v) for v in imn1['location_prototype'].values()])
    loc0 = np.arange(0, len(imn0loc))
    loc1 = np.arange(0, len(imn1loc)) + max(loc0) + 1
    sup0 = np.array([lf['loc_support'] for lid, lf in imn0['location_features'].items()])
    sup1 = np.array([lf['loc_support'] for lid, lf in imn1['location_features'].items()])

    graph0 = imn0['graph']
    graph1 = imn1['graph']
    if not isinstance(graph0, nx.DiGraph):
        graph0 = json_graph.node_link_graph(imn0['graph'], directed=True, multigraph=False,
                                            attrs={'link': 'edges', 'source': 'from', 'target': 'to'})

    if not isinstance(graph1, nx.DiGraph):
        graph1 = json_graph.node_link_graph(imn1['graph'], directed=True, multigraph=False,
                                            attrs={'link': 'edges', 'source': 'from', 'target': 'to'})

    sup0mov = np.array([mf['mov_support'] for mid, mf in imn0['mov_features'].items()])
    sup1mov = np.array([mf['mov_support'] for mid, mf in imn1['mov_features'].items()])

    dmatrix = cdist(imn0loc, imn1loc, metric=spherical_distance)
    mapping = dict()
    mapping_loc = dict()
    for j, i in enumerate(np.argmin(dmatrix, axis=0)):
        if dmatrix[i][j] < loc_dist_thr:
            if i not in mapping or dmatrix[i][j] < dmatrix[i][mapping[i]]:
                mapping[i] = j
                mapping_loc[loc0[i]] = loc1[j]

    for lid in loc0:
        if lid not in mapping_loc:
            mapping_loc[lid] = lid

    loc0to1set = set([mapping_loc[lid] for lid in loc0])
    loc1set = set(loc1)
    jaccard = 1 - len(loc0to1set & loc1set) / len(loc0to1set | loc1set)

    sup0remap = np.zeros(max(loc1) + 1)
    sup1remap = np.zeros(max(loc1) + 1)
    for i, v in enumerate(sup0):
        sup0remap[mapping_loc[i]] = v
    for j, v in enumerate(sup1):
        sup1remap[loc1[j]] = v

    cosined = cosine(sup0remap, sup1remap)
    graph0to1 = nx.relabel_nodes(graph0, mapping_loc, copy=True)
    mov0to1set = set(graph0to1.edges())
    mov1set = set(graph1.edges())
    jaccard_mov = 1 - len(mov0to1set & mov1set) / len(mov0to1set | mov1set)

    sup0mov_remap = np.zeros(len(mov0to1set | mov1set))
    sup1mov_remap = np.zeros(len(mov0to1set | mov1set))
    for lflt in graph0.edges():
        lflt_key = lflt
        if isinstance(list(imn0['location_from_to_movement'].keys())[0], str):
            lflt_key = str(lflt)
        mid = imn0['location_from_to_movement'][lflt_key]
        lflt1 = (mapping_loc[lflt[0]], mapping_loc[lflt[1]])
        index = list(graph0to1.edges()).index(lflt1)
        sup0mov_remap[index] = sup0mov[mid]
    for lflt in graph1.edges():
        lflt_key = lflt
        if isinstance(list(imn1['location_from_to_movement'].keys())[0], str):
            lflt_key = str(lflt)
        mid = imn1['location_from_to_movement'][lflt_key]
        index = list(graph1.edges()).index(lflt)
        sup1mov_remap[index] = sup1mov[mid]
    cosined_mov = cosine(sup0mov_remap, sup1mov_remap)

    features = {
        'delta_locations': delta_locations,
        'delta_movements': delta_movements,
        'delta_reg_locations': delta_reg_locations,
        'delta_rg': delta_rg,
        'jaccard': jaccard,
        'cosine': cosined,
        'jaccard_mov': jaccard_mov,
        'cosine_mov': cosined_mov,
    }

    return features


def get_imn_features(imn_list, event_traj2evntlist):

    nbr_locations = list()
    nbr_movements = list()
    nbr_reg_locations = list()
    nbr_reg_movements = list()
    radius_of_gyration = list()
    regular_radius_of_gyration = list()
    entropy = list()
    rentropy = list()
    avg_mov_length = list()
    std_mov_length = list()
    avg_mov_duration = list()
    std_mov_duration = list()
    avg_reg_mov_length = list()
    std_reg_mov_length = list()
    avg_reg_mov_duration = list()
    std_reg_mov_duration = list()

    density = list()
    triangles = list()
    clustering_coefficient = list()
    degree = list()
    indegree = list()
    outdegree = list()
    diameter = list()
    eccentricity = list()
    assortativity = list()

    l1_count = list()
    l2_count = list()
    l3_count = list()
    l1_indegree = list()
    l2_indegree = list()
    l3_indegree = list()
    l1_outdegree = list()
    l2_outdegree = list()
    l3_outdegree = list()
    l1_dcentrality = list()
    l2_dcentrality = list()
    l3_dcentrality = list()
    l1_bcentrality = list()
    l2_bcentrality = list()
    l3_bcentrality = list()
    l1_events = defaultdict(list)
    l2_events = defaultdict(list)
    l3_events = defaultdict(list)

    l1l2_count = list()
    l2l1_count = list()
    l1l3_count = list()
    l3l1_count = list()
    l2l3_count = list()
    l3l2_count = list()
    l1l2_betweenness = list()
    l2l1_betweenness = list()
    l1l3_betweenness = list()
    l3l1_betweenness = list()
    l2l3_betweenness = list()
    l3l2_betweenness = list()
    l1l2_events = defaultdict(list)
    l2l1_events = defaultdict(list)
    l1l3_events = defaultdict(list)
    l3l1_events = defaultdict(list)
    l2l3_events = defaultdict(list)
    l3l2_events = defaultdict(list)

    mov_event_entropy = defaultdict(list)

    for m0m1, imn in imn_list.items():
        if imn is None:
            continue
        # print(m0m1, imn.keys())
        # print(json.dumps(clear_tuples4json(imn), default=agenda_converter))
        nbr_locations.append(imn['n_locs'])
        nbr_movements.append(imn['n_movs'])
        nbr_reg_locations.append(imn['n_reg_locs'])
        nbr_reg_movements.append(imn['n_reg_movs'])
        radius_of_gyration.append(imn['rg'])
        regular_radius_of_gyration.append(imn['rrg'])
        entropy.append(imn['entropy'])
        rentropy.append(imn['rentropy'])
        avg_mov_length.append(imn['avg_mov_length'])
        std_mov_length.append(imn['std_mov_length'])
        avg_mov_duration.append(string2timedelta(imn['avg_mov_duration']).total_seconds())
        std_mov_duration.append(string2timedelta(imn['std_mov_duration']).total_seconds())
        avg_reg_mov_length.append(imn['avg_reg_mov_length'])
        std_reg_mov_length.append(imn['std_reg_mov_length'])
        avg_reg_mov_duration.append(string2timedelta(imn['avg_reg_mov_duration']).total_seconds())
        std_reg_mov_duration.append(string2timedelta(imn['std_reg_mov_duration']).total_seconds())

        graph = imn['graph']
        if not isinstance(graph, nx.DiGraph):
            graph = json_graph.node_link_graph(imn['graph'], directed=True, multigraph=False,
                                               attrs={'link': 'edges', 'source': 'from', 'target': 'to'})
        density.append(nx.density(graph))
        triangles.append(np.mean(list(nx.triangles(nx.to_undirected(graph)).values())))
        clustering_coefficient.append(nx.average_clustering(graph))
        degree.append(np.mean(list(dict(nx.to_undirected(graph).degree()).values())))
        indegree.append(np.mean(list(dict(graph.in_degree()).values())))
        outdegree.append(np.mean(list(dict(graph.out_degree()).values())))
        if nx.is_connected(nx.to_undirected(graph)):
            diameter.append(nx.diameter(nx.to_undirected(graph)))
            eccentricity.append(np.mean(list(nx.eccentricity(nx.to_undirected(graph)).values())))
            assortativity.append(nx.degree_assortativity_coefficient(nx.to_undirected(graph)))
        else:
            Gc = max(nx.connected_component_subgraphs(nx.to_undirected(graph)), key=len)
            diameter.append(nx.diameter(Gc))
            eccentricity.append(np.mean(list(nx.eccentricity(Gc).values())))
            assortativity.append(nx.degree_assortativity_coefficient(Gc))

        # print(imn['location_features'].keys())
        # print(list(imn['location_features'].keys())[0], type(list(imn['location_features'].keys())[0]))
        if isinstance(list(imn['location_features'].keys())[0], int):
            l1, l2, l3 = 0, 1, 2
        else:
            l1, l2, l3 = '0', '1', '2'

        l1_count.append(imn['location_features'][l1]['loc_support'])
        l2_count.append(imn['location_features'][l2]['loc_support'])
        if l3 in imn['location_features']:
            l3_count.append(imn['location_features'][l3]['loc_support'])
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())
        dcentrality = nx.degree_centrality(graph)
        bcentrality = nx.betweenness_centrality(graph)
        l1_indegree.append(in_degree[0])
        l2_indegree.append(in_degree[1])
        if 2 in in_degree:
            l3_indegree.append(in_degree[2])
        l1_outdegree.append(out_degree[0])
        l2_outdegree.append(out_degree[1])
        if 2 in out_degree:
            l3_outdegree.append(out_degree[2])
        l1_dcentrality.append(dcentrality[0])
        l2_dcentrality.append(dcentrality[1])
        if 2 in dcentrality:
            l3_dcentrality.append(dcentrality[2])
        l1_bcentrality.append(bcentrality[0])
        l2_bcentrality.append(bcentrality[1])
        if 2 in bcentrality:
            l3_bcentrality.append(bcentrality[2])

        l1_nbr_events_type = defaultdict(int)
        l2_nbr_events_type = defaultdict(int)
        l3_nbr_events_type = defaultdict(int)

        l1l2_nbr_events_type = defaultdict(int)
        l2l1_nbr_events_type = defaultdict(int)
        l1l3_nbr_events_type = defaultdict(int)
        l3l1_nbr_events_type = defaultdict(int)
        l2l3_nbr_events_type = defaultdict(int)
        l3l2_nbr_events_type = defaultdict(int)

        mov_event_count = defaultdict(lambda: defaultdict(int))

        for tid in imn['traj_location_from_to']:
            for evnt in event_traj2evntlist[tid]:
                if imn['traj_location_from_to'][tid][1] == 0:
                    l1_nbr_events_type[evnt['event_type']] += 1
                elif imn['traj_location_from_to'][tid][1] == 1:
                    l2_nbr_events_type[evnt['event_type']] += 1
                elif imn['traj_location_from_to'][tid][1] == 2:
                    l3_nbr_events_type[evnt['event_type']] += 1

                if imn['traj_location_from_to'][tid][0] == 0 and imn['traj_location_from_to'][tid][1] == 1:
                    l1l2_nbr_events_type[evnt['event_type']] += 1
                elif imn['traj_location_from_to'][tid][0] == 1 and imn['traj_location_from_to'][tid][1] == 0:
                    l2l1_nbr_events_type[evnt['event_type']] += 1
                elif imn['traj_location_from_to'][tid][0] == 0 and imn['traj_location_from_to'][tid][1] == 2:
                    l1l3_nbr_events_type[evnt['event_type']] += 1
                elif imn['traj_location_from_to'][tid][0] == 2 and imn['traj_location_from_to'][tid][1] == 0:
                    l3l1_nbr_events_type[evnt['event_type']] += 1
                elif imn['traj_location_from_to'][tid][0] == 1 and imn['traj_location_from_to'][tid][1] == 2:
                    l2l3_nbr_events_type[evnt['event_type']] += 1
                elif imn['traj_location_from_to'][tid][0] == 2 and imn['traj_location_from_to'][tid][1] == 1:
                    l3l2_nbr_events_type[evnt['event_type']] += 1

                lft = imn['traj_location_from_to'][tid][1]
                mov_event_count[evnt['event_type']][lft] += 1

        for event_type in ['Q', 'B', 'A', 'C', 'stop', 'start']:
            if event_type in l1_nbr_events_type:
                l1_events[event_type].append(l1_nbr_events_type[event_type])
            else:
                l1_events[event_type].append(0)
            if event_type in l2_nbr_events_type:
                l2_events[event_type].append(l2_nbr_events_type[event_type])
            else:
                l2_events[event_type].append(0)
            if event_type in l3_nbr_events_type:
                l3_events[event_type].append(l3_nbr_events_type[event_type])
            else:
                l3_events[event_type].append(0)

            if event_type in l1l2_nbr_events_type:
                l1l2_events[event_type].append(l1l2_nbr_events_type[event_type])
            else:
                l1l2_events[event_type].append(0)
            if event_type in l2l1_nbr_events_type:
                l2l1_events[event_type].append(l2l1_nbr_events_type[event_type])
            else:
                l2l1_events[event_type].append(0)

            if event_type in l1l3_nbr_events_type:
                l1l3_events[event_type].append(l1l3_nbr_events_type[event_type])
            else:
                l1l3_events[event_type].append(0)
            if event_type in l3l1_nbr_events_type:
                l3l1_events[event_type].append(l3l1_nbr_events_type[event_type])
            else:
                l3l1_events[event_type].append(0)

            if event_type in l2l3_nbr_events_type:
                l2l3_events[event_type].append(l2l3_nbr_events_type[event_type])
            else:
                l2l3_events[event_type].append(0)
            if event_type in l3l1_nbr_events_type:
                l3l2_events[event_type].append(l3l2_nbr_events_type[event_type])
            else:
                l3l2_events[event_type].append(0)

            if event_type in mov_event_count:
                vals = list(mov_event_count[event_type].values())
                mov_event_entropy[event_type].append(calculate_entropy(vals, classes=len(vals)))
            else:
                mov_event_entropy[event_type].append(0.0)

        l1l2_count.append(imn['location_nextlocs'][l1].get(l2, 0))
        l2l1_count.append(imn['location_nextlocs'][l2].get(l1, 0))
        l1l3_count.append(imn['location_nextlocs'][l1].get(l3, 0))
        if '2' in imn['location_nextlocs']:
            l3l1_count.append(imn['location_nextlocs'][l3].get(l1, 0))
            l2l3_count.append(imn['location_nextlocs'][l2].get(l3, 0))
            l3l2_count.append(imn['location_nextlocs'][l3].get(l2, 0))
        else:
            l3l1_count.append(0)
            l2l3_count.append(0)
            l3l2_count.append(0)
        edge_betweenness = nx.edge_betweenness(graph)
        l1l2_betweenness.append(edge_betweenness.get((0, 1), 0))
        l2l1_betweenness.append(edge_betweenness.get((1, 0), 0))
        l1l3_betweenness.append(edge_betweenness.get((0, 2), 0))
        l3l1_betweenness.append(edge_betweenness.get((2, 0), 0))
        l2l3_betweenness.append(edge_betweenness.get((1, 2), 0))
        l3l2_betweenness.append(edge_betweenness.get((2, 1), 0))

    imn_temporal_features = get_imn_temporal_features(imn_list)

    features = {
        'nbr_locations': np.mean(nbr_locations),
        'nbr_movements': np.mean(nbr_movements),
        'nbr_reg_locations': np.mean(nbr_reg_locations),
        'nbr_reg_movements': np.mean(nbr_reg_movements),
        'radius_of_gyration': np.mean(radius_of_gyration),
        'regular_radius_of_gyration': np.mean(regular_radius_of_gyration),
        'entropy': np.mean(entropy),
        'rentropy': np.mean(rentropy),
        'avg_mov_length': np.mean(avg_mov_length),
        'std_mov_length': np.mean(std_mov_length),
        'avg_mov_duration': np.mean(avg_mov_duration),
        'std_mov_duration': np.mean(std_mov_duration),
        # 'avg_reg_mov_length': np.mean(avg_reg_mov_length),
        # 'std_reg_mov_length': np.mean(std_reg_mov_length),
        'avg_reg_mov_duration': np.mean(avg_reg_mov_duration),
        'std_reg_mov_duration': np.mean(std_reg_mov_duration),

        'density': np.mean(density),
        'triangles': np.mean(triangles),
        'clustering_coefficient': np.mean(clustering_coefficient),
        'avg_degree': np.mean(degree),
        'avg_indegree': np.mean(indegree),
        'avg_outdegree': np.mean(outdegree),
        'diameter': np.mean(diameter),
        'eccentricity': np.mean(eccentricity),
        'assortativity': np.mean(assortativity),

        'l1_count': np.mean(l1_count),
        'l2_count': np.mean(l2_count),
        'l3_count': np.mean(l3_count),
        'l1_indegree': np.mean(l1_indegree),
        'l2_indegree': np.mean(l2_indegree),
        'l3_indegree': np.mean(l3_indegree),
        'l1_outdegree': np.mean(l1_outdegree),
        'l2_outdegree': np.mean(l2_outdegree),
        'l3_outdegree': np.mean(l3_outdegree),
        'l1_dcentrality': np.mean(l1_dcentrality),
        'l2_dcentrality': np.mean(l2_dcentrality),
        'l3_dcentrality': np.mean(l3_dcentrality),
        'l1_bcentrality': np.mean(l1_bcentrality),
        'l2_bcentrality': np.mean(l2_bcentrality),
        'l3_bcentrality': np.mean(l3_bcentrality),

        'l1l2_count': np.mean(l1l2_count),
        'l2l1_count': np.mean(l2l1_count),
        'l1l3_count': np.mean(l1l3_count),
        'l3l1_count': np.mean(l3l1_count),
        'l2l3_count': np.mean(l2l3_count),
        'l3l2_count': np.mean(l3l2_count),
        'l1l2_betweenness': np.mean(l1l2_betweenness),
        'l2l1_betweenness': np.mean(l2l1_betweenness),
        'l1l3_betweenness': np.mean(l1l3_betweenness),
        'l3l1_betweenness': np.mean(l3l1_betweenness),
        'l2l3_betweenness': np.mean(l2l3_betweenness),
        'l3l2_betweenness': np.mean(l3l2_betweenness),
    }

    features.update(imn_temporal_features)

    for event_type in ['Q', 'B', 'A', 'C', 'stop', 'start']:
        features['l1_%s' % event_type] = np.mean(l1_events[event_type])
        features['l2_%s' % event_type] = np.mean(l2_events[event_type])
        features['l3_%s' % event_type] = np.mean(l3_events[event_type])
        features['l1l2_%s' % event_type] = np.mean(l1l2_events[event_type])
        features['l2l1_%s' % event_type] = np.mean(l2l1_events[event_type])
        features['l1l3_%s' % event_type] = np.mean(l1l3_events[event_type])
        features['l3l1_%s' % event_type] = np.mean(l3l1_events[event_type])
        features['l2l3_%s' % event_type] = np.mean(l2l3_events[event_type])
        features['l3l2_%s' % event_type] = np.mean(l3l2_events[event_type])
        features['mov_entropy_%s' % event_type] = np.mean(mov_event_entropy[event_type])

    for k, v in features.items():
        if np.isnan(v):
            features[k] = -1

    return features


def path_in_tree(tree, path, max_depth=16):
    idx = 0
    node = tree
    while True:
        if node['is_leaf'] or idx == len(path) or (max_depth is not None and node['depth'] >= max_depth):
            break
        code = path[idx]
        # print(idx, code)
        if code not in node:
            # print('path not found')
            break
        node = node[code]
        idx += 1

    return path[:idx]


def get_collective_features(trajectories, imn_list, quadtree, quadtree_features):

    features_path_count = defaultdict(int)
    for traj in trajectories.values():
        paths_of_this_traj = set()
        for i, point in enumerate(traj.object):
            lon, lat, _ = point
            path = lon_lat_to_quadtree_path(lon, lat, depth=16)
            if path not in paths_of_this_traj:
                features_path_count[path] += 1
                paths_of_this_traj.add(path)

    is_regular_path = {path: False for path in features_path_count}
    for m0m1, imn in imn_list.items():
        if imn is None:
            continue
        if len(imn['regular_locations']) > 0:
            for lid in imn['regular_locations']:
                if isinstance(list(imn['location_prototype'].keys())[0], str):
                    lid = str(lid)
                lon, lat = imn['location_prototype'][lid]
                path = lon_lat_to_quadtree_path(lon, lat, depth=16)
                is_regular_path[path] = True

            for mid, movement_traj in imn['movement_traj'].items():
                lft = movement_traj[0]
                if lft[0] in imn['regular_locations'] and lft[1] in imn['regular_locations']:
                    if isinstance(imn['movement_prototype'][mid], dict):
                        traj_object = imn['movement_prototype'][mid]['object']
                    else:
                        traj_object = imn['movement_prototype'][mid].object
                    for i, point in enumerate(traj_object):
                        lon, lat, _ = point
                        path = lon_lat_to_quadtree_path(lon, lat, depth=16)
                        is_regular_path[path] = True

    aggregated_quadtree_features_reg = dict()
    aggregated_quadtree_features_irrreg = dict()
    aggregated_quadtree_features_reg_count = dict()
    aggregated_quadtree_features_irrreg_count = dict()
    for path in features_path_count:
        if path not in quadtree_features:
            continue
        apath = path_in_tree(quadtree, path, max_depth=16)
        if is_regular_path[path]:
            if apath not in aggregated_quadtree_features_reg:
                aggregated_quadtree_features_reg[apath] = quadtree_features[path]
                aggregated_quadtree_features_reg_count[apath] = features_path_count[path]
            else:
                for k, v in quadtree_features[path].items():
                    aggregated_quadtree_features_reg[apath][k] += v
                aggregated_quadtree_features_reg_count[apath] += features_path_count[path]
        else:
            if apath not in aggregated_quadtree_features_irrreg:
                aggregated_quadtree_features_irrreg[apath] = quadtree_features[path]
                aggregated_quadtree_features_irrreg_count[apath] = features_path_count[path]
            else:
                for k, v in quadtree_features[path].items():
                    aggregated_quadtree_features_irrreg[apath][k] += v
                aggregated_quadtree_features_irrreg_count[apath] += features_path_count[path]

    aggregated_quadtree_features_reg_comb = dict()
    aggregated_quadtree_features_irrreg_comb = dict()
    for aqf, aqfc in zip([aggregated_quadtree_features_reg, aggregated_quadtree_features_irrreg],
                   [aggregated_quadtree_features_reg_comb, aggregated_quadtree_features_irrreg_comb]):
        for path in aqf:
            aqfc[path] = {
                'nbr_traj_start': aqf[path]['nbr_traj_start'],
                'nbr_traj_stop': aqf[path]['nbr_traj_stop'],
                'nbr_traj_move': aqf[path]['nbr_traj_move'],
                'avg_traj_speed': aqf[path]['traj_speed_count'] / aqf[path]['traj_speed_count']
                if aqf[path]['traj_speed_count'] > 0 else 0,
                'nbr_evnt_A': aqf[path]['nbr_evnt_A'],
                'nbr_evnt_B': aqf[path]['nbr_evnt_B'],
                'nbr_evnt_C': aqf[path]['nbr_evnt_C'],
                'nbr_evnt_Q': aqf[path]['nbr_evnt_Q'],
                'nbr_evnt_start': aqf[path]['nbr_evnt_start'],
                'nbr_evnt_stop': aqf[path]['nbr_evnt_stop'],
                'avg_speed_A': aqf[path]['speed_A_sum'] / aqf[path]['nbr_evnt_A'] if aqf[path]['nbr_evnt_A'] > 0 else 0,
                'avg_max_acc_A': aqf[path]['max_acc_A_sum'] / aqf[path]['nbr_evnt_A'] if aqf[path]['nbr_evnt_A'] > 0 else 0,
                'avg_avg_acc_A': aqf[path]['avg_acc_A_sum'] / aqf[path]['nbr_evnt_A'] if aqf[path]['nbr_evnt_A'] > 0 else 0,
                'avg_speed_B': aqf[path]['speed_B_sum'] / aqf[path]['nbr_evnt_B'] if aqf[path]['nbr_evnt_B'] > 0 else 0,
                'avg_max_acc_B': aqf[path]['max_acc_B_sum'] / aqf[path]['nbr_evnt_B'] if aqf[path]['nbr_evnt_B'] > 0 else 0,
                'avg_avg_acc_B': aqf[path]['avg_acc_B_sum'] / aqf[path]['nbr_evnt_B'] if aqf[path]['nbr_evnt_B'] > 0 else 0,
                'avg_speed_C': aqf[path]['speed_C_sum'] / aqf[path]['nbr_evnt_C'] if aqf[path]['nbr_evnt_C'] > 0 else 0,
                'avg_max_acc_C': aqf[path]['max_acc_C_sum'] / aqf[path]['nbr_evnt_C'] if aqf[path]['nbr_evnt_C'] > 0 else 0,
                'avg_avg_acc_C': aqf[path]['avg_acc_C_sum'] / aqf[path]['nbr_evnt_C'] if aqf[path]['nbr_evnt_C'] > 0 else 0,
                'avg_speed_Q': aqf[path]['speed_Q_sum'] / aqf[path]['nbr_evnt_Q'] if aqf[path]['nbr_evnt_Q'] > 0 else 0,
                'avg_max_acc_Q': aqf[path]['max_acc_Q_sum'] / aqf[path]['nbr_evnt_Q'] if aqf[path]['nbr_evnt_Q'] > 0 else 0,
                'avg_avg_acc_Q': aqf[path]['avg_acc_Q_sum'] / aqf[path]['nbr_evnt_Q'] if aqf[path]['nbr_evnt_Q'] > 0 else 0,
                'nbr_crash': aqf[path]['nbr_crash'],
            }
    aggregated_quadtree_features_reg = aggregated_quadtree_features_reg_comb
    aggregated_quadtree_features_irrreg = aggregated_quadtree_features_irrreg_comb

    features = defaultdict(float)
    total_reg = np.sum(list(aggregated_quadtree_features_reg_count.values()))
    for path, values in aggregated_quadtree_features_reg.items():
        count = aggregated_quadtree_features_reg_count[path]
        for k, v in values.items():
            if total_reg > 0:
                features['reg_%s' % k] += v * count / total_reg
            else:
                features['reg_%s' % k] += 0
    total_occ = np.sum(list(aggregated_quadtree_features_irrreg_count.values()))
    for path, values in aggregated_quadtree_features_irrreg.items():
        count = aggregated_quadtree_features_irrreg_count[path]
        for k, v in values.items():
            if total_occ > 0:
                features['occ_%s' % k] += v * count / total_occ
            else:
                features['occ_%s' % k] += 0

    return features


def extract_features_data(uid, data, quadtree, quadtree_features):

    features = dict()
    for index, values in data.items():
        trajectories = values['trajectories']
        events = values['events']
        imn_list = values['imns']
        event_traj2evntlist = defaultdict(list)
        for eid, evnt in events.items():
            event_traj2evntlist[evnt['tid']].append(evnt)

        if len(trajectories) == 0:
            continue

        traj_features = get_trjectory_features(trajectories)
        evnt_features = get_events_features(events)
        imn_features = get_imn_features(imn_list, event_traj2evntlist)
        collective_features = get_collective_features(trajectories, imn_list, quadtree, quadtree_features[index])

        features[index] = {
            'uid': uid,
            'crash': values['crash'],
        }

        features[index].update(traj_features)
        features[index].update(evnt_features)
        features[index].update(imn_features)
        features[index].update(collective_features)

    return features


def extract_features(uid, tr_data, ts_data, quadtree, tr_quadtree_features, ts_quadtree_features):
    # print('train')
    training = extract_features_data(uid, tr_data, quadtree, tr_quadtree_features)
    # print('test')
    test = extract_features_data(uid, ts_data, quadtree, ts_quadtree_features)
    return training, test


def store_features(filename, store_obj):
    json_str = '%s\n' % json.dumps(store_obj, cls=NumpyEncoder)
    json_bytes = json_str.encode('utf-8')
    # print(json_str)
    with gzip.GzipFile(filename, 'a') as fout:
        fout.write(json_bytes)


# def main():
#     area = sys.argv[1]
#     country = 'uk' if area == 'london' else 'italy'
#
#     path = './'
#     path_dataset = path + 'dataset/'
#     path_imn = path + 'imn/'
#     path_crash = path + 'crash/'
#
#     crash_users_filename = path_dataset + '%s_users_list.csv' % area
#     nocrash_users_filename = path_dataset + '%s_nocrash_users_list.csv' % area
#
#     crash_users_list = sorted(pd.read_csv(crash_users_filename).values[:, 0].tolist())
#     nocrash_users_list = sorted(pd.read_csv(nocrash_users_filename).values[:, 0].tolist())
#
#
#
# if __name__ == "__main__":
#     main()
