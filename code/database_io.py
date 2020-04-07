import psycopg2
from collections import defaultdict

from trajectory import *
from util import default_to_regular

__author__ = 'Riccardo Guidotti'


def get_connection():
    properties = {
        'dbname': 'dbname',
        'user': 'user',
        'host': 'host',
        'port': 'port',
        'password': 'password',
    }
    db_params = 'dbname=\'' + properties['dbname'] + '\' ' \
                'user=\'' + properties['user'] + '\' ' \
                'host=\'' + properties['host'] + '\' ' \
                'port=\'' + properties['port'] + '\' ' \
                'password=\'' + properties['password'] + '\''

    con = psycopg2.connect(db_params)

    return con


def extract_users_list(input_table, cur):
    query = """SELECT DISTINCT(uid) AS uid FROM %s""" % input_table
    cur.execute(query)
    rows = cur.fetchall()

    users = list()
    for r in rows:
        uid = str(r[0])
        users.append(uid)

    return sorted(users)


def load_individual_mobility_history(cur, uid, input_table, min_length=0, min_duration=0):

    query = """SELECT tid, ST_AsGeoJSON(traj) AS object, uid, length, duration, start_time, end_time
        FROM %s
        WHERE uid = '%s'""" % (input_table, uid)

    cur.execute(query)
    rows = cur.fetchall()
    trajectories = dict()

    for r in rows:
        # trajectories[str(r[0])] = Trajectory(id=str(r[0]), object=json.loads(r[1])['coordinates'], vehicle=uid)
        traj = [[p[0], p[1], p[2] * 1000] for p in json.loads(r[1])['coordinates']]
        traj = Trajectory(id=str(r[0]), object=traj, vehicle=uid, length=float(r[3]), duration=float(r[4]),
                          start_time=r[5], end_time=r[6])

        if traj.length() > min_length and traj.duration() > min_duration:
            trajectories[str(r[0])] = traj
        # if len(trajectories) >= 100:
        #     break

    imh = {'uid': uid, 'trajectories': trajectories}

    return imh


def load_individual_event_history(cur, uid, table_name):
    query = """SELECT uid, tid, eid, event_type, speed, ABS(max_acceleration) AS max_acc, 
            ABS(avg_acceleration) AS avg_acc, event_angle AS angle, location_type, duration, date, lat, lon 
            FROM %s
            WHERE uid = '%s'""" % (table_name, uid)

    cur.execute(query)
    rows = cur.fetchall()
    events = defaultdict(list)

    for r in rows:
        event = {
            'uid': str(r[0]),
            'tid': str(r[1]),
            'eid': str(r[2]),
            'event_type': str(r[3]),
            'speed': int(r[4]),
            'max_acc': int(r[5]),
            'avg_acc': int(r[6]),
            'angle': int(r[7]),
            'location_type': str(r[8]),
            'duration': int(r[9]),
            'date': r[10],
            'lat': float(r[11]),
            'lon': float(r[12]),
        }
        events[str(r[2])].append(event)

    return default_to_regular(events)


def load_mobility_histories(cur, users, input_table):

    users_str = '(\'%s\')' % ('\',\''.join(users))

    query = """SELECT id, ST_AsGeoJSON(object) AS object, uid
        FROM %s
        WHERE uid IN %s
        ORDER BY id""" % (input_table, users_str)

    cur.execute(query)
    rows = cur.fetchall()
    trajectories = dict()

    uid = None

    mobility_histories = dict()

    for r in rows:
        id = str(r[0])
        cur_uid = id.split('_')[0]

        if uid is None:
            uid = cur_uid

        if uid != cur_uid:
            trajectories[str(r[0])] = Trajectory(id=str(r[0]), object=json.loads(r[1])['coordinates'], vehicle=uid)
            mobility_histories[uid] = {'uid': uid, 'trajectories': trajectories}
            trajectories = dict()
            uid = cur_uid

        trajectories[str(r[0])] = Trajectory(id=str(r[0]), object=json.loads(r[1])['coordinates'], vehicle=uid)

    mobility_histories[uid] = {'uid': uid, 'trajectories': trajectories}

    return mobility_histories
