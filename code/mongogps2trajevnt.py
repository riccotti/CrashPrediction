import sys
import time
import argparse
import pymongo
import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta

from shapely.geometry import Point, LineString


import database_io


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
        evnt_values = [lat, lon, evnt_ts, event_type, speed, max_acceleration, avg_acceleration, event_angle, heading,
                       gps_quality, pv, location_type, duration, status]
        return evnt_user, evnt_ts, evnt_values
    except StopIteration:
        return None, None, None
    except Exception:
        return None, None, None


def next_crash(crash_data):
    try:
        crash_row = next(crash_data)
        flag_validity = crash_row['FLAG_VALIDITY']
        crash_user = crash_row['T&K_VOUCHER_ID']
        crash_ts = crash_row['TIMESTAMP_LOCAL']
        lon = crash_row['LONGITUDE'] / 1000000.0
        lat = crash_row['LATITUDE'] / 1000000.0
        max_acceleration = crash_row['MAX_ACCELERATION']
        speed = crash_row['SPEED']
        heading = crash_row['HEADING']
        gps_quality = crash_row['GPS_QUALITY']
        if flag_validity != 'Y':
            return -1, crash_ts, None
        crash_values = [lat, lon, crash_ts, speed, max_acceleration, heading, gps_quality]
        return crash_user, crash_ts, crash_values
    except StopIteration:
        return None, None, None
    except Exception:
        return None, None, None


def gps2trajevntcrash(traj_data, evnt_data, crash_data, traj_table, evnt_table, crash_table, con,
                      drop_table=True, max_speed=0.07, space_treshold=0.05, time_treshold=1200, min_time_gap=300):

    cur = con.cursor()

    traj_datalist = list()
    evnt_datalist = list()
    crash_datalist = list()

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

    if drop_table:
        cur.execute("DROP TABLE IF EXISTS " + traj_table)
        cur.execute("DROP TABLE IF EXISTS " + evnt_table)
        cur.execute("DROP TABLE IF EXISTS " + crash_table)
        con.commit()

    traj = list()
    length = None

    evnt_id = 0
    crash_id = 0
    tot_events = 1
    tot_crash = 0
    first_iteration = True
    evnt_user, evnt_ts, evnt_values = next_evnt(evnt_data)
    crash_user, crash_ts, crash_values = next_crash(crash_data)

    for row in traj_data:

        tot_points += 1
        user = row['T&K_VOUCHER_ID']
        next_ts = row['TIMESTAMP_LOCAL']
        next_status = row['STATUS']
        next_gps_quality = row['GPS_QUALITY']
        next_lon = row['LONGITUDE'] / 1000000.0
        next_lat = row['LATITUDE'] / 1000000.0

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

                    while crash_user is not None and crash_ts < start_time:
                        if crash_user != -1:
                            tot_crash += 1
                        crash_user, crash_ts, crash_values = next_crash(crash_data)
                        if crash_user is None:
                            break

                    if crash_user is not None:
                        while crash_ts <= end_time:
                            if crash_user != -1:
                                crash_datarecord = [user, tid, crash_id] + crash_values
                                crash_datalist.append(crash_datarecord)
                                crash_id += 1
                                tot_crash += 1
                            crash_user, crash_ts, crash_values = next_crash(crash_data)
                            if crash_user is None:
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
                    if ts0 > traj[0][0] + timedelta(seconds=min_time_gap):  # traj[0][0] contains last timestamp of last traj
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

        # scorro i crash finche' non ne raggiungo uno maggiore ugule di quello corrente
        while crash_user is not None and crash_ts < start_time:
            tot_crash += 1
            crash_user, crash_ts, crash_values = next_crash(crash_data)
            if crash_user is None:
                break

        # aggiungo i crash alla traiettoria finche' non ne raggiungo uno maggiore
        if crash_user is not None:
            while crash_ts <= end_time:
                if crash_user != -1:
                    crash_datarecord = [user, tid, crash_id] + crash_values
                    crash_datalist.append(crash_datarecord)
                    crash_id += 1
                    tot_crash += 1
                crash_user, crash_ts, crash_values = next_crash(crash_data)
                if crash_user is None:
                    break

    count_traj = count_traj + len(traj_datalist)
    print(datetime.now(),
          'Inserting %d trajs / %d points / %d tot points - %d events / %d tot events - %d tot crash - %d omitted fake' % (
              count_traj, count_points, tot_points, evnt_id, tot_events, tot_crash, count_omitted))
    save(traj_datalist, evnt_datalist, crash_datalist, traj_table, evnt_table, crash_table, con)


def save(traj_datalist, evnt_datalist, crash_datalist, traj_table, evnt_table, crash_table, con):
    cur = con.cursor()
    create_traj_query = """CREATE TABLE IF NOT EXISTS %s (uid text, tid numeric, traj geometry, length numeric, 
                      duration numeric, start_time timestamp, end_time timestamp)""" % traj_table
    create_evnt_query = """CREATE TABLE IF NOT EXISTS %s (uid text, tid numeric, eid numeric, lat numeric, lon numeric, 
                          date timestamp, event_type text, speed numeric, max_acceleration numeric, 
                          avg_acceleration numeric, event_angle numeric, heading numeric, gps_quality numeric, 
                          pv numeric, location_type numeric, duration numeric, status numeric)""" % evnt_table
    create_crash_query = """CREATE TABLE IF NOT EXISTS %s (uid text, tid numeric, cid numeric, lat numeric, lon numeric,
                                date timestamp, speed numeric, max_acceleration numeric, heading numeric)""" % crash_table

    cur.execute(create_traj_query)
    cur.execute(create_evnt_query)
    cur.execute(create_crash_query)
    con.commit()

    for row in traj_datalist:
        t = list(row[2].coords)
        ts = row[3]
        t_str_points = ''
        for i in range(len(t)):
            t_str_points += "ST_MakePoint(%s,%s,%s)," % (t[i][0], t[i][1], time.mktime(ts[i].timetuple()) / 1000.0)

        t_str = "ST_SetSRID(ST_MakeLine(ARRAY[%s]),4326)" % t_str_points[:-1]  # -1 removes the last ,

        insert_query = """INSERT INTO %s VALUES (%s, %s, %s, %s, %s, TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS'), 
                          TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS'))""" % (
            traj_table, row[0], row[1], t_str, row[4], row[5], row[8], row[9])

        try:
            cur.execute(insert_query)
        except Exception:
            print("Exception " + insert_query)
            con.commit()
            cur.close()
            cur = con.cursor()

    for row in evnt_datalist:
        insert_query = """INSERT INTO %s VALUES (%s, %s, %s, %s, %s, TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS'), 
        '%s', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""" % (
            evnt_table, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10],
            row[11], row[12], row[13], row[14], row[15], row[16])
        try:
            cur.execute(insert_query)
        except Exception:
            print("Exception " + insert_query)
            con.commit()
            cur.close()
            cur = con.cursor()

    for row in crash_datalist:
        insert_query = """INSERT INTO %s VALUES (%s, %s, %s, %s, %s, TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS'), 
        %s, %s, %s)""" % (crash_table, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8])
        try:
            cur.execute(insert_query)
        except Exception:
            print("Exception " + insert_query)
            con.commit()
            cur.close()
            cur = con.cursor()

    con.commit()
    cur.close()


def main():

    country = 'uk'

    traj_table = 'tak.%s_traj' % country
    evnt_table = 'tak.%s_evnt' % country
    crash_table = 'tak.%s_crash' % country

    drop_table = True

    max_speed = 0.07
    space_treshold = 0.05
    time_treshold = 1200   # split traj if time gap between consecutive points > max_time_gap

    users = pd.read_csv('../../dataset/users_%s.csv' % country, header=None)[0].values.tolist()
    client = pymongo.MongoClient('mongodb://username@ipaddress:port/')
    db = client['dataset2']
    con = database_io.get_connection()
    print(datetime.now(), 'Building trajectories')
    for i, uid in enumerate(users):
        traj_data = db.POSITIONS.find({'T&K_VOUCHER_ID': uid}).sort('TIMESTAMP_LOCAL', pymongo.ASCENDING)
        evnt_data = db.EVENTS.find({'T&K_VOUCHER_ID': uid}).sort('TIMESTAMP_LOCAL', pymongo.ASCENDING)
        crash_data = db.CRASH.find({'T&K_VOUCHER_ID': uid}).sort('TIMESTAMP_LOCAL', pymongo.ASCENDING)

        print(datetime.now(), 'Processing user %s, %s of %s (%.2f)' % (uid, i, len(users), 100.0*i/len(users)))
        gps2trajevntcrash(traj_data, evnt_data, crash_data, traj_table, evnt_table, crash_table, con,
                          drop_table=drop_table, max_speed=max_speed, space_treshold=space_treshold,
                          time_treshold=time_treshold)
        drop_table = False

    print(datetime.now(), 'Process ended.')


if __name__ == "__main__":
    main()
