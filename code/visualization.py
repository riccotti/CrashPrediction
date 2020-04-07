import bezier
import colormap
import webbrowser
import numpy as np
import pandas as pd
import colorlover as cl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import folium
from folium.plugins import HeatMap

c_lat, c_lon = None, None


def visualize_points(filename, points, tiles='stamentoner', zoom_start=11):
    global c_lat, c_lon
    c_lat, c_lon = np.mean(points, axis=0)
    m = folium.Map(location=[c_lon, c_lat], tiles=tiles, zoom_start=zoom_start)
    HeatMap(np.c_[points[:, 1], points[:, 0]],
            min_opacity=0.5, max_zoom=15, max_val=1.0, radius=15, blur=15).add_to(m)
    title_html = """<div style="position: fixed; 
        top: 20px; left: 50px; width: 800px; height: 90px; 
        z-index:9999; font-size:40px; font-weight:bold; color: #3175b7">Raw GPS Points</div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(filename)
    webbrowser.open('file://%s' % filename)


def visualize_trajectories(filename, trajectories, tiles='stamentoner', zoom_start=11):
    global c_lat, c_lon
    if c_lat is None:
        points = np.array([[traj.start_point()[1], traj.start_point()[0]] for traj in trajectories.values()])
        c_lat, c_lon = np.mean(points, axis=0)
    m = folium.Map(location=[c_lon, c_lat], tiles=tiles, zoom_start=zoom_start)

    for tid, traj in trajectories.items():
        folium.PolyLine([[p[1], p[0]] for p in traj.object]).add_to(m)
    title_html = """<div style="position: fixed; 
            top: 20px; left: 50px; width: 800px; height: 90px; 
            z-index:9999; font-size:40px; font-weight:bold; color: #3175b7">Reconstructed Trajectories</div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(filename)
    webbrowser.open('file://%s' % filename)


def visualize_stops(filename, trajectories, tiles='stamentoner', zoom_start=11, radius=50):
    global c_lat, c_lon
    if c_lat is None:
        points = np.array([[traj.start_point()[1], traj.start_point()[0]] for traj in trajectories.values()])
        c_lat, c_lon = np.mean(points, axis=0)

    lat_list = list()
    lon_list = list()
    for tid, traj in trajectories.items():
        lat_list.append(traj.object[-1][1])
        lon_list.append(traj.object[-1][0])

    m = folium.Map(location=[c_lon, c_lat], tiles=tiles, zoom_start=zoom_start)
    for i in range(0, len(trajectories)):
        folium.Circle(location=(lat_list[i], lon_list[i]), radius=radius, fill=True, fill_opacity=0.8).add_to(m)
    title_html = """<div style="position: fixed; 
                top: 20px; left: 50px; width: 800px; height: 90px; 
                z-index:9999; font-size:40px; font-weight:bold; color: #3175b7">Stop Points</div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(filename)
    webbrowser.open('file://%s' % filename)


def visualize_locations(filename, location_prototype, location_features, tiles='stamentoner', zoom_start=11,
                        q=np.array([0.0, 0.25, 0.50, 0.75, 1.0])):
    global c_lat, c_lon
    if c_lat is None:
        points = np.array([[p[1], p[0]] for p in location_prototype.values()])
        c_lat, c_lon = np.mean(points, axis=0)

    lat_list = list()
    lon_list = list()
    sup_list = list()
    for lid, p in location_prototype.items():
        lat_list.append(p[1])
        lon_list.append(p[0])
        sup_list.append(np.sqrt(location_features[lid]['loc_support'] * 10000))

    sup_colors = pd.qcut(sup_list, q=q, duplicates='drop')
    colors = list(cl.scales['9']['seq']['Blues'])[9 - len(sup_colors.categories):]
    sup_colors = pd.qcut(sup_list, q=q, labels=colors, duplicates='drop')

    m = folium.Map(location=[c_lon, c_lat], tiles=tiles, zoom_start=zoom_start)
    for i in range(0, len(location_prototype)):
        folium.Circle(
            location=(lat_list[i], lon_list[i]),
            radius=sup_list[i],
            color=sup_colors[i],
            fill=True,
            fill_color=sup_colors[i],
            fill_opacity=0.8
        ).add_to(m)
    title_html = """<div style="position: fixed; 
                top: 20px; left: 50px; width: 800px; height: 90px; 
                z-index:9999; font-size:40px; font-weight:bold; color: #3175b7">Detected Locations</div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(filename)
    webbrowser.open('file://%s' % filename)


def get_bearing(p1, p2):
    '''
    Returns compass bearing from p1 to p2

    Parameters
    p1 : namedtuple with lat lon
    p2 : namedtuple with lat lon

    Return
    compass bearing of type float

    Notes
    Based on https://gist.github.com/jeromer/2005586
    '''

    long_diff = np.radians(p2[0] - p1[0])

    lat1 = np.radians(p1[1])
    lat2 = np.radians(p2[1])

    x = np.sin(long_diff) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2)
         - (np.sin(lat1) * np.cos(lat2)
            * np.cos(long_diff)))
    bearing = np.degrees(np.arctan2(x, y))

    # adjusting for compass bearing
    if bearing < 0:
        return bearing + 360
    return bearing


def visualize_imn(filename, location_nextlocs, location_prototype, location_features,
                  tiles='stamentoner', zoom_start=11, q=np.array([0.0, 0.25, 0.50, 0.75, 1.0])):
    global c_lat, c_lon
    if c_lat is None:
        points = np.array([[p[1], p[0]] for p in location_prototype.values()])
        c_lat, c_lon = np.mean(points, axis=0)

    fmov = list()
    weight = list()
    for lid1 in location_nextlocs:
        for lid2 in location_nextlocs[lid1]:
            s = location_prototype[lid1]
            e = location_prototype[lid2]
            gap = 0.05 * abs(e[1] - s[1]) / 0.05
            nodes = np.asfortranarray([
                [s[1], (s[1] + e[1]) / 2 + np.random.choice([gap, -gap]), e[1]],
                [s[0], (s[0] + e[0]) / 2 + np.random.choice([gap, -gap]), e[0]],
            ])
            curve = bezier.Curve(nodes, degree=2)
            val = curve.evaluate_multi(np.linspace(0.0, 1.0, 10))
            x_val = val[0]
            y_val = val[1]
            mov = list()
            for xv, yv in zip(x_val, y_val):
                mov.append([xv, yv])

            fmov.append(mov)
            weight.append(np.log(location_nextlocs[lid1][lid2] * 10))

    sup_colors = pd.qcut(weight, q=q, duplicates='drop')
    colors = list(cl.scales['9']['seq']['Greens'])[9 - len(sup_colors.categories):]
    sup_colors = pd.qcut(weight, q=q, labels=colors, duplicates='drop')

    m = folium.Map(location=[c_lon, c_lat], tiles=tiles, zoom_start=zoom_start)
    for i, fm in enumerate(fmov):
        folium.PolyLine(fm, color=sup_colors[i], weight=weight[i], opacity=0.8).add_to(m)
        s, e = fm[0], fm[-2]
        rotation = get_bearing(s, e) - 90
        folium.RegularPolygonMarker(location=e, color=sup_colors[i], fill=True, fill_color=sup_colors[i],
                                    fill_opacity=0.8, number_of_sides=3, radius=6, rotation=rotation).add_to(m)

    lat_list = list()
    lon_list = list()
    sup_list = list()
    for lid, p in location_prototype.items():
        lat_list.append(p[1])
        lon_list.append(p[0])
        sup_list.append(np.sqrt(location_features[lid]['loc_support'] * 10000))

    sup_colors = pd.qcut(sup_list, q=q, duplicates='drop')
    colors = list(cl.scales['9']['seq']['Blues'])[9 - len(sup_colors.categories):]
    sup_colors = pd.qcut(sup_list, q=q, labels=colors, duplicates='drop')

    for i in range(0, len(lon_list)):
        folium.Circle(location=(lat_list[i], lon_list[i]), radius=sup_list[i], color=sup_colors[i], fill=True,
                      fill_color=sup_colors[i], fill_opacity=0.8).add_to(m)

    title_html = """<div style="position: fixed; 
                top: 20px; left: 50px; width: 800px; height: 90px; 
                z-index:9999; font-size:40px; font-weight:bold; color: #3175b7">Individual Mobility Network</div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(filename)
    webbrowser.open('file://%s' % filename)


def cl2hex(c):
    r, g, b = c
    r, g, b = int(r), int(g), int(b)
    return colormap.rgb2hex(r, g, b)


def visualize_features(filename, user_features, df_train, features):
    features_map = dict()
    for ft, flist in features.items():
        for f in flist:
            features_map[f] = ft

    vals = list()
    names = list()
    mean_values = df_train.mean().to_dict()
    max_values = df_train.max().to_dict()
    min_values = df_train.min().to_dict()

    for f, v in user_features.items():
        if f in ['uid', 'crash']:
            continue
        if np.isnan(v) or np.isinf(v) or v == -1 or max_values[f] == min_values[f] or np.isinf(
                min_values[f]) or np.isinf(max_values[f]):
            vals.append(0)
        else:
            v1 = (v - min_values[f]) / (max_values[f] - min_values[f])
            v2 = (mean_values[f] - min_values[f]) / (max_values[f] - min_values[f])
            d = v1 - v2
            vals.append(d)
        names.append('%s-%s' % (features_map[f], f))

    gap = (max(vals) - min(vals)) / 6
    bins = np.arange(min(vals), max(vals), gap)
    color_scale = list(cl.to_numeric(cl.scales[str(len(bins) - 1)]['div']['RdYlGn']))
    color_scale = [cl2hex(c) for c in color_scale]
    colors = pd.cut(vals, bins=bins, labels=color_scale)
    colors = [c if not isinstance(c, float) else color_scale[3] for c in colors]

    fetures_per_plot = 20
    fig = plt.figure(figsize=(50, 40))
    fontsize = 13

    pid = 0
    for cid in range(0, 3):
        for rid in range(0, 7):
            plt.subplot(3, 7, pid + 1)
            ifrom = pid * fetures_per_plot
            ito = pid * fetures_per_plot + fetures_per_plot
            svals = vals[ifrom:ito]
            snames = names[ifrom:ito]
            scolors = colors[ifrom:ito]

            x = np.arange(len(svals))
            plt.barh(x, svals, color=scolors)
            for i, v, n in zip(x, svals, snames):
                if v < 0:
                    plt.text(0.01, i, n, fontsize=fontsize)
                elif v > 0:
                    plt.text(-0.01, i, n, horizontalalignment='right', fontsize=fontsize)
                elif v == 0:
                    plt.text(0.01, i, n, horizontalalignment='center', fontsize=fontsize)
            plt.axvline(0, color='k')
            plt.axis('off')
            plt.xlim(min(vals), max(vals))
            pid += 1

    st = fig.suptitle('Final Features', fontsize=fontsize*10)
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    browser = webbrowser.get('chrome')
    browser.open('file://%s' % filename)


def visualize_crash_risk(filename, uid, area, period, crash_proba, path):
    odo_idx = int((1.0-crash_proba) * 6)
    img = mpimg.imread(path + 'fig/odometer/odometer_%s.png' % odo_idx)
    plt.imshow(img)
    plt.title('User %s - %s - %s - Crash Risk: %.2f' % (
        uid, area.capitalize(), period.capitalize(), crash_proba), fontsize=16)
    plt.axis('off')
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    browser = webbrowser.get('chrome')
    browser.open('file://%s' % filename)
