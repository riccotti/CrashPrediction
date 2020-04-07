import math

# in degree, ~1m at equator
min_resolution = 0.00001

##in degree ~100m, used by previous
# min_resolution = 0.0001/4

max_y = 180 / min_resolution
split_num = int(math.log2(max_y))
real_min_resolution = 180. / 2 ** split_num

# num_digit = 48
num_digit = 2 * split_num


def interleave_64bit(x, y):
    """
        Modified according to interleave_32bit
    """
    B = [0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, \
         0x0000FFFF0000FFFF, 0x00000000FFFFFFFF]
    S = [1, 2, 4, 8, 16, 32]

    x = (x | (x << S[5])) & B[5]
    x = (x | (x << S[4])) & B[4]
    x = (x | (x << S[3])) & B[3]
    x = (x | (x << S[2])) & B[2]
    x = (x | (x << S[1])) & B[1]
    x = (x | (x << S[0])) & B[0]

    y = (y | (y << S[5])) & B[5]
    y = (y | (y << S[4])) & B[4]
    y = (y | (y << S[3])) & B[3]
    y = (y | (y << S[2])) & B[2]
    y = (y | (y << S[1])) & B[1]
    y = (y | (y << S[0])) & B[0]

    z = x | (y << 1)
    return z


def lng_lat_to_z(lng, lat):
    """
        Denote west-north as 0,0
        0----> x
        0
        |
        |
        y
        (-180, 90)->(0,0)
    """
    trans_y = int((-lat + 90.) / real_min_resolution)
    trans_x = int((lng + 180.) / 2 / real_min_resolution)
    return interleave_64bit(trans_x, trans_y)


def z_to_4_based(z):
    branch = {'00': '0', '01': '1', '10': '2', '11': '3'}
    bin_z = bin(z)[2:]
    bin_z = '0' * (num_digit - len(bin_z)) + bin_z
    r = ''
    for i in range(0, num_digit, 2):
        r += branch[bin_z[i:i + 2]]
    return r


def lon_lat_to_quadtree_path(lon, lat, depth=None):
    """
        The main function to call
    """
    val = z_to_4_based(lng_lat_to_z(lon, lat))
    if depth is None:
        return val
    return val[:depth]


def main():
    lat = 41.89193
    lon = 12.51133

    trans_y = int((-lat + 90.) / real_min_resolution)
    trans_x = int((lon + 180.) / 2 / real_min_resolution)

    print(real_min_resolution, num_digit)
    print(trans_x, trans_y)
    z = lng_lat_to_z(lon, lat)
    print(z)
    print(z_to_4_based(z))

    print(lon_lat_to_quadtree_path(lat, lon))
    print(lon_lat_to_quadtree_path(lat, lon, 16))


if __name__ == "__main__":
    main()
