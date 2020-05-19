import os
import sys
import math
import imageio
import numpy as np
import pprint
import fiona
import progressbar

from PIL import Image
from skimage.transform import resize


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    xfrac, xtile = math.modf(x)
    yfrac, ytile = math.modf(y)
    return (int(xtile), int(ytile), int(xfrac * tile_res), int(yfrac * tile_res))


# load from existing png image
im = imageio.imread(sys.argv[1])
print('image shape', im.shape)

# load shape
# shape_path = 'sys.argv[2]'
# shape = fiona.open(shape_path)
# location = [[c[1], c[0]] for c in shape.next()['geometry']['coordinates'][0]]
#                                                     dim0=13346  dim1=13390
#                                                      lat i   lon j
#                                                      10215   10212
# upper-left  [48.793847257470574, 9.15551702148224] (137738, 90254, 216, 99)
# lower-right [48.757745326173854, 9.21031869331685] (137778, 90294, 191, 71)
#
# for origin in location[1:]:
#     index = deg2num(origin[0], origin[1], zoom_level)
#     print(origin, index)

ibase = [10215, 10212, 216, 99]
zoom_level = 18
tile_res = 256


class Tiles:
    def __init__(self, zoom_level):
        self.tiles = dict()
        self.zoom_level = zoom_level

    def __getitem__(self, key):
        if key not in self.tiles:
            self.tiles[key] = self.create_tile()
        return self.tiles[key]

    @staticmethod
    def ind2tile(i, j):
        xtile, xpixel = divmod(int(i / 13345 * ibase[0] + ibase[2]), tile_res)
        ytile, ypixel = divmod(int(j / 13389 * ibase[1] + ibase[3]), tile_res)
        return (137738 + xtile, 90254 + ytile, xpixel, ypixel)

    @staticmethod
    def create_tile(tile_res=256):
        return np.zeros((tile_res, tile_res, 3))

    def save(self):
        for index, tile in self.tiles.items():
            im = Image.fromarray(tile.astype(np.uint8))
            path = '{}/{}/{}.png'.format(self.zoom_level, index[0], index[1])
            ensure_dir(path)
            im.save(path)


leaftiles = Tiles(zoom_level)
for i in progressbar.progressbar(range(im.shape[0])):
    for j in range(im.shape[1]):
        xtile, ytile, xpixel, ypixel = Tiles.ind2tile(i, j)
        leaftiles[(xtile, ytile)][ypixel, xpixel] = im[i, j]

print('save {} tiles of zoom {}'.format(
    len(leaftiles.tiles), leaftiles.zoom_level))
leaftiles.save()


for z in range(zoom_level, 0, -1):
    parents = Tiles(z-1)
    while len(leaftiles.tiles) != 0:
        index = list(leaftiles.tiles.keys())[0]
        base_x = index[0] if index[0] % 2 == 0 else index[0] - 1
        base_y = index[1] if index[1] % 2 == 0 else index[1] - 1
        for x in range(base_x, base_x+2):
            for y in range(base_y, base_y+2):
                if (x, y) in leaftiles.tiles:
                    leaftile = leaftiles.tiles.pop((x, y))
                    parent_part = resize(
                        leaftile, (int(leaftile.shape[0]/2), int(leaftile.shape[0]/2)))
                    offset_x = 0 if x % 2 == 0 else int(tile_res / 2)
                    offset_y = 0 if y % 2 == 0 else int(tile_res / 2)
                    parents[(int(base_x/2), int(base_y/2))][offset_y:offset_y +
                                                            int(tile_res/2), offset_x:offset_x+int(tile_res/2)] = parent_part
    print('save {} tiles of zoom {}'.format(
        len(parents.tiles), parents.zoom_level))
    parents.save()
    leaftiles = parents
