import os
import sys
import math
import numpy as np
import pprint
import fiona
import progressbar

from PIL import Image
from skimage.transform import resize

zoom_level = 19
tile_res = 256


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
im = Image.open(sys.argv[1])
newshape = (im.size[0]*2, im.size[1]*2)
im = im.resize(newshape, resample=Image.BILINEAR)
im = np.array(im)
print('image shape', im.shape)

# load shape
# [48.793847257470574, 9.15551702148224] (275477, 180508, 176, 199)
# [48.793847257470574, 9.210318693316855] (275557, 180508, 127, 199)
# [48.757745326173854, 9.210318693316855] (275557, 180588, 127, 143)
# [48.757745326173854, 9.15551702148224] (275477, 180588, 176, 143)
#                                                     dim0=13346  dim1=13390
#                                                      lat i   lon j
#                                                      10215   10212
# upper-left  [48.793847257470574, 9.15551702148224] (275477, 180508, 176, 199)
# lower-right [48.757745326173854, 9.21031869331685] (275557, 180588, 127, 143)

shape_path = sys.argv[2]
shape = fiona.open(shape_path)
location = [[c[1], c[0]] for c in shape.next()['geometry']['coordinates'][0]]

ul_index = deg2num(location[1][0], location[1][1], zoom_level)
lr_index = deg2num(location[3][0], location[3][1], zoom_level)

ibase = [
    (lr_index[0]-ul_index[0])*tile_res + (lr_index[2]-ul_index[2]),
    (lr_index[1]-ul_index[1])*tile_res + (lr_index[3]-ul_index[3]),
    ul_index[2],
    ul_index[3]
]


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
        xtile, xpixel = divmod(
            int(j / (im.shape[0]-1) * ibase[0] + ibase[2]), tile_res)
        ytile, ypixel = divmod(
            int(i / (im.shape[1]-1) * ibase[1] + ibase[3]), tile_res)
        return (ul_index[0] + xtile, ul_index[1] + ytile, xpixel, ypixel)

    @staticmethod
    def create_tile(tile_res=256):
        return np.zeros((tile_res, tile_res, 4))

    def save(self):
        for index, tile in self.tiles.items():
            im = Image.fromarray(tile.astype(np.uint8))
            path = 'frontend/tiles/{}/{}/{}.png'.format(
                self.zoom_level, index[0], index[1])
            ensure_dir(path)
            im.save(path)


leaftiles = Tiles(zoom_level)
for i in progressbar.progressbar(range(im.shape[0])):
    for j in range(im.shape[1]):
        xtile, ytile, xpixel, ypixel = Tiles.ind2tile(i, j)
        leaftiles[(xtile, ytile)][ypixel, xpixel, :im.shape[2]] = im[i, j]
        if im[i, j].tolist() != [0,0,0]:
            leaftiles[(xtile, ytile)][ypixel, xpixel, -1] = 255

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
