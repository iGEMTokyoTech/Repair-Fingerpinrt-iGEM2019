import numpy as np
from scipy import signal
import collections
from matplotlib import pyplot as plt
from datetime import datetime

from edit_img import image_processing
from graph.histgram_maker import make_hist


# 円のマスクを作成
# radius: 整数(半径)
# 返り値: 2次元配列(半径の2倍+1の大きさ)
def make_mask(radius):
    result = np.zeros((radius * 2 + 1, radius * 2 + 1))
    for y in range(0, radius * 2):
        for x in range(0, radius * 2):
            if np.sqrt((x - radius) ** 2 + (y - radius) ** 2) <= radius:
                result[y][x] = 1
    return result


# 特定の座標の特定の半径の円は隣の線にかぶっているか
# thined_img: 2次元配列(細線化済み)(0と何かの二値になっていること)
# coordinate: [x,y]
# radius: 整数(半径)
# 返り値: Boolean(Trueで隣の線にかぶってる、Falseでかぶっていない)
def check_overlap(thined_img, coordinate, radius):
    height, width = thined_img.shape
    if coordinate[1] - radius < 0 or coordinate[1] + radius > height or coordinate[0] - radius < 0 or coordinate[
        0] + radius > width:
        return True
    mask = make_mask(radius)
    clipping = thined_img[coordinate[1] - radius: coordinate[1] + radius + 1,
               coordinate[0] - radius: coordinate[0] + radius + 1]
    clipping = clipping.astype(np.uint16)
    result = signal.correlate2d(clipping, mask, mode="same", boundary="fill")
    return result[radius][radius] > thined_img[coordinate[1]][coordinate[0]] * (radius * 2 + 1)


# 特定の座標を通る2本の直線について、指紋との交点における指紋の線同士の距離のリストを返す
# thined_img: 2次元配列(細線化済み)
# coordinate: [x,y]
# 返り値: [距離, 距離, 距離, ...]
def get_radius_list(thined_img, coordinate, *, debug=False):
    if debug: print("start: nakano.width_getter.get_radius_list")
    height, width = thined_img.shape
    result = []
    if debug: plt.imshow(thined_img, cmap="gray")
    if debug: ax = plt.gca()
    # y軸と平行な直線(行)について
    last_intersection = 0
    for x in range(width):
        if thined_img[coordinate[1]][x] > 0:
            if last_intersection != x - 1:
                continue_flag = True
                radius = 0
                while continue_flag and radius < width:
                    radius += 1
                    if check_overlap(thined_img, [x, coordinate[1]], radius):
                        continue_flag = False
                        result.append(radius)
                        if debug: ax.add_artist(
                            plt.Circle((x, coordinate[1]), radius, facecolor='None', edgecolor='red', linewidth=0.3,
                                       alpha=0.7))
            last_intersection = x
    # x軸と平行な直線(列)について
    last_intersection = 0
    for y in range(height):
        if thined_img[y][coordinate[0]] > 0:
            if last_intersection != y - 1:
                continue_flag = True
                radius = 0
                while continue_flag and radius < height:
                    radius += 1
                    if check_overlap(thined_img, [coordinate[0], y], radius):
                        continue_flag = False
                        result.append(radius)
                        if debug: ax.add_artist(
                            plt.Circle((coordinate[0], y), radius, facecolor='None', edgecolor='yellow', linewidth=0.3,
                                       alpha=0.7))
            last_intersection = y
    # if debug: plt.title("get_radius")
    if debug: plt.savefig("./image-radius" + str(int(datetime.now().timestamp() * (10 ** 3))) + ".png", dpi=1000)
    if debug: print(str(np.min(result)) + " ~ " + str(np.max(result)))
    if debug: print("end  : nakano.width_getter.get_radius_list")
    return result


# get_radius_listをscan_line_pairs回行って、幅の最頻値を求める
# thined_img: 2次元配列(細線化済み)
# scan_line_pairs: 整数(この値の2倍の線について幅を確認する)
# 返り値: 数値(幅)
def get_width(thined_img, scan_line_pairs=1, *, debug=False):
    height, width = thined_img.shape
    radius_list = []
    for i in range(scan_line_pairs):
        radius_list += get_radius_list(thined_img, [int(width / (scan_line_pairs + 1) * (i + 1)),
                                                    int(height / (scan_line_pairs + 1) * (i + 1))], debug=debug)
    if debug:
        # histogram
        make_hist(radius_list)
    return collections.Counter(radius_list).most_common()[0][0]


if __name__ == '__main__':
    img = image_processing.load_image_grayscale("../data/img/fingerprint.png")
    img = image_processing.blur(img)
    img = image_processing.threshold(img)
    img = image_processing.bitwise_not(img)
    img = image_processing.morphological_transformations(img)
    img = image_processing.thinning(img)
    image_processing.image_save(img, title="sample")
    width = get_width(img, debug=True)
    print("最頻値(幅): " + str(width))
