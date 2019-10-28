import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


# 画像を表示
# img: n次元配列
# title: 文字列(タイトル)
# 返り値: (なし)
def image_show(img, title=""):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


# 画像を保存
# img: n次元配列
# title: 文字列(タイトル)
# img_path: 文字列(iGEM2019ディレクトリからの相対パス)(デフォルト値はUnixtimeにミリ秒を加えた名前)
# 返り値: (なし)
def image_save(img, title="", img_path="./image-" + str(int(datetime.now().timestamp() * (10 ** 3))) + ".png",
               dpi=1000):
    plt.imshow(img)
    plt.title(title)
    plt.savefig(img_path, dpi=dpi)


# 画像をグレイスケールで読み込む
# img_path: 文字列(iGEM2019ディレクトリからの相対パス)
# 返り値: 2次元配列(グレースケール、0~255)
def load_image_grayscale(img_path, *, debug=False):
    if debug:
        print("start: nakano.image_processing.load_image")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if debug:
        image_show(img, "loaded_image")
        print("end  : nakano.image_processing.load_image")
    return img


# ノイズ処理(ぼやかす)
# img: n次元配列
# 返り値: n次元配列(入力の配列の次元による)
def blur(img, *, debug=False):
    if debug:
        print("start: nakano.image_processing.blur")
    result = cv2.medianBlur(img, 5)
    result = cv2.GaussianBlur(result, (5, 5), 0)
    if debug:
        image_show(img, "blur - before")
        image_show(result, "blur - after")
        print("end  : nakano.image_processing.blur")
    return result


# 二値化(適応的しきい値処理)
# img: n次元配列
# 返り値: 2次元配列(0か255)
def threshold(img, *, debug=False):
    if debug:
        print("start: nakano.image_processing.threshold")
    result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    if debug:
        image_show(img, "threshold - before")
        image_show(result, "threshold - after")
        print("end  : nakano.image_processing.threshold")
    return result


# ネガポジ反転
# img: n次元配列
# 返り値: n次元配列(入力の配列の次元による)
def bitwise_not(img, *, debug=False):
    if debug:
        print("start: nakano.image_processing.bitwise_not")
    result = cv2.bitwise_not(img)
    if debug:
        image_show(img, "bitwise_not - before")
        image_show(result, "bitwise_not - after")
        print("end  : nakano.image_processing.bitwise_not")
    return result


# ノイズ処理(モルフォロジー変換)(二値化済み)
# img: 2次元配列
# 返り値: 2次元配列
def morphological_transformations(img, *, kernel_size=3, debug=False):
    if debug:
        print("start: nakano.image_processing.morphological_transformations")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    result = cv2.erode(result, kernel, iterations=1)
    if debug:
        image_show(img, "morphological_transformations - before")
        image_show(result, "morphological_transformations - after")
        print("end  : nakano.image_processing.morphological_transformations")
    return result


# 細線化
# img: 2次元配列(二値化済み)
# 返り値: 2次元配列(0か255)
def thinning(img, *, debug=False):
    if debug:
        print("start: nakano.image_processing.thinning")
    height, width = img.shape
    result = np.zeros((height, width), np.uint8)
    result = cv2.ximgproc.thinning(img, result, cv2.ximgproc.THINNING_GUOHALL)
    if debug:
        image_show(img, "thinning - before")
        image_show(result, "thinning - after")
        print("end  : nakano.image_processing.thinning")
    return result


# 特徴点抽出
# img: 2次元配列(細線化済み)
# 返り値: 2次元配列(特徴点マップ)(0か1か10)(1は端点、10は三路分岐点)
def generate_features_map(img, *, debug=False):
    if debug:
        print("start: nakano.image_processing.generate_features_map")
    height, width = img.shape
    # 画面の端を0にする
    for y in range(0, height - 1):
        for x in range(0, width - 1):
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                img[y][x] = 0
    # 各ピクセルについて周囲のピクセルの値を調べる
    img = img.astype(np.uint16)
    neighbour_point_map = np.zeros((height, width), np.uint16)
    for y in range(1, height - 2):
        for x in range(1, width - 2):
            if img[y][x] == 255:
                neighbour_point = img[y - 1][x - 1] + img[y][x - 1] + img[y + 1][x - 1] + img[y - 1][x] + img[y + 1][
                    x] + img[y - 1][x + 1] + img[y][x + 1] + img[y + 1][x + 1]
                neighbour_point_map[y][x] = neighbour_point
    # 重複を排除し、特徴点マップを作成
    if debug: plt.imshow(img, cmap="gray")
    result = np.zeros((height, width), np.uint8)
    for y in range(1, height - 2):
        for x in range(1, width - 2):
            if neighbour_point_map[y][x] == 255:
                result[y][x] = 1
                if debug: plt.scatter(x, y, marker='.', linewidths=0.3, facecolor='None', edgecolors='yellow',
                                      alpha='0.5')
            if neighbour_point_map[y][x] == 765:
                if not (neighbour_point_map[y - 1][x - 1] >= 765 or neighbour_point_map[y][x - 1] >= 765 or
                        neighbour_point_map[y + 1][x - 1] >= 765 or neighbour_point_map[y + 1][x] >= 765):
                    result[y][x] = 10
                    if debug: plt.scatter(x, y, marker='.', linewidths=0.3, facecolor='None', edgecolors='red',
                                          alpha='0.5')

    if debug:
        plt.title("features")
        plt.savefig("./image-" + str(int(datetime.now().timestamp() * (10 ** 3))) + ".png", dpi=1000)
        print("end  : nakano.image_processing.generate_features_map")
    return result


# 特徴点の座標を取得
# features_map: 2次元配列(特徴点マップ)(0か1か10)(1は端点、10は三路分岐点)
# 返り値: 端点の座標の配列([[x,y],[x,y],...])、三路分岐点の座標の配列([[x,y],[x,y],...])
def list_features(features_map, *, debug=False):
    if debug:
        print("start: nakano.image_processing.list_features")
    height, width = features_map.shape
    endpoints = []
    trifurcations = []
    for y in range(0, height - 1):
        for x in range(0, width - 1):
            if features_map[y][x] == 1: endpoints.append([x, y])
            if features_map[y][x] == 10: trifurcations.append([x, y])
    if debug:
        print("end  : nakano.image_processing.list_features")
    return endpoints, trifurcations


# 特徴点の数を取得
# features_map: 2次元配列(特徴点マップ)(0か1か10)(1は端点、10は三路分岐点)
# 返り値: 端点の数、三路分岐点の数
def count_features(features_map, *, debug=False):
    endpoints, trifurcations = list_features(features_map)
    if debug:
        print("端点: " + str(len(endpoints)))
        print("三路分岐点: " + str(len(trifurcations)))
    return len(endpoints), len(trifurcations)


if __name__ == '__main__':
    img = load_image_grayscale("../data/img/fingerprint.png", debug=True)
    img = blur(img, debug=True)
    img = threshold(img, debug=True)
    img = bitwise_not(img, debug=True)
    img = morphological_transformations(img, debug=True)
    img = thinning(img, debug=True)
    features_map = generate_features_map(img, debug=True)
    count_features(features_map, debug=True)
