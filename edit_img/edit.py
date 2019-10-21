import numpy as np
from logging import getLogger, DEBUG, StreamHandler

from edit_img import image_processing

logger = getLogger(__name__)
logger.setLevel(DEBUG)
StreamHandler().setLevel(DEBUG)
logger.addHandler(StreamHandler())


def noise(img, noise_param=10000, debug=False):
    logger.debug("generating noise")
    row, col = img.shape
    logger.debug("noise" + str(noise_param/(row*col)*100) + "%")

    # 白
    pts_x = np.random.randint(0, col - 1, noise_param)
    pts_y = np.random.randint(0, row - 1, noise_param)
    img[(pts_y, pts_x)] = 255  # y,xの順番になることに注意

    # 黒
    pts_x = np.random.randint(0, col - 1, noise_param)
    pts_y = np.random.randint(0, row - 1, noise_param)
    img[(pts_y, pts_x)] = 0
    if debug:
        image_processing.image_show(img)
    return img


def trim(img, left=0., right=1., top=0., bottom=1., debug=False):
    """
    トリミングする
    :param debug:
    :param img: 0~1
    :param left: 0~1
    :param right: 0~1
    :param top: 0~1
    :param bottom: 0~1
    :return:
    """
    if debug:
        print("start: trim")
    height = img.shape[0]
    width = img.shape[1]

    if (left > right) or (top > bottom):
        print("left & top should be lower than right & bottom")
        exit(1)
    elif (0 > left) and (left > 1):
        print("parameter left should be 0 to 1")
        exit(1)
    elif (0 > right) and (right > 1):
        print("parameter right should be 0 to 1")
        exit(1)
    elif (0 > top) and (top > 1):
        print("parameter top should be 0 to 1")
        exit(1)
    elif (0 > bottom) and (bottom > 1):
        print("parameter bottom should be 0 to 1")
        exit(1)
    else:
        result = img[int(height * top):int(height * bottom), int(width * left):int(width * right)]
        if debug:
            image_processing.image_show(img, "trim - before")
            image_processing.image_show(result, "trim - after")
            print("end  : trim")
        return result


def make_blank(img, size=50, num=5):
    """
    画像を欠損させる
    :param size: 正方形のサイズ
    :param num: 正方形の数
    :return: ndarray
    """
    logger.debug("{0} blank".format(num))
    for i in range(num):
        a = np.random.randint(0, img.shape[0] - size - 1)
        b = np.random.randint(0, img.shape[0] - size - 1)
        img[a:a + size, b:b + size] = np.ones((size, size)) * 255
    return img



