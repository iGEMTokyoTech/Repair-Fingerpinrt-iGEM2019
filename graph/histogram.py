import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logging import getLogger, DEBUG, StreamHandler

from edit_img import image_processing
from optimise.width_getter import get_radius_list

logger = getLogger(__name__)
logger.setLevel(DEBUG)
StreamHandler().setLevel(DEBUG)
logger.addHandler(StreamHandler())


def young_pattern_hist(img, hist_file='./histogram.png', debug=False):
    # young patternのクラスからヒストグラムを作成する関数
    logger.debug("make histogram of pattern's width")
    img = img.astype(np.uint8)
    if img.max() == 1:
        img = img * 255
    img_thin = image_processing.thinning(img, debug=debug)
    width_hist(img_thin, file=hist_file)


def width_hist(img, file="./histogram.png", debug=False):
    #ヒストグラムを作成する関数
    height, width = img.shape
    width_list = get_radius_list(img, [int(width / 2), int(height / 2)], debug=debug)
    sns.set()
    fig, ax = plt.subplots()
    sns.distplot(
        width_list, bins=np.arange(min(width_list)-1, max(width_list)) + 0.5, color='#123456', label='data',
        kde=False,
        rug=False,
    )
    ax.set(ylabel='the number of circles', xlabel='radius of circle')
    plt.savefig(file)
    plt.show()




def main():
    # sample code
    print("histogram")


if __name__ == "__main__":
    main()