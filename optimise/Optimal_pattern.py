import cv2
import numpy as np
from logging import getLogger, DEBUG, StreamHandler

from edit_img import image_processing
from graph.histogram import young_pattern_hist
from optimise.width_getter import get_width
from pattern.Young_finger import Young_Finger

logger = getLogger(__name__)
logger.setLevel(DEBUG)
StreamHandler().setLevel(DEBUG)
logger.addHandler(StreamHandler())


def width_init_img(img, debug=False):
    logger.debug("width of initial img")
    img = image_processing.blur(img, debug=debug)
    img = image_processing.threshold(img, debug=debug)
    img = image_processing.bitwise_not(img, debug=debug)
    img = image_processing.morphological_transformations(img, debug=debug)
    img = image_processing.thinning(img, debug=debug)
    # image_processing.image_save(img, title="sample")
    if debug:
        image_processing.image_show(img, title="width_init_img")
    return get_width(img, debug=debug)


def Optimal_pattern(img,
                    young=Young_Finger(3, 6, 15, -5),
                    pattern_width=10,
                    generation=10,
                    debug=False):
    logger.info("Optimal_pattern start")
    img_width = width_init_img(img, debug=debug)
    logger.debug("画像の最頻値(幅): " + str(img_width))

    magnify = pattern_width / img_width
    resize_img = cv2.resize(img, dsize=None, fx=magnify, fy=magnify)
    logger.debug("reshaped to " + str(resize_img.shape))
    young.load_ndarray(resize_img)
    young.far_generation(generation)
    logger.debug("Optimal_pattern end")
    return young.state


# sample code
def main():
    filename = "../data/img/fingerprint.png"
    img = np.empty((2, 2))
    if cv2.os.path.exists(filename):
        img = image_processing.load_image_grayscale(filename)
    else:
        print("file does not exist")
        exit(1)
    result_img = Optimal_pattern(img)
    image_processing.image_show(result_img)
    young_pattern_hist(result_img)


if __name__ == "__main__":
    main()
    exit(0)
