import logging

import cv2
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from edit_img import image_processing
from pattern.Young_pattern import Young_Pattern

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.StreamHandler().setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class Young_Finger(Young_Pattern):
    r_fn = 0.01
    r_rd = 0.5

    def __init__(self, r1, r2, w1, w2, init_alive_prob=0.5):
        super().__init__(r1, r2, w1, w2, init_alive_prob)
        self.img_finger = self.state
        self.__noise = np.zeros((self.width, self.height))
        self.noise()

    def noise(self):
        N = self.width * self.height
        # v = np.array(np.random.rand(N), dtype=float) * 2 - 1
        v = np.array(np.random.rand(N), dtype=float)
        self.__noise = v.reshape(self.height, self.width)
        return self.__noise

    def load_text(self, filename):
        self.img_finger = super().load_text(filename)

    def load_ndarray(self, ndarray):
        self.img_finger = super().load_ndarray(ndarray)

    def next_generation(self):
        """
        次の世代にstateを更新する
        :return: ndarray  state
        """
        N = signal.correlate2d(self.state, self.mask, mode="same", boundary="wrap")
        N = N * (1 - self.r_fn) + self.img_finger * self.r_fn + self.noise() * self.r_rd
        self.state = N > 0
        self.noise()
        self.generation += 1
        return self.state

    def show_ini_end(self, save_filename, gen=30):
        fig, ax = plt.subplots(ncols=2,
                               sharex="col", sharey="all",
                               facecolor="lightgray")
        fig.suptitle(
            'r1={0:.2g} r2={1:.2g} w1={2:.2g} w2={3:.2g} r_fn={4:.2g} r_rd={5:.2g} gen={6:.2g}'.format(self.r1, self.r2,
                                                                                                       self.w1, self.w2,
                                                                                                       self.r_fn,
                                                                                                       self.r_rd,
                                                                                                       gen), fontsize=9)
        ax[0].imshow(self.init_state(), cmap='pink')
        ax[0].set_title("initial state ", fontsize=7)
        ax[1].imshow(self.far_generation(gen), cmap='pink')
        ax[1].set_title("generation={0:.2g} ".format(gen), fontsize=7)
        plt.savefig("../data/compare_finger_" + save_filename + ".png")
        plt.show()


def main():
    # sample code
    filename = "../data/img/fingerprint.png"
    img = np.empty((2, 2))
    if cv2.os.path.exists(filename):
        img = image_processing.load_image_grayscale(filename)
    else:
        print("file does not exist")
        exit(1)
    YP = Young_Finger(3, 6, 16.0, -5.0)
    YP.load_ndarray(img)
    YP.far_generation(10)
    YP.show()


if __name__ == "__main__":
    main()
    # exit()
