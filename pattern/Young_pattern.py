import numpy as np
from scipy import signal
import cv2
from pattern.dots import make_mask
import matplotlib.pyplot as plt
import logging
from datetime import datetime


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.StreamHandler().setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class Young_Pattern:
    width = 300  # 格子の横幅
    height = 300  # 格子の縦幅
    generation = 0

    def __init__(self, r1, r2, w1, w2, init_alive_prob=0.5, status=False):
        """
        コンストラクタ
        :param r1: radius of inner circle
        :param r2: radius of outer circle
        :param w1: parameter of inner circle
        :param w2: parameter of outer circle
        :param init_alive_prob: percentage of the number of initial black
        """
        logger.debug("__init__")
        self.mask = make_mask(r1, r2, w1, w2, side=21)
        self.r1 = r1
        self.r2 = r2
        self.w1 = w1
        self.w2 = w2
        self.generation = 0
        self.state = self.init_state(self.width, self.height, init_alive_prob)

    def init_state(self, width=width, height=height, init_alive_prob=0.5):
        """
        初期化
        :param height: 格子の横幅
        :param width: 格子の縦幅
        :param init_alive_prob: percentage of the number of initial black
        :return: ndarray  state
        """
        self.width = width
        self.height = height
        N = self.width * self.height
        v = np.array(np.random.rand(N) + init_alive_prob, dtype=int)
        self.state = v.reshape(self.height, self.width)
        self.generation = 0
        return self.state

    def next_generation(self):
        """
        次の世代にstateを更新する
        :return: ndarray  state
        """
        N = signal.correlate2d(self.state, self.mask, mode="same", boundary="wrap")
        self.state = N > 0
        self.generation += 1
        return self.state

    def far_generation(self, generation):
        """
        generation世代後にstateを変更する
        :param generation: 世代数
        :return:
        """
        for i in range(generation):
            self.next_generation()
        return self.state

    def load_text(self, filename):
        """
        初期値としてndarrayを他のファイルから取ってくる
        :param filename:
        """
        logger.debug("load_text")
        if cv2.os.path.exists(filename):
            self.state = np.loadtxt(filename)
            self.width = self.state.shape[1]
            self.height = self.state.shape[0]
            self.generation = 0
        else:
            print("file is not existed")
        return self.state

    def load_ndarray(self, ndarray):
        """
        初期値としてndarrayを入れる
        :param ndarray: 初期値
        :return: state
        """
        if ndarray is None:
            print("ndarray is empty")
        else:
            self.state = ndarray.astype(np.float64)
            self.width = self.state.shape[1]
            self.height = self.state.shape[0]
            self.generation = 0
        return self.state

    def to_image(self, w=800, h=800):
        """
        imageをcv2で出力する
        :param w: resize to display by cv2
        :param h: resize to display by cv2
        :return: cv2
        """
        logger.debug("display by cv2")
        logger.debug("generation=%d", self.generation)
        img = np.array(self.state, dtype=np.uint8) * 255
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img

    def save_text(self, filename):
        """
        ndarrayをファイルに保存する
        :param filename:
        """
        logger.debug("saved to %s", filename)
        logger.debug("generation=%d", self.generation)
        np.savetxt(filename, self.state, "%d")

    def save_img(self, img_path="./image-"+str(int(datetime.now().timestamp()*(10**3)))+".png"):
        """
        matplotlib 出力と保存
        """
        logger.debug("file saved to %s", img_path)
        logger.debug("generation =", self.generation)
        plt.figure()
        plt.imshow(self.state, cmap='pink', vmin=0, vmax=1)
        plt.savefig(img_path)
        plt.show()

    def show(self):
        logger.debug("display by pyplot")
        logger.debug("generation=%d", self.generation)
        plt.figure()
        plt.imshow(self.state, cmap='pink', vmin=0, vmax=1)
        plt.show()

    def show_cv2(self):
        winname = "Young Pattern"
        # ret = 0
        wait = 50

        while True:
            img = self.to_image()
            cv2.imshow(winname, img)
            ret = cv2.waitKey(wait)
            self.next_generation()
            # prop_val = cv2.getWindowProperty(winname, cv2.WND_PROP_ASPECT_RATIO)
            if ret == ord('r'):
                self.init_state(init_alive_prob=0.08)
            if ret == ord('s'):
                wait = min(wait * 2, 1000)
            if ret == ord('f'):
                wait = max(wait // 2, 10)
            if ret == ord('q') or ret == 27:
                break
            if not is_visible(winname):
                break
            if ret == ord('w'):
                self.save_text("../data/save.txt")
            if ret == ord('l'):
                self.load_text("../data/save.txt")
        cv2.waitKey(1)  # macの都合
        cv2.destroyAllWindows()
        return 0


BackendError = type('BackendError', (Exception,), {})


def is_visible(winname):
    # cv2の閉じるエラーを解決したい（未解決）
    try:
        ret = cv2.getWindowProperty(
            winname, cv2.WND_PROP_VISIBLE
        )

        if ret == -1:
            raise BackendError('Use Qt as backend to check whether window is visible or not.')

        return bool(ret)

    except cv2.error:
        return False


def main():
    # sample code
    YP = Young_Pattern(3, 6, 1.0, -0.3, 0.08)
    YP.far_generation(10)
    YP.show()


if __name__ == "__main__":
    main()
