import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
from edit_img import image_processing
from edit_img.edit import trim
from pattern.Young_finger import Young_Finger


def gif_maker(YP, generation=20, duration=300,
              img_path="./gif-" + str(int(datetime.now().timestamp() * (10 ** 3))) + ".gif"):
    """
    gif画像を作る
    :param generation:
    :param img_path:
    :param duration:
    :type YP: Young_Pattern
    """
    ims = []

    fig = plt.figure(figsize=(3, 3))
    plt.imshow(YP.state, cmap='gray')
    fig.canvas.draw()

    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    im = Image.fromarray(image_array)
    ims.append(im)
    plt.close()

    for i in range(generation):
        # 描画
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(YP.next_generation(), cmap='gray')
        fig.canvas.draw()

        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        im = Image.fromarray(image_array)
        ims.append(im)
        plt.close()
    ims[0].save(img_path, save_all=True, append_images=ims[1:], loop=1, duration=duration)
    print(img_path)


def main():
    filename = "../data/img/fingerprint.png"
    img = np.empty((2, 2))
    if cv2.os.path.exists(filename):
        img = image_processing.load_image_grayscale(filename)
    else:
        print("file does not exist")
        exit(1)
    trimed = trim(img, 0.05, 0.9, 0.1, 0.9, True)
    magnify = 10 / 8
    resize_img = cv2.resize(trimed, dsize=None, fx=magnify, fy=magnify)
    young = Young_Finger(3, 6, 15, -5)
    young.load_ndarray(resize_img)
    gif_maker(young, generation=5, duration=1000)


if __name__ == "__main__":
    main()
    exit(0)
