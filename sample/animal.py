import cv2
import numpy as np

from edit_img import image_processing
from optimise.Optimal_pattern import Optimal_pattern
import sys

print("animal")
print(sys.path)

for i in range(1, 5):
    filename = "../data/img_animal/animal0{0}.jpg".format(i)
    print(filename)
    img = np.empty((2, 2))
    if cv2.os.path.exists(filename):
        img = image_processing.load_image_grayscale(filename)
    else:
        print("file does not exist")
        exit(1)
    result_img = Optimal_pattern(img)
    image_processing.image_show(result_img)
