# Compute the contrast between two images
# Image 1 has high bone-dura contrast
# Image 2 has low bone-dura contrast

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

MEDIA_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/luma/position3/700nm.png'
MEDIA_SAVE_NAME1 = '700nm_qualHist.png'
MEDIA_SAVE_NAME2 = '700nm_CLAHE.png'
MEDIA_SAVE_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/luma/position3/'


def bgr2luma(frame):
    # R*0.222 + G*0.717 + B*0.061
    luma = frame[:, :, 0] * 0.061 + frame[:, :, 1] * 0.717 + frame[:, :, 2] * 0.222
    return luma


if __name__ == "__main__":
    # load image
    image = cv2.imread(MEDIA_PATH)
    image_luma = bgr2luma(image)

    # # plot histogram
    # # plt.hist(image.flatten(), 256, [0, 256], color='r')
    # plt.hist(image_luma.flatten(), 256, [0, 256], color='b')
    # plt.xlim([0, 256])
    # plt.show()


    # Simple Histogram Equalization
    image_equ = cv2.equalizeHist(np.uint8(image_luma))
    cv2.imshow('equal Simple Histogram', image_equ)


    # CLAHE Contrast Limited Adaptive Histogram Equalization
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(np.uint8(image_luma))
    cv2.imshow('CLAHE', image_clahe)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.hist(image_luma.flatten(), 256, [0, 256], color='r', alpha=0.5)
    plt.hist(image_equ.flatten(), 256, [0, 256], color='g', alpha=0.5)
    plt.hist(image_clahe.flatten(), 256, [0, 256], color='b', alpha=0.5)
    plt.xlim([0, 256])
    plt.legend(('original', 'simple', 'CLAHE'))
    plt.title('Histogram comparison after different equalizations')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.show()

    # cv2.imwrite(os.path.join(MEDIA_SAVE_PATH, MEDIA_SAVE_NAME1), image_equ)
    # cv2.imwrite(os.path.join(MEDIA_SAVE_PATH, MEDIA_SAVE_NAME2), image_clahe)
