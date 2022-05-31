# Compute the contrast between two images
# Image 1 has high bone-dura contrast
# Image 2 has low bone-dura contrast

import numpy as np
import cv2

MEDIA_PATH1 = '../res/monochromatic_illumination/2022-05-02 experiment1/600nm.png'
MEDIA_PATH2 = '../res/monochromatic_illumination/2022-05-02 experiment1/700nm.png'

'''-------------------------------Luma--------------------------------------------------------'''


def bgr2luma(frame):
    # R*0.222 + G*0.717 + B*0.061
    luma = frame[:, :, 0] * 0.061 + frame[:, :, 1] * 0.717 + frame[:, :, 2] * 0.222
    return luma


if __name__ == "__main__":

    # load image
    frame1 = cv2.imread(MEDIA_PATH1)
    frame2 = cv2.imread(MEDIA_PATH2)
    # Convert BGR image into luminance images
    frame1_luma = bgr2luma(frame1)
    frame2_luma = bgr2luma(frame2)

    # # Image1 / Image 2
    # contrast_frame = np.uint8(np.divide(frame1_luma, frame2_luma))

    # Image1 * Image 2
    contrast_frame = np.uint8(np.multiply(frame1_luma, frame2_luma))


    # # (Image1-Image2) / (Image1+Image2)
    # contrast_frame = np.uint8(np.divide(frame1_luma - frame2_luma, frame1_luma + frame2_luma))

    # Spread into values between 0-255
    # contrast_frame = contrast_frame * (255 / contrast_frame.max())
    contrast_frame = contrast_frame / np.linalg.norm(contrast_frame) * 255

    cv2.imshow('frame1_luma', cv2.resize(np.uint8(frame1_luma), None, fx=0.8, fy=0.8))
    cv2.imshow('frame2_luma', cv2.resize(np.uint8(frame2_luma), None, fx=0.8, fy=0.8))
    cv2.imshow('contrast_frame', cv2.resize(contrast_frame, None, fx=0.8, fy=0.8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
