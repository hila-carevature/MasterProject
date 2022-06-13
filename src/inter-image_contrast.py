# Compute the contrast between two images
# Image 1 has high bone-dura contrast
# Image 2 has low bone-dura contrast

import numpy as np
import cv2
import os

# # Test 1
# MEDIA_PATH1 = '../res/monochromatic_illumination/2022-05-02 experiment1 - edited/600nm.png'
# MEDIA_PATH2 = '../res/monochromatic_illumination/2022-05-02 experiment1 - edited/700nm.png'

# Test 2
MEDIA_PATH1 = '../res/monochromatic_illumination/2022-06-01 Experiment 2/position3/430nm.png'
MEDIA_PATH2 = '../res/monochromatic_illumination/2022-06-01 Experiment 2/position3/700nm.png'
MEDIA_SAVE_NAME = '430nm_700nm_clahe.png'
MEDIA_SAVE_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/image_division_uint8_before_normal/position3/'
# MEDIA_SAVE_LUMA_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/luma/position1/'
# MEDIA_SAVE_LUMA_NAME = 'white.png'

'''-------------------------------Luma--------------------------------------------------------'''


def bgr2luma(frame):
    # R*0.222 + G*0.717 + B*0.061
    luma = frame[:, :, 0] * 0.061 + frame[:, :, 1] * 0.717 + frame[:, :, 2] * 0.222
    return luma


def equal_clahe(frame):
    # CLAHE Contrast Limited Adaptive Histogram Equalization
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_clahe = clahe.apply(np.uint8(frame))
    return frame_clahe


if __name__ == "__main__":

    # load image
    frame1 = cv2.imread(MEDIA_PATH1)
    frame2 = cv2.imread(MEDIA_PATH2)
    # Convert BGR image into luminance images
    frame1_luma = bgr2luma(frame1)
    frame2_luma = bgr2luma(frame2)

    frame1_clahe = equal_clahe(frame1_luma)
    frame2_clahe = equal_clahe(frame2_luma)
    frame1_luma = frame1_clahe
    frame2_luma = frame2_clahe

    # division: Image1 (high contrast) / Image 2 (low contrast)
    # Supress/hide the warning
    np.seterr(invalid='ignore')
    contrast_frame = np.uint8(np.divide(frame1_luma, frame2_luma))
    # contrast_frame = cv2.divide(frame1_luma, frame2_luma)

    # # subtraction: Image1 (high contrast) - Image 2 (low contrast)
    # # contrast_frame = cv2.subtract(frame1_luma, frame2_luma)
    # contrast_frame = cv2.absdiff(frame1_luma, frame2_luma)

    # # Multiplication: Image1 (high contrast) * Image 2 (high contrast)
    # contrast_frame = np.float32(np.multiply(frame1_luma, frame2_luma))


    # # (Image1-Image2) / (Image1+Image2)
    # contrast_frame = np.uint8(np.divide(frame1_luma - frame2_luma, frame1_luma + frame2_luma))

    # Spread into values between 0-255
    # contrast_frame = contrast_frame * (255 / contrast_frame.max())
    contrast_frame_normal = cv2.normalize(contrast_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # contrast_frame_normal = np.uint8(contrast_frame_normal)
    # contrast_frame = contrast_frame / np.linalg.norm(contrast_frame) * 255

    cv2.imshow('frame1_luma', cv2.resize(np.uint8(frame1_luma), None, fx=0.8, fy=0.8))
    cv2.imshow('frame2_luma', cv2.resize(np.uint8(frame2_luma), None, fx=0.8, fy=0.8))
    cv2.imshow('contrast_frame before normalization', cv2.resize(np.uint8(contrast_frame), None, fx=0.8, fy=0.8))
    # cv2.imshow('contrast_frame after normalization', cv2.resize(contrast_frame_normal, None, fx=0.8, fy=0.8))
    cv2.imshow('contrast_frame after normalization', contrast_frame_normal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(MEDIA_SAVE_PATH, MEDIA_SAVE_NAME), contrast_frame_normal)
    # cv2.imwrite(os.path.join(MEDIA_SAVE_LUMA_PATH, MEDIA_SAVE_LUMA_NAME), frame2_luma)

