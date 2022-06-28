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
MEDIA_PATH1 = '../res/monochromatic_illumination/2022-06-15 Experiment 4/430nm.png'
MEDIA_PATH2 = '../res/monochromatic_illumination/2022-06-15 Experiment 4/700nm.png'
MEDIA_SAVE_NAME1 = '430nm.png'
MEDIA_SAVE_NAME2 = '700nm.png'
MEDIA_SAVE_NAME = 'divide430_700nm.png'
MEDIA_SAVE_PATH = '../res/monochromatic_illumination/2022-06-15 Experiment 4/luma/'
# MEDIA_SAVE_LUMA_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/luma/position1/'
# MEDIA_SAVE_LUMA_NAME = 'white.png'
IS_DISPLAY = True
IS_SAVE = True

IS_ENHANCE = True
ENHANCE_MADIA_PATH = '../res/monochromatic_illumination/2022-06-15 Experiment 4/white.png'
ENHANCE_MADIA_PATH2 = '../res/monochromatic_illumination/2022-06-15 Experiment 4/luma/divide430_700nm.png'
ENHANCE_SAVE_NAME = 'divide430_700nm.png'
ENHANCE_SAVE_PATH = '../res/monochromatic_illumination/2022-06-15 Experiment 4/enhancing/'


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


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self):
        return self.x, self.y


class Rectangle:
    def __init__(self, px_GIMP, py_GIMP, size_x_GIMP, size_y_GIMP):
        self.start_pos = Position(py_GIMP, px_GIMP)
        self.end_pos = Position(py_GIMP + size_y_GIMP, px_GIMP + size_x_GIMP)


# Pos 3: covered bone - large rectangles
bone_rect = Rectangle(359, 464, 85, 85)         # rectangle pos & dimensions in GIMP programme
dura_rect = Rectangle(460, 300, 85, 85)


'''----------------------main-----------------------------------------------------------------'''
if __name__ == "__main__":

    # load image
    frame1 = cv2.imread(MEDIA_PATH1)
    frame2 = cv2.imread(MEDIA_PATH2)
    # Convert BGR image into luminance images
    frame1_luma = bgr2luma(frame1)
    frame2_luma = bgr2luma(frame2)

    frame1_clahe = equal_clahe(frame1_luma)
    frame2_clahe = equal_clahe(frame2_luma)
    # frame1_luma = frame1_clahe
    # frame2_luma = frame2_clahe

    # # division: Image1 (high contrast) / Image 2 (low contrast)
    # # Supress/hide the warning
    # np.seterr(invalid='ignore')
    # # contrast_frame = np.uint8(np.divide(frame1_luma, frame2_luma))
    contrast_frame = cv2.divide(frame1_luma, frame2_luma)

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

    if IS_DISPLAY:
        cv2.imshow('frame1_luma', cv2.resize(np.uint8(frame1_luma), None, fx=0.8, fy=0.8))
        cv2.imshow('frame2_luma', cv2.resize(np.uint8(frame2_luma), None, fx=0.8, fy=0.8))
        # cv2.imshow('contrast_frame before normalization', cv2.resize(np.uint8(contrast_frame), None, fx=0.8, fy=0.8))
        # cv2.imshow('contrast_frame after normalization', cv2.resize(contrast_frame_normal, None, fx=0.8, fy=0.8))
        cv2.imshow('contrast_frame after normalization', np.uint8(contrast_frame_normal))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if IS_SAVE:
        cv2.imwrite(os.path.join(MEDIA_SAVE_PATH, MEDIA_SAVE_NAME), np.uint8(contrast_frame_normal))
        cv2.imwrite(os.path.join(MEDIA_SAVE_PATH, MEDIA_SAVE_NAME1), frame1_luma)
        cv2.imwrite(os.path.join(MEDIA_SAVE_PATH, MEDIA_SAVE_NAME2), frame2_luma)
        # cv2.imwrite(os.path.join(MEDIA_SAVE_LUMA_PATH, MEDIA_SAVE_LUMA_NAME), frame2_luma)


    # enhance white image with 0-1 contrast image
    if IS_ENHANCE:
        white_im = cv2.imread(ENHANCE_MADIA_PATH)
        enhancing_im = cv2.imread(ENHANCE_MADIA_PATH2)
        enhancing_im = cv2.normalize(enhancing_im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        enhanced_im = np.float32(np.multiply(white_im, enhancing_im))
        enhanced_im = cv2.normalize(enhanced_im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imshow('white', white_im)
        cv2.imshow('enhancing', cv2.normalize(enhancing_im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imshow('enhanced', enhanced_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(ENHANCE_SAVE_PATH, ENHANCE_SAVE_NAME), enhanced_im)
