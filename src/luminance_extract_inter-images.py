# Compute luminance difference between two rectangles in an image/video
# For every Xth frame from a given video,
# Find luminance mean & std values in a rectangular part of the image corresponding to bone/dura
# input: video
# output: .csv file with array (sample x features): features = [test_type, mean_tissue1, mean_tissue2]

import os
import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt

# # single frame:
# MEDIA_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/image_division_uint8_after_normal/position3/'
# MEDIA_SAVE_PATH = '../res/monochromatic_illumination/2022-05-02 experiment1/luminance/'
# MEDIA_NAME = '430nm_700nm_clahe.png'
# IMAGE_NAME_LUMA = '430nm_luma.png'
# video:
MEDIA_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/position3/'
MEDIA_NAMES = ['430', '461', '470', '502', '515', '600', '621', '640', '660', '700', 'white']
# MEDIA_NAMES = ['430', '660', '700', 'white']
# MEDIA_NAME1 = '430.mkv'            ### CHANGE 4 EVERY TEST
# MEDIA_NAME2 = '700.mkv'            ### CHANGE 4 EVERY TEST
# LIGHTING_TYPE = '430/700nm'             ### CHANGE 4 EVERY TEST

MEDIA_SAVE_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/contrast_inter_image/'
STATS_NAME = 'absdiff.csv'            ### Don't change, data is added to the same excel
NB_FRAME_SKIP = 2
RECT_DISPLAY = False
IS_SAVE_STATS = True
IS_CLAHE = False         # is perform CLAHE equalization


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


# # Pos 3: covered bone
# bone_rect = Rectangle(382, 479, 30, 30)         # rectangle pos & dimensions in GIMP programme
# dura_rect = Rectangle(475, 317, 30, 30)

# Pos 3: covered bone - large rectangles
bone_rect = Rectangle(359, 464, 85, 85)         # rectangle pos & dimensions in GIMP programme
dura_rect = Rectangle(460, 300, 85, 85)

# # Pos 2
# bone_rect = Rectangle(502, 519, 85, 85)         # rectangle pos & dimensions in GIMP programme
# dura_rect = Rectangle(512, 383, 85, 85)

# # Pos 1 : [430-470nm, 515]
# bone_rect = Rectangle(382, 333, 60, 60)         # rectangle pos & dimensions in GIMP programme
# dura_rect = Rectangle(562, 157, 60, 60)
# # Pos 1 : [502, 621, 640, 660, 700]
# bone_rect = Rectangle(348, 333, 60, 60)         # rectangle pos & dimensions in GIMP programme
# dura_rect = Rectangle(528, 157, 60, 60)


# # Experiment 1:
# bone_rect = Rectangle(680, 385, 70, 70)         # rectangle pos & dimensions in GIMP programme
# dura_rect = Rectangle(604, 265, 70, 70)

'''-------------------------------Main code--------------------------------------------------------'''

# def srgb_to_linsrgb (srgb):
#     """Convert sRGB values to physically linear ones. The transformation is
#        uniform in RGB, so *srgb* can be of any shape.
#
#        *srgb* values should range between 0 and 1, inclusively.
#
#     """
#     gamma = ((srgb + 0.055) / 1.055)**2.4
#     scale = srgb / 12.92
#     return np.where(srgb > 0.04045, gamma, scale)


def bgr2luma(frame):
    # R*0.222 + G*0.717 + B*0.061
    luma = frame[:, :, 0] * 0.061 + frame[:, :, 1] * 0.717 + frame[:, :, 2] * 0.222
    return luma


if __name__ == "__main__":
    stats = [["lighting_type", "luma_bone", "luma_dura"]]
    frame_counter = 0
    sample_counter = 0

    '''
    # image:
    frame = cv2.imread(os.path.join(MEDIA_PATH, MEDIA_NAME))
    frame_luma = bgr2luma(frame)

    # Take desired rectangles for bone & dura:
    rect_bone = frame_luma[bone_rect.start_pos.x:bone_rect.end_pos.x, bone_rect.start_pos.y:bone_rect.end_pos.y]
    rect_dura = frame_luma[dura_rect.start_pos.x:dura_rect.end_pos.x, dura_rect.start_pos.y:dura_rect.end_pos.y]
    stats = np.append(stats, [[LIGHTING_TYPE, np.mean(rect_bone), np.mean(rect_dura)]], axis=0)

    frame_rects = np.copy(frame_luma)
    cv2.rectangle(frame_rects, tuple(reversed(bone_rect.start_pos())), tuple(reversed(bone_rect.end_pos())),
                  (255, 0, 0), thickness=2)
    cv2.rectangle(frame_rects, tuple(reversed(dura_rect.start_pos())), tuple(reversed(dura_rect.end_pos())),
                  (255, 0, 0), thickness=2)
    cv2.imshow('frame', frame)
    cv2.imshow('frame_luma', np.uint8(frame_luma))
    cv2.imshow('frame_rects', np.uint8(frame_rects))
    cv2.waitKey(0)

    # # average RGB
    # frame_average = np.mean(frame, 2)
    # cv2.imwrite(os.path.join(IMAGE_SAVE_PATH, IMAGE_NAME_LUMA), frame_luma)

    if IS_SAVE_STATS:
        with open(os.path.join(MEDIA_PATH, STATS_NAME), 'a', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=",")
            my_writer.writerows(stats)
    '''

    # go through all combinations of XXnm / XXnm
    lighting_type = []
    media_name1 = []
    media_name2 = []
    temp = []
    for i in MEDIA_NAMES:
        for j in MEDIA_NAMES:
            if i != j:
                media_name1 = i + '.mkv'
                media_name2 = j + '.mkv'
                # media_name2 = os.path.join(j, '.mkv')
                lighting_type = i + '/' + j


                # Load video:
                cap1 = cv2.VideoCapture(os.path.join(MEDIA_PATH, media_name1))
                cap2 = cv2.VideoCapture(os.path.join(MEDIA_PATH, media_name2))

                # take values for every 10 frames
                while cap1.isOpened() and cap2.isOpened():
                    # print('Cap Open')
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    if ret1 and ret2:
                        if frame_counter % NB_FRAME_SKIP == 0:
                            print(frame_counter)

                            frame1_luma = bgr2luma(frame1)
                            frame2_luma = bgr2luma(frame2)

                            if IS_CLAHE:
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
                                frame1_luma = clahe.apply(np.uint8(np.copy(frame1_luma)))
                                frame2_luma = clahe.apply(np.uint8(np.copy(frame2_luma)))

                            # Inter-image contrast map:
                            np.seterr(invalid='ignore')
                            # contrast_map = cv2.divide(frame1_luma, frame2_luma)
                            contrast_map = cv2.absdiff(frame1_luma, frame2_luma)
                            contrast_map_normal = cv2.normalize(contrast_map, None, alpha=0, beta=255,
                                                                norm_type=cv2.NORM_MINMAX)

                            contrast_map_normal = np.uint8(contrast_map_normal)

                            # Take desired rectangles for bone & dura:
                            rect_bone = contrast_map_normal[bone_rect.start_pos.x:bone_rect.end_pos.x, bone_rect.start_pos.y:bone_rect.end_pos.y]
                            rect_dura = contrast_map_normal[dura_rect.start_pos.x:dura_rect.end_pos.x, dura_rect.start_pos.y:dura_rect.end_pos.y]

                            if frame_counter == 0:
                                print('lighting_type', lighting_type)
                            if frame_counter < 3 and RECT_DISPLAY:
                                # Display rectangles
                                frame_rects = np.copy(contrast_map_normal)
                                cv2.rectangle(frame_rects, tuple(reversed(bone_rect.start_pos())), tuple(reversed(bone_rect.end_pos())), (255, 0, 0), thickness=2)
                                cv2.rectangle(frame_rects, tuple(reversed(dura_rect.start_pos())), tuple(reversed(dura_rect.end_pos())), (255, 0, 0), thickness=2)
                                cv2.imshow('frame_rect', frame_rects)
                                cv2.imshow('frame1_luma', np.uint8(frame1_luma))
                                cv2.imshow('frame2_luma', np.uint8(frame2_luma))
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()

                            sample_counter += 1
                            # append stats only if rects are not empty
                            if np.mean(rect_bone) != 0 and np.mean(rect_dura) != 0:
                                # print('add stats', np.mean(rect_bone), ' ', np.mean(rect_dura))
                                stats = np.append(stats, [[lighting_type, np.mean(rect_bone), np.mean(rect_dura)]], axis=0)
                        frame_counter += 1
                    else:
                        break

                if IS_SAVE_STATS:
                    with open(os.path.join(MEDIA_SAVE_PATH, STATS_NAME), 'a', newline='') as csvfile:
                        my_writer = csv.writer(csvfile, delimiter=",")
                        my_writer.writerows(stats)

                frame_counter = 0
                sample_counter = 0
    #print('sample counter', sample_counter)
    cv2.destroyAllWindows()


