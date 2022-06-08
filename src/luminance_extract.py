# Convert image/video to luminance photos
# For every 10th frame from a given video,
# Find luminance mean & std values in a rectangular part of the image corresponding to bone/dura
# input: video
# output: .csv file with array (sample x features): features = [test_type, mean_tissue1, mean_tissue2]

import os
import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt

# # single frame:
# MEDIA_PATH = '../res/monochromatic_illumination/2022-05-02 experiment1/'
# MEDIA_SAVE_PATH = '../res/monochromatic_illumination/2022-05-02 experiment1/luminance/'
# MEDIA_NAME = '430nm.png'
# IMAGE_NAME_LUMA = '430nm_luma.png'
# video:
MEDIA_PATH = '../res/monochromatic_illumination/2022-06-01 Experiment 2/position1/'
# FILE_NAME = 'test_video1'
LIGHTING_TYPE = '700nm'             ### CHANGE 4 EVERY TEST
MEDIA_NAME = '700.mkv'            ### CHANGE 4 EVERY TEST
STATS_NAME = 'test2-pos1.csv'            ### Don't change, data is added to the same excel
# # TEST 1
# BONE_RECT_POS = [670, 395]       # rectangle selection within image
# BONE_RECT_SIZE = [88, 59]
# DURA_RECT_POS = [632, 285]       # rectangle selection within image
# DURA_RECT_SIZE = [88, 59]
# # TEST 2
# BONE_RECT_POS = [585, 222]       # rectangle selection within image
# BONE_RECT_SIZE = [37, 37]
# DURA_RECT_POS = [433, 317]       # rectangle selection within image
# DURA_RECT_SIZE = [37, 37]
NB_FRAME_SKIP = 2
RECT_DISPLAY = True
IS_SAVE_STATS = True


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self):
        return self.x, self.y


class Rectangle:
    def __init__(self, px_GIMP, py_GIMP, size_x_GIMP, size_y_GIMP):
        # self.px = py_GIMP
        # self.py = px_GIMP
        # self.size_x = size_y_GIMP
        # self.size_y = size_x_GIMP
        # self.start_pos = (py_GIMP, px_GIMP)
        # self.end_pos = (py_GIMP + size_y_GIMP, px_GIMP + size_x_GIMP)
        self.start_pos = Position(py_GIMP, px_GIMP)
        self.end_pos = Position(py_GIMP + size_y_GIMP, px_GIMP + size_x_GIMP)

        # self.build_xy(self.start_pos)
        # self.build_xy(self.end_pos)
    #
    # def build_xy(self, attrib):
    #     setattr(attrib, 'x', attrib[0])
    #     setattr(attrib, 'y', attrib[1])

# # Pos 3: covered bone
# bone_rect = Rectangle(382, 479, 30, 30)         # rectangle pos & dimensions in GIMP programme
# dura_rect = Rectangle(475, 317, 30, 30)

# # Pos 2
# bone_rect = Rectangle(502, 519, 85, 85)         # rectangle pos & dimensions in GIMP programme
# dura_rect = Rectangle(512, 383, 85, 85)

# Pos 1 : [430-470nm, 515]
bone_rect = Rectangle(382, 333, 60, 60)         # rectangle pos & dimensions in GIMP programme
dura_rect = Rectangle(562, 157, 60, 60)
# Pos 1 : [502, 621, 640, 660, 700]
bone_rect = Rectangle(348, 333, 60, 60)         # rectangle pos & dimensions in GIMP programme
dura_rect = Rectangle(528, 157, 60, 60)

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
    # stats = [["lighting_type", "bone", "dura", "luma_mean", "luma_std"]]
    stats = [["lighting_type", "luma_bone", "luma_dura"]]
    frame_counter = 0
    sample_counter = 0

    # # image:
    # frame = cv2.imread(os.path.join(MEDIA_PATH, MEDIA_NAME))
    # # cv2.imshow('original', frame)
    # frame_luma = bgr2luma(frame)
    # # # average RGB
    # # frame_average = np.mean(frame, 2)
    # # cv2.imwrite(os.path.join(IMAGE_SAVE_PATH, IMAGE_NAME_LUMA), frame_luma)

    # Load video:
    cap = cv2.VideoCapture(os.path.join(MEDIA_PATH, MEDIA_NAME))

    # take values for every 10 frames
    while cap.isOpened():
        print('Cap Open')
        ret, frame = cap.read()
        if ret:
            if frame_counter % NB_FRAME_SKIP == 0:
                print(frame_counter)
                frame_luma = bgr2luma(frame)
                # # Take desired rectangles for bone & dura:
                # rect_bone = frame_luma[BONE_RECT_POS[1]:BONE_RECT_POS[1] + BONE_RECT_SIZE[1], BONE_RECT_POS[0]:BONE_RECT_POS[0] + BONE_RECT_SIZE[0]]
                # rect_dura = frame_luma[DURA_RECT_POS[1]:DURA_RECT_POS[1] + DURA_RECT_SIZE[1], DURA_RECT_POS[0]:DURA_RECT_POS[0] + DURA_RECT_SIZE[0]]

                # Take desired rectangles for bone & dura:
                rect_bone = frame_luma[bone_rect.start_pos.x:bone_rect.end_pos.x, bone_rect.start_pos.y:bone_rect.end_pos.y]
                rect_dura = frame_luma[dura_rect.start_pos.x:dura_rect.end_pos.x, dura_rect.start_pos.y:dura_rect.end_pos.y]

                if frame_counter == 0 and RECT_DISPLAY:
                    # Display rectangles
                    frame_rects = np.copy(frame)
                    cv2.rectangle(frame_rects, tuple(reversed(bone_rect.start_pos())), tuple(reversed(bone_rect.end_pos())), (255, 0, 0), thickness=2)
                    cv2.rectangle(frame_rects, tuple(reversed(dura_rect.start_pos())), tuple(reversed(dura_rect.end_pos())), (255, 0, 0), thickness=2)
                    cv2.imshow('frame_rects', frame_rects)

                    # Print rectangles
                    cv2.imshow('bone', cv2.resize(np.uint8(rect_bone), None, fx=10, fy=10))
                    cv2.imshow('dura', cv2.resize(rect_dura, None, fx=10, fy=10))
                    cv2.waitKey(0)

                    # Print rectangles
                    f = plt.figure()
                    # Plot the data using imshow with gray colormap
                    f.add_subplot(1, 2, 1)
                    plt.imshow(rect_bone, cmap='gray')
                    f.add_subplot(1, 2, 2)
                    # plt.show()
                    # f = plt.figure()
                    # Plot the data using imshow with gray colormap
                    plt.imshow(rect_dura, cmap='gray')
                    plt.show()

                    cv2.destroyAllWindows()

                # # stats array (sample x features) : luminance features=["tissue_type", mean, std]
                # stats = np.append(stats, [[LIGHTING_TYPE, frame_counter, "bone", np.mean(rect_bone), np.std(rect_bone)]], axis=0)
                # stats = np.append(stats, [[LIGHTING_TYPE, frame_counter, "dura", np.mean(rect_dura), np.std(rect_dura)]], axis=0)

                sample_counter += 1
                stats = np.append(stats, [[LIGHTING_TYPE, np.mean(rect_bone), np.mean(rect_dura)]], axis=0)
            frame_counter += 1
        else:
            break
    if IS_SAVE_STATS:
        with open(os.path.join(MEDIA_PATH, STATS_NAME), 'a', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=",")
            my_writer.writerows(stats)


    # f = plt.figure()
    # # Plot the data using imshow with gray colormap
    # # f.add_subplot(1, 2, 0 + 1)
    # plt.imshow(frame_small, cmap='gray')
    # # Display the plot
    # plt.show()

    # while True:
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
    #
    print('sample counter', sample_counter)
    cv2.destroyAllWindows()

