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
MEDIA_PATH = '../res/monochromatic_illumination/python_tests/'
# FILE_NAME = 'test_video1'
LIGHTING_TYPE = 'test2'             ### CHANGE 4 EVERY TEST
MEDIA_NAME = 'test_video2.mkv'      ### CHANGE 4 EVERY TEST
STATS_NAME = 'luma_values a.csv'
BONE_RECT_POS = [670, 395]       # rectangle selection within image
BONE_RECT_SIZE = [88, 59]
DURA_RECT_POS = [632, 285]       # rectangle selection within image
DURA_RECT_SIZE = [88, 59]
NB_FRAME_SKIP = 2

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
                # Take desired rectangles for bone & dura:
                rect_bone = frame_luma[BONE_RECT_POS[1]:BONE_RECT_POS[1] + BONE_RECT_SIZE[1], BONE_RECT_POS[0]:BONE_RECT_POS[0] + BONE_RECT_SIZE[0]]
                rect_dura = frame_luma[DURA_RECT_POS[1]:DURA_RECT_POS[1] + DURA_RECT_SIZE[1], DURA_RECT_POS[0]:DURA_RECT_POS[0] + DURA_RECT_SIZE[0]]

                # # stats array (sample x features) : luminance features=["tissue_type", mean, std]
                # stats = np.append(stats, [[LIGHTING_TYPE, frame_counter, "bone", np.mean(rect_bone), np.std(rect_bone)]], axis=0)
                # stats = np.append(stats, [[LIGHTING_TYPE, frame_counter, "dura", np.mean(rect_dura), np.std(rect_dura)]], axis=0)

                stats = np.append(stats, [[LIGHTING_TYPE, np.mean(rect_bone), np.mean(rect_dura)]], axis=0)
            frame_counter += 1
        else:
            break

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
    cv2.destroyAllWindows()

