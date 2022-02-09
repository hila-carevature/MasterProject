# Processing of the image/video for extracting dura&bone
import cv2
import numpy as np
import math
import skimage.io
import skimage.measure
import os

MEDIA_TYPE = 'image'                            # 'image' or 'video'
MEDIA_PATH = '../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short/cropped/frame142.jpg'
# MEDIA_PATH = '../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram/swab_wonder.PNG'

# MEDIA_TYPE = 'video'                          # 'image' or 'video'
# MEDIA_PATH ='C:/Users/User/Dropbox (Carevature Medical)/Robotic Decompression/Media/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short.mp4'
VIDEO_OUT_PATH = '../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram'
VIDEO_OUT_NAME = 'Foraminotomy - short - out HSV-growing_coloured.avi'
# MEDIA_PATH ='C:/Users/User/Dropbox (Carevature Medical)/Robotic Decompression/Media/210829 animal carevature_CASE0005 Robotic/Keynan/Foraminotomy.mp4'
# VIDEO_OUT_PATH = '../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Keynan'
# VIDEO_OUT_NAME = 'Foraminotomy - out HSV-coloured.avi'


# Filters in HSV
BONE_LOWER_RANGE = np.array([0, 0, 163])        # np.array([0, 0, 0])
BONE_UPPER_RANGE = np.array([179, 255, 255])    # np.array([179, 255, 175])
DURA_LOWER_RANGE = np.array([126, 0, 0])        # np.array([0, 39, 0])
DURA_UPPER_RANGE = np.array([166, 255, 255])    # np.array([169, 255, 149])
KERNEL_MORPH = np.ones((21, 21), np.uint8)      # kernel for morphology close & open
# HSV_MAX = np.asarray((179, 255, 255))

IS_REGION_GROWING = False                       # boolean if to apply region growing or not
IS_REDO_GROWING = False
IS_REDO_MERGING = False
IMG_COLOR = 1
IMG_GRAY = 0
HOMOGENEITY_THRESHOLD = 0.75
MIN_SIZE_REGION = 50
MAX_SIZE_REGION = 10000
REGION_LABELS_FILE = 'labels.npy'
REGION_NB_LABELS_FILE = 'nb_labels.npy'
REGION_MERGED_LABEL_PXL_FILE = 'merged_labels.npy'
REGION_MERGED_LABELS_FILE = 'merged_label_list.npy'
DISTANCE_THRESH = 20                            # maximum distance between objects for merging


def exit_programme():
    cv2.destroyAllWindows()
    exit()


def find_mask(image, lower_range, upper_range, inv_bool, kernel):
    """
    Compute mask, given ranges for each channel in image. Keeps pixels whose values are within the lower
    & upper ranges
    :param image: image to be masked
    :param lower_range: (typically 1x3)
    :param upper_range: (typically 1x3)
    :param inv_bool: bool determining if mask needs to be inverted (if we use the inverse of the ranges)
    :param kernel: kernel size for morphology operations
    :return: mask (same dimension as input image)
    """
    img = np.copy(image)
    mask = cv2.inRange(img, lower_range, upper_range)
    if inv_bool:
        mask = cv2.bitwise_not(mask)
    # 'close' the image to remove noise & fill holes inside object
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 'open' the image to smoothen boundaries
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def image_processing(image):
    """
    Finds regions with bone & dura, computes their corresponding masks and returns an image with the dura & bone regions
    :param image: in BGR
    :return: masked image, colored image(original image with overlaying colored masks)
    """
    # Convert the BGR image to HSV image.
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute masks
    bone_mask = find_mask(hsv_frame, BONE_LOWER_RANGE, BONE_UPPER_RANGE, False, KERNEL_MORPH)
    dura_mask = find_mask(hsv_frame, DURA_LOWER_RANGE, DURA_UPPER_RANGE, False, KERNEL_MORPH)

    # create color image
    bone_overlay = color_overlay(image, bone_mask, (0, 255, 255), 0.2)
    bone_n_dura_overlay = color_overlay(bone_overlay, dura_mask, (255, 0, 0), 0.2)

    # Combine the two images == combine masks
    bone_n_dura = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_or(bone_mask, dura_mask))

    return bone_n_dura, bone_n_dura_overlay


def color_overlay(img, mask, color, mask_transparency):
    # create color image
    colored_image = color * np.ones(img.shape)
    # convert masks into colored masks
    colored_mask = cv2.bitwise_and(colored_image, colored_image, mask=mask)
    combined_image = cv2.addWeighted(img, 1, np.asarray(colored_mask, img.dtype), mask_transparency, 0)
    return combined_image


'''-------------------------------Region Growing--------------------------------------------------------'''


# Region growing
def homogeneity_criterion(int_1, int_2, img_type):
    # return if the homogenity criterion is respected between the two pixels
    if img_type == IMG_GRAY:
        if int_1 < (HOMOGENEITY_THRESHOLD + int_2) and int_2 < (HOMOGENEITY_THRESHOLD + int_1):
            return True
        else:
            return False
    else:
        # int_val_1 = math.sqrt(int_1[2] ** 2 + int_1[1] ** 2 + int_1[0] ** 2)
        # int_val_2 = math.sqrt(int_2[2] ** 2 + int_2[1] ** 2 + int_2[0] ** 2)
        int_val_1 = math.sqrt(int_1[0] ** 2 + int_1[1] ** 2)
        int_val_2 = math.sqrt(int_2[0] ** 2 + int_2[1] ** 2)
        if np.abs(int_val_1 - int_val_2) < HOMOGENEITY_THRESHOLD:
            return True
        else:
            return False

        # #normalise pixel values
        # int_1_norm = int_1 / HSV_MAX
        # int_2_norm = int_2 / HSV_MAX
        # # if np.abs(int_1_norm[0] - int_2_norm[0]) < HOMOGENEITY_THRESHOLD and \
        # #         np.abs(int_1_norm[1] - int_2_norm[1]) < HOMOGENEITY_THRESHOLD and \
        # #         np.abs(int_1_norm[2] - int_2_norm[2]) < HOMOGENEITY_THRESHOLD:
        # if np.abs(int_1_norm[2] - int_2_norm[2]) < HOMOGENEITY_THRESHOLD:
        #     return True
        # else:
        #     return False


def in_range(x, y, size_x, size_y):
    # return if the pixel is in the range of the image
    return size_x > x >= 0 and size_y > y >= 0


def check_and_add_neighbours(im, x, y, current_label, counter, labels, queue_list, img_type):
    size_x, size_y = im.shape[0:2]
    # for each neighbour, if he respects the homogeneity criterion and has not already a label, add it to the queue list
    for i, j in zip([-1, 0, 0, 1, -1, 1, -1, 1], [0, 1, -1, 0, -1, 1, 1, -1]):
        xb = x + i
        yb = y + j
        if in_range(xb, yb, size_x, size_y) and labels[xb, yb] == -1 and homogeneity_criterion(im[x, y], im[xb, yb],
                                                                                               img_type):
            labels[xb, yb] = current_label
            queue_list.append([xb, yb])
            counter = counter + 1
    return counter


def region_growing(image, img_type):
    print('started region growing')
    # proceeds to region growing on the entire image
    size_x, size_y = image.shape[0:2]
    # Init of variables
    queue_list = []
    labels = np.ones((size_x, size_y), int) * (-1)
    current_point_x = -1  # because we will start by incrementing the point x
    current_point_y = 0
    current_label = 0  # note that first label will be 1
    counter = 0

    # Main loop
    while counter != size_x * size_y:  # while we didn't check all the points on the image
        # Update current point position and check if we reached the end
        if current_point_x < size_x - 1:
            current_point_x = current_point_x + 1
        elif current_point_y < size_y - 1:
            current_point_x = 0
            current_point_y = current_point_y + 1

        # Test if the point is labelled and create a new label if needed
        if labels[current_point_x, current_point_y] == -1:
            current_label = current_label + 1
            labels[current_point_x, current_point_y] = current_label
            counter = counter + 1
            counter = check_and_add_neighbours(image, current_point_x, current_point_y, current_label, counter, labels,
                                               queue_list, img_type)  # first seed

        # While this label has still points to add, we add them and look at their neighbours
        while len(queue_list) != 0:
            current_queue_point_x = queue_list[0][0]
            current_queue_point_y = queue_list[0][1]
            counter = check_and_add_neighbours(image, current_queue_point_x, current_queue_point_y, current_label,
                                               counter,
                                               labels, queue_list, img_type)
            queue_list.pop(0)

    return labels, current_label


# computes the area(pixels) and coordinates of all the pixels included in the object
def find_region_coords(mask_obj, img_size):
    bool_im = np.zeros(img_size, dtype=np.int8)
    bool_im[mask_obj > 0] = 1
    region_list = skimage.measure.regionprops(bool_im)
    area = region_list[0].area
    pxl_coords = region_list[0].coords
    return pxl_coords, area


def eucl_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def merge_regions(labeled_pxl, nb_labels, img):
    print('started merging')
    # Join regions that are close to each other.
    # Find the labels of large-area objects (saved in big_label_list) & their center position

    # remove small objects from label_list
    labels_list, label_pxl_count = np.unique(labeled_pxl, return_counts=True)             # 2D array [label, nb_pixel w/ label]
    # delete label corresponding to small object
    small_pxl_label_id = np.concatenate((np.where(MIN_SIZE_REGION > label_pxl_count)[0], np.where(label_pxl_count > MAX_SIZE_REGION)[0]))
    big_label_list = np.delete(labels_list, small_pxl_label_id)

    obj_center = []
    # compute object centers for each object
    for current_label in big_label_list:
        mask_region = cv2.inRange(labeled_pxl, int(current_label), int(current_label))
        pxl_coords, area = find_region_coords(mask_region, img.shape[:2])
        obj_center.append(np.mean(pxl_coords, 0).astype(int))

    # merging objects close to each other
    merged_label_list = np.copy(big_label_list)
    merged_labeled_pxl = np.copy(labeled_pxl)

    # go through large objects label list. For objects close to each other, label their corresponding pixels with the same label
    for i in range(len(merged_label_list) - 1):
        for j in range(i + 1, len(merged_label_list)):
            if eucl_distance(obj_center[i], obj_center[j]) < DISTANCE_THRESH:
                merged_labeled_pxl[labeled_pxl == merged_label_list[i]] = merged_label_list[i]
                merged_labeled_pxl[labeled_pxl == merged_label_list[j]] = merged_label_list[i]
                merged_label_list[j] = merged_label_list[i]
    print('finished merging')
    return merged_labeled_pxl, np.unique(merged_label_list)


def apply_growing_n_merge(img, img_type):
    cv2.imshow('input image', img)
    cv2.waitKey(0)

    # apply region growing & merging of close regions together
    if IS_REDO_GROWING:
        # UNCOMMENT to compute region growing
        # do region growing to label each region in final image]
        # on HSV image
        # [labeled_pxl, nb_labels] = region_growing(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), img_type)
        [labeled_pxl, nb_labels] = region_growing(img, img_type)
        print('nb_labels', nb_labels)
        np.save(REGION_LABELS_FILE, labeled_pxl)
        np.save(REGION_NB_LABELS_FILE, nb_labels)
        # exit_programme()
    else:
        # load region growing arrays
        labeled_pxl = np.load(REGION_LABELS_FILE)
        nb_labels = np.load(REGION_NB_LABELS_FILE)
    if IS_REDO_MERGING:
        merged_labeled_pxl, merged_label_list = merge_regions(labeled_pxl, nb_labels, img)
        np.save(REGION_MERGED_LABEL_PXL_FILE, merged_labeled_pxl)
        np.save(REGION_MERGED_LABELS_FILE, merged_label_list)
    else:
        # load merged regions
        merged_labeled_pxl = np.load(REGION_MERGED_LABEL_PXL_FILE)
        merged_label_list = np.load(REGION_MERGED_LABELS_FILE)

    # display each label

    for current_label in np.unique(labeled_pxl):
        current_mask_region = cv2.inRange(labeled_pxl, int(current_label), int(current_label))
        nb_pxl = np.count_nonzero(current_mask_region)
        print('nb pxls', nb_pxl)
        if nb_pxl > MIN_SIZE_REGION:
            region = cv2.bitwise_and(img, img, mask=current_mask_region)
            cv2.imshow('region', region)
            key1 = cv2.waitKey(0)
            # if press escape key
            if key1 == 27:
                break

    # combine masks of all objects into one
    mask_region = cv2.inRange(merged_labeled_pxl, int(merged_label_list[0]), int(merged_label_list[0]))
    for current_label in merged_label_list[1:]:         # start from 2nd element
        # choose one region
        current_mask_region = cv2.inRange(merged_labeled_pxl, int(current_label), int(current_label))
        mask_region = cv2.bitwise_or(mask_region, current_mask_region)
        # mask_region = cv2.bitwise_and(img, img, mask=cv2.bitwise_or(mask_region, current_mask_region))
        # print('nb pxls', np.count_nonzero(mask_region))
        # region = cv2.bitwise_and(img, img, mask=mask_region)
        # cv2.imshow('region', region)
        # key1 = cv2.waitKey(0)
        # # if press escape key
        # if key1 == 27:
        #     break
        #
    return mask_region
    # region = cv2.bitwise_and(img, img, mask=mask_region)
    # cv2.imshow('region', region)
    # cv2.waitKey(0)


'''-------------------------------Canny filter--------------------------------------------------------'''


def canny_filter(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    v = np.median(img_blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))

    # cv2.imshow('input image', cv2.resize(img, None, fx=0.6, fy=0.6))
    cv2.imshow('blurred image', cv2.resize(img, None, fx=0.8, fy=0.8))

    # frame_edges = cv2.Canny(img_blur, lower, upper, 3)
    frame_edges = cv2.Canny(img_blur, 10, 50)
    print('lower', lower)
    print('upper', upper)
    cv2.imshow('canny', cv2.resize(frame_edges, None, fx=0.8, fy=0.8))
    cv2.waitKey(0)
    return


'''
# Preprocessing steps: examples
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# remove noise
frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)

# Find edges
# convolute with proper kernels
v = np.median(image)
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))
frame_edges = cv2.Canny(frame_blur, lower, upper, 3)
# frame_edges = cv2.Canny(frame_blur, 60, 100, 1)
# 'close' the image
kernel_edge = np.ones((3, 3), np.uint8)
close = np.copy(frame_edges)
frame_closed = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel_edge)

cv2.imshow('original', frame)
cv2.waitKey(0)
cv2.imshow('gray', frame_gray)
cv2.waitKey(0)
cv2.imshow('blurry', frame_blur)
cv2.waitKey(0)
cv2.imshow('edges', frame_edges)
cv2.waitKey(0)
cv2.imshow('closed', frame_closed)
cv2.waitKey(0)
'''
'''-------------------------------Main code--------------------------------------------------------'''

# main programme
if MEDIA_TYPE == 'image':
    # load image
    frame = cv2.imread(MEDIA_PATH)
    frame_masked, colored_frame = image_processing(frame)

    # canny_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    canny_filter(frame[:, :, 1])
    # cv2.imwrite('../res/surgery_images/frame52GreenChannel.jpg', frame[:, :, 1])

    if IS_REGION_GROWING:
        # region growing
        # frame = frame[:, :, 1]
        # mask = apply_growing_n_merge(frame, IMG_GRAY)
        mask = apply_growing_n_merge(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), IMG_COLOR)
        # mask = apply_growing_n_merge(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), IMG_COLOR)
        frame_grown = cv2.bitwise_and(frame, frame, mask=mask)
        # frame_grown = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        cv2.imshow('frame_grown', cv2.resize(frame_grown, None, fx=0.6, fy=0.6))
        cv2.waitKey(0)

        # color overlay
        overlay_region = color_overlay(frame, mask, (255, 0, 0), 0.4)
        cv2.imshow('grown_overlay', cv2.resize(overlay_region, None, fx=0.6, fy=0.6))
        cv2.waitKey(0)

elif MEDIA_TYPE == 'video':
    # Load video:
    cap = cv2.VideoCapture(MEDIA_PATH)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # create directory for saving images: if it doesn't already exists
    try:
        os.makedirs(VIDEO_OUT_PATH)
    except OSError:
        pass

    out = cv2.VideoWriter(os.path.join(VIDEO_OUT_PATH, VIDEO_OUT_NAME), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (frame_width, frame_height))
    while cap.isOpened():
        print('Cap Open')
        ret, frame = cap.read()
        if ret:
            frame_masked, colored_frame = image_processing(frame)
            # cv2.imshow('masked', cv2.resize(colored_frame, None, fx=0.6, fy=0.6))
            # key = cv2.waitKey(0)
            # # if press escape key
            # if key == 27:
            #     cap.release()
            #     out.release()
            #     break
            # save filtered frame in video

            if IS_REGION_GROWING:
                # region growing
                mask = apply_growing_n_merge(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), IMG_COLOR)
                # color overlay
                colored_frame = color_overlay(frame, mask, (255, 0, 0), 0.4)

            out.write(colored_frame)  # to store the video
        else:
            cap.release()
            out.release()
            break
print('finished')
cv2.destroyAllWindows()
