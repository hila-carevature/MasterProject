# Processing of the image/video for extracting dura&bone
import cv2
import numpy as np
import math
import skimage.io
import skimage.measure

MEDIA_TYPE = 'image'                            # 'image' or 'video'
MEDIA_PATH = '../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short/frame1_cropped.jpg'
# MEDIA_TYPE = 'video'                          # 'image' or 'video'
# MEDIA_PATH ='C:/Users/User/Dropbox (Carevature Medical)/Robotic Decompression/Media/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short.mp4'
# Filters in HSV
BONE_LOWER_RANGE = np.array([0, 0, 163])        # np.array([0, 0, 0])
BONE_UPPER_RANGE = np.array([179, 255, 255])    # np.array([179, 255, 175])
DURA_LOWER_RANGE = np.array([126, 0, 0])        # np.array([0, 39, 0])
DURA_UPPER_RANGE = np.array([166, 255, 255])    # np.array([169, 255, 149])
KERNEL_MORPH = np.ones((21, 21), np.uint8)      # kernel for morphology close & open

IMG_COLOR = 1
IMG_GREY = 0
HOMOGENEITY_THRESHOLD = 0.7
MIN_SIZE_REGION = 10
MAX_SIZE_REGION = 10000
REGION_LABELS_FILE = 'labels.npy'
REGION_NB_LABELS_FILE = 'nb_labels.npy'
REGION_MERGED_LABEL_PXL_FILE = 'merged_labels.npy'
REGION_MERGED_LABELS_FILE = 'merged_label_list.npy'
DISTANCE_THRESH = 15                            # maximum distance between objects for merging


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
    :return: masked image
    """
    # Convert the BGR image to HSV image.
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute masks
    bone_mask = find_mask(hsv_frame, BONE_LOWER_RANGE, BONE_UPPER_RANGE, False, KERNEL_MORPH)
    dura_mask = find_mask(hsv_frame, DURA_LOWER_RANGE, DURA_UPPER_RANGE, False, KERNEL_MORPH)

    # Combine the two images == combine masks
    bone_n_dura = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_or(bone_mask, dura_mask))

    # show images of bone & dura
    bone = cv2.bitwise_and(frame, frame, mask=bone_mask)
    dura = cv2.bitwise_and(frame, frame, mask=dura_mask)
    stacked = np.hstack((bone, dura))
    cv2.imshow('bone, dura', cv2.resize(stacked, None, fx=0.4, fy=0.4))
    cv2.waitKey(0)
    return bone_n_dura


'''-------------------------------Region Growing--------------------------------------------------------'''

# Region growing
def homogeneity_criterion(int_1, int_2, img_type):
    # return if the homogenity criterion is respected between the two pixels
    if img_type == IMG_GREY:
        if int_1 < (HOMOGENEITY_THRESHOLD + int_2) and int_2 < (HOMOGENEITY_THRESHOLD + int_1):
            return True
        else:
            return False
    else:
        int_val_1 = math.sqrt(int_1[2] ** 2 + int_1[1] ** 2 + int_1[0] ** 2)
        int_val_2 = math.sqrt(int_2[2] ** 2 + int_2[1] ** 2 + int_2[0] ** 2)
        if int_val_1 < (HOMOGENEITY_THRESHOLD + int_val_2) and int_val_2 < (HOMOGENEITY_THRESHOLD + int_val_1):
            return True
        else:
            return False


def in_range(x, y, size_x, size_y):
    # return if the pixel is in the range of the image
    return x < size_x and x >= 0 and y < size_y and y >= 0


def check_and_add_neighbours(im, x, y, current_label, counter, labels, queueList, IMG_TYPE):
    SIZE_X, SIZE_Y = im.shape[0:2]
    # for each neighbour, if he respects the homogeneity criterion and has not already a label, add it to the queue list
    for i, j in zip([-1, 0, 0, 1, -1, 1, -1, 1], [0, 1, -1, 0, -1, 1, 1, -1]):
        xb = x + i
        yb = y + j
        if in_range(xb, yb, SIZE_X, SIZE_Y) and labels[xb, yb] == -1 and homogeneity_criterion(im[x, y], im[xb, yb],
                                                                                               IMG_TYPE):
            labels[xb, yb] = current_label
            queueList.append([xb, yb])
            counter = counter + 1
    return counter


def region_growing(image, img_type):
    print('started region growing')
    # proceeds to region growing on the entire image
    SIZE_X, SIZE_Y = image.shape[0:2]
    # Init of variables
    queue_list = []
    labels = np.ones((SIZE_X, SIZE_Y), int) * (-1)
    current_point_x = -1  # because we will start by incrementing the point x
    current_point_y = 0
    current_label = 0  # note that first label will be 1
    counter = 0

    # Main loop
    while counter != SIZE_X * SIZE_Y:  # while we didn't check all the points on the image
        # Update current point position and check if we reached the end
        if current_point_x < SIZE_X - 1:
            current_point_x = current_point_x + 1
        elif current_point_y < SIZE_Y - 1:
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
    big_label_list = []  # list of labels of large objects
    obj_center = []  # list of center positions of large objects
    for current_label in range(1, nb_labels + 1):
        # find number of pixels in object
        mask_region = cv2.inRange(labeled_pxl, current_label, current_label)
        pxl_coords, area = find_region_coords(mask_region, img.shape[:2])

        if MIN_SIZE_REGION < area < MAX_SIZE_REGION:
            big_label_list.append(current_label)
            obj_center.append(np.mean(pxl_coords, 0).astype(int))
        # nb_pxl = np.count_nonzero(mask_region)

        print('big', current_label)
        # ignore small objects to reduce computation time & ignore noise
        # if MIN_SIZE_REGION < nb_pxl < MAX_SIZE_REGION:
        #     pxl_coords, area = find_region_coords(mask_region, img.shape[:2])
        #     big_label_list.append(current_label)
        #     obj_center.append(np.mean(pxl_coords, 0).astype(int))
    print('finished large regions')
    # merging objects close to each other
    merged_label_list = np.copy(big_label_list)
    # merged_labeled_pxl = np.copy(labeled_pxl)
    merged_labeled_pxl = np.zeros(labeled_pxl.shape[:2])
    # merged_labeled_pxl = np.ones(img.shape[:2])* (-1)

    # go through large objects label list. For objects close to each other, label their corresponding pixels with the same label
    for i in range(len(merged_label_list) - 1):
        for j in range(i + 1, len(merged_label_list)):
            if eucl_distance(obj_center[i], obj_center[j]) < DISTANCE_THRESH:
                print('merge', i)
                merged_labeled_pxl[labeled_pxl == merged_label_list[i]] = merged_label_list[i]
                merged_labeled_pxl[labeled_pxl == merged_label_list[j]] = merged_label_list[i]
                merged_label_list[j] = merged_label_list[i]
    print('finished merging')
    return merged_labeled_pxl, np.unique(merged_label_list)


def apply_growing(img, img_type):
    # UNCOMMENT to compute region growing
    # # do region growing to label each region in final image]
    # # on HSV image
    # [labeled_pxl, nb_labels] = region_growing(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), img_type)
    # # [labeled_pxl, nb_labels] = region_growing(img, img_type)
    # print(nb_labels)
    # np.save(REGION_LABELS_FILE, labeled_pxl)
    # np.save(REGION_NB_LABELS_FILE, nb_labels)
    # # exit_programme()

    # load region growing arrays
    labeled_pxl = np.load(REGION_LABELS_FILE)
    nb_labels = np.load(REGION_NB_LABELS_FILE)
    # cv2.imshow('img', img)
    # key1 = cv2.waitKey(0)

    merged_labeled_pxl, merged_label_list = merge_regions(labeled_pxl, nb_labels, img)
    np.save(REGION_MERGED_LABEL_PXL_FILE, merged_labeled_pxl)
    np.save(REGION_MERGED_LABELS_FILE, merged_label_list)
    # exit_programme()

    # load merged regions
    merged_labeled_pxl = np.load(REGION_MERGED_LABEL_PXL_FILE)
    merged_label_list = np.load(REGION_MERGED_LABELS_FILE)

    # display regions with large area
    for current_label in merged_label_list:
        # choose one region
        print('label', current_label)
        mask_region = cv2.inRange(merged_labeled_pxl, int(current_label), int(current_label))
        print('nb pxls', np.count_nonzero(mask_region))
        region = cv2.bitwise_and(img, img, mask=mask_region)
        cv2.imshow('region', region)
        key1 = cv2.waitKey(0)
        # if press escape key
        if key1 == 27:
            break
        #
        # # cv2.imshow('mask', mask_region)
        # # nb_pxl = sum(sum(mask_region))
        # nb_pxl = np.count_nonzero(mask_region)
        #
        # if MIN_SIZE_REGION < nb_pxl < MAX_SIZE_REGION:
        #     print('label', l, 'nb pxl', nb_pxl)
        #     # create image of region
        #     region = cv2.bitwise_and(img, img, mask=mask_region)
        #     # region = img[mask_region]
        #     cv2.imshow('region', region)
        #     key1 = cv2.waitKey(0)
        #     # if press escape key
        #     if key1 == 27:
        #         break

'''
# Preprocessing steps: examples
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# remove noise
frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)

# Find edges
# convolute with proper kernels
frame_edges = cv2.Canny(frame_blur, 60, 100, 1)
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

# main programme
if MEDIA_TYPE == 'image':
    # load image
    frame = cv2.imread(MEDIA_PATH)
    frame_masked = image_processing(frame)
    # cv2.imshow('masked', cv2.resize(frame_masked, None, fx=0.6, fy=0.6))
    # cv2.waitKey(0)

    # region growing
    apply_growing(frame, IMG_COLOR)

elif MEDIA_TYPE == 'video':
    # Load video:
    cap = cv2.VideoCapture(MEDIA_PATH)
    while cap.isOpened():
        print('Cap Open')
        ret, frame = cap.read()
        if ret:
            frame_masked = image_processing(frame)
            cv2.imshow('masked', cv2.resize(frame_masked, None, fx=0.6, fy=0.6))
            key = cv2.waitKey(0)
            # if press escape key
            if key == 27:
                break
        else:
            break

cv2.destroyAllWindows()
