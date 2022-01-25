# Processing of the image/video for extracting dura&bone
import cv2
import numpy as np


# MEDIA_TYPE = 'image'        # 'image' or 'video'
# MEDIA_PATH ='../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short/frame1.jpg'
MEDIA_TYPE = 'video'        # 'image' or 'video'
MEDIA_PATH ='C:/Users/User/Dropbox (Carevature Medical)/Robotic Decompression/Media/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short.mp4'
# Filters in HSV
BONE_LOWER_RANGE = np.array([0, 0, 163])            #np.array([0, 0, 0])
BONE_UPPER_RANGE = np.array([179, 255, 255])        #np.array([179, 255, 175])
DURA_LOWER_RANGE = np.array([126, 0, 0])            #np.array([0, 39, 0])
DURA_UPPER_RANGE = np.array([166, 255, 255])        #np.array([169, 255, 149])
KERNEL_MORPH = np.ones((21, 21), np.uint8)            # kernel for morphology close & open


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

    return bone_n_dura


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
    cv2.imshow('masked', cv2.resize(frame_masked, None, fx=0.6, fy=0.6))
    cv2.waitKey(0)
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
