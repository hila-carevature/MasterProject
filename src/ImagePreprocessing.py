# Processing of the image/video for extracting dura&bone
import cv2
import numpy as np

# load image
IMAGE_PATH ='../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short/frame1.jpg'
original_frame = cv2.imread(IMAGE_PATH)
# Resize image
frame = cv2.resize(original_frame, (960, 540))

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
cv2.destroyAllWindows()