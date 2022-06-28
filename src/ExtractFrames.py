# Extract & save each frame from video
import cv2
import os

VIDEO_PATH = '../res/monochromatic_illumination/2022-06-15 Experiment 4/white.mkv'
IMAGE_SAVE_PATH = '../res/monochromatic_illumination/2022-06-15 Experiment 4/430'


# Load video:
cap = cv2.VideoCapture(VIDEO_PATH)
# create directory for saving images: if it doesn't already exists
try:
    os.makedirs(IMAGE_SAVE_PATH)
except OSError:
    pass


while cap.isOpened():
    print('Cap Open')
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(IMAGE_SAVE_PATH, 'frame%d.jpg' % cap.get(cv2.CAP_PROP_POS_FRAMES)), frame)
    else:
        break

cv2.destroyAllWindows()
exit()
