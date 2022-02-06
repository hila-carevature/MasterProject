# Extract & save each frame from video
import cv2
import os

VIDEO_PATH = 'C:/Users/User/Dropbox (Carevature Medical)/Robotic Decompression/Media/210829 animal carevature_CASE0005 Robotic/Keynan/Foraminotomy.mp4'
IMAGE_SAVE_PATH = '../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Keynan/Foraminotomy'


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
