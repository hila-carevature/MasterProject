# Interactive tool to adjust HSV filters on a video/image
# https://medium.com/programming-fever/how-to-find-hsv-range-of-an-object-for-computer-vision-applications-254a8eb039fc

# finding hsv range of target object(pen)
import cv2
import numpy as np


VIDEO = 0
IMAGE = 1
INPUT_TYPE = IMAGE
# IMAGE_PATH ='../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram/Foraminotomy - short/frame975.jpg'
IMAGE_PATH ='../res/surgery_images/210829 animal carevature_CASE0005 Robotic/Millgram/swab_wonder.PNG'
HSV = 0
RGB = 1
FILTER_TYPE = HSV


# A required callback method that goes into the trackbar function.
def nothing(x):
    pass


# main:
cap = 0
frame = 0
if INPUT_TYPE == VIDEO:
    # Initializing the webcam feed.
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
elif INPUT_TYPE == IMAGE:
    frame = cv2.imread(IMAGE_PATH)
    # resize image
    resize_scale = 0.6  # percent of original size
    width = int(frame.shape[1] * resize_scale)
    height = int(frame.shape[0] * resize_scale)
    dim = (width, height)
    frame = cv2.resize(frame, dim)
else:
    print('input media type unexpected')
    exit()
# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of
# H,S and V channels. The Arguments are like this: Name of trackbar,
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
if FILTER_TYPE == HSV:
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
else:
    cv2.createTrackbar("L - R", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - G", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - B", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - R", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - G", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - B", "Trackbars", 255, 255, nothing)


while True:

    if INPUT_TYPE == VIDEO and cap != 0:
        # Start reading the webcam feed frame by frame.
        ret, frame = cap.read()
        if not ret:
            break
        # Flip the frame horizontally (since webcam is flipped)
        frame = cv2.flip(frame, 1)

    original_frame = np.copy(frame)

    hsv_frame = np.copy(frame)
    rgb_frame = np.copy(frame)
    if FILTER_TYPE == HSV:
        # Convert the BGR image to HSV image.
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get the new values of the trackbar in real time as the user changes
        # them
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        # Set the lower and upper HSV range according to the value selected
        # by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        l_r = cv2.getTrackbarPos("L - R", "Trackbars")
        l_g = cv2.getTrackbarPos("L - G", "Trackbars")
        l_b = cv2.getTrackbarPos("L - B", "Trackbars")
        u_r = cv2.getTrackbarPos("U - R", "Trackbars")
        u_g = cv2.getTrackbarPos("U - G", "Trackbars")
        u_b = cv2.getTrackbarPos("U - B", "Trackbars")

        # Set the lower and upper HSV range according to the value selected
        # by the trackbar BGR
        lower_range = np.array([l_r, l_g, l_b])
        upper_range = np.array([u_r, u_g, u_b])

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsv_frame if FILTER_TYPE == HSV else rgb_frame, lower_range, upper_range)

    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Converting the binary mask to 3 channel image, this is just so
    # we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask_3 = cv2.cvtColor(cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR), cv2.COLOR_GRAY2BGR)

    # stack the mask, original frame and the filtered result
    stacked = np.hstack((mask_3, original_frame, res))

    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

    # If the user presses `s` then print this array.
    if key == ord('s'):
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]] if FILTER_TYPE == HSV else [[l_r, l_g, l_b], [u_r, u_g, u_b]]
        print(thearray)

        # # Also save this array as penval.npy
        # np.save('filter_value', thearray)
        # break

if INPUT_TYPE == VIDEO:
    # Release the camera & destroy the windows.
    cap.release()
cv2.destroyAllWindows()
