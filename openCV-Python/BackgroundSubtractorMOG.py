# Roman Danko, 25.1.2017
#
# BackgroundSubstractorMOG dynamic background model from video
# http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html

import cv2
import numpy as np

count = 0

#cap = cv2.VideoCapture('vtest.avi')
cap = cv2.VideoCapture('raw/MVI_1156.MOV')
print cap

# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)         #OpenCV MOG2, shodow detection off
fgbg = cv2.createBackgroundSubtractorMOG2()              #OpenCV MOG2, shodow detection on

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        fgmask = fgbg.apply(frame)
        # cv2.imwrite("out/MOG/off/MVI_1156.MOV_frame%d.jpg" % count, fgmask)
        cv2.imwrite("out/MOG/on/MVI_1156.MOV_frame%d.jpg" % count, fgmask)

        cv2.imshow('frame', fgmask)
        # wait for 'q' key to exit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1

    # if video source stops than exit loop
    else:
        break

cap.release()
cv2.destroyAllWindows()