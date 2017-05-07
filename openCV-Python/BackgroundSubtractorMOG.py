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

fgbg = cv2.createBackgroundSubtractorMOG2()

while (True):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    cv2.imwrite("MOG/frame%d.jpg" % count, fgmask)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    count += 1

cap.release()
cv2.destroyAllWindows()