import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.medianBlur(frame,5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 85, param1=100, param2=60, minRadius=0, maxRadius=0)

    # try:
    #     circles = np.uint16(np.around(circles))
    #
    #     for i in circles[0, :]:
    #         # draw the outer circle
    #         cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # draw the center of the circle
    #         cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    # except:
    #     print "blabla"


    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture.py
cap.release()
cv2.destroyAllWindows()