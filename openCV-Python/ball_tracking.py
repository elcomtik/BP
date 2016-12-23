# Roman Danko, 23.12.2016
#
# Object tracking(orange ball) implementation based on two principles combined together. Fist is used colour filtering
# of tracked object. Later is used Hough circle detection, found circles are written to image.

import cv2
import numpy as np

# open video capture
cap = cv2.VideoCapture(0)

# define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  #only this one currently working under windows
out = cv2.VideoWriter('output.avi',fourcc, 15.0, (640,480))  #640,480 and 15fps for my webcam,   1280,720
print out.isOpened()

while cap.isOpened():

    # take each frame
    ret, frame = cap.read()
    if ret:

        # convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of orange color in HSV
        lower_orange = np.array([3, 100, 10]) #-15
        upper_orange = np.array([33, 255, 255]) #+15

        # threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        # display mask and masked source image
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)

        # we need grayscale for hugh circle algorithm, so we transform it
        gray = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # for better detection, ew need apply some blur (the best permformance provides for me gaussian)
        gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)

        # display grayscale image before detection of circles
        cv2.imshow('gray', gray)

        # find circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

        # lets display result of deection on source image
        try:
            circles = np.uint16(np.around(circles))
            print "found ball"
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        except:
            print "nothing here"

        # display current frame
        cv2.imshow('frame',frame)

        # write frame
        out.write(frame)

        #wait for 'q' key to exit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #if video source stops than exit loop
    else:
        break

#close properly resources
cap.release()
out.release()
cv2.destroyAllWindows()