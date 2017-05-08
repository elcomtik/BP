# Roman Danko, 20.4.2017
#
# Object tracking(orange ball) implementation based on two principles combined together. Fist is used colour filtering
# of tracked object. Second is Hough circle detection, found circles are written to image.

import os
import re
import operator
import cv2
import numpy as np

directory = 'datasety/'
dataset = 'sikmy'
# directory = 'priamy'
# directory = 'test'
# directory = 'test4'
width = 1280
height = 720
aspect_ratio = (width, height)


video = 0
detections = {}
for filename in os.listdir(directory+dataset):

    if filename.endswith(".mp4"):

        path = os.path.join(directory+dataset, filename)
        #print(path)

        cap = cv2.VideoCapture(path)

        # define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # only this one currently working under windows

        name = re.sub('\mp4$', '', path)
        out = cv2.VideoWriter('out/' + name + "avi", fourcc, 50.0, aspect_ratio)  # 640,480 and 15fps for my webcam,   1280,720

        #print out.isOpened()

        video += 1
        #print video

        count = 0

        images = {}
        while cap.isOpened():

            # take each frame
            ret, frame = cap.read()
            if ret:
                count += 1
                # print "frame no: " + str(count)
                # convert BGR to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # define range of orange color in HSV
                # lower_orange = np.array([3, 100, 10]) #-15
                # lower_orange = np.array([3, 120, 10])  # -15
                # lower_orange = np.array([3, 130, 150])  # -15
                lower_orange = np.array([3, 80, 40])  # -15
                upper_orange = np.array([33, 255, 255])  # +15

                # threshold the HSV image to get only blue colors
                mask = cv2.inRange(hsv, lower_orange, upper_orange)

                # bitwise-AND mask and original image
                res = cv2.bitwise_and(frame, frame, mask=mask)

                # display mask and masked source image
                # cv2.imshow('mask', mask)
                # cv2.imshow('res', res)

                # for better detection, ew need apply some blur (the best permformance provides for me gaussian)
                # without blur, multiple countours detected, for exapmle pointd from noise
                gray = cv2.GaussianBlur(mask, (25, 25), 2, 2)

                # display grayscale image before detection of circles
                # cv2.imshow('gray', gray)

                # find circles by contours
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=45, param2=18, minRadius=0, maxRadius=0)

                try:
                    circles = np.uint16(np.around(circles))
                    found = False
                    scX = 0
                    scY = 0
                    c = 0
                    for i in circles[0, :]:

                        # if diameter detected object is smaller than defined, we asume that it is false detection
                        # exclude this one
                        if i[2] > 50:

                            found = True
                            c += 1

                            cX = i[0]
                            cY = i[1]
                            scX += cX
                            scY += cY

                    if found:
                        pcX = scX/c
                        pcY = scY/c

                        center = (int(pcX), int(pcY))
                        print str(video) + ";" + str(count) + ";" + str(pcX) + ";" + str(pcY) + ";" + str(i[2])
                        images[count] = [pcX, pcY, i[2]]

                        # draw the outer circle
                        cv2.circle(frame, center, 65, (0, 255, 0), 2)
                        # draw the center of the circle
                        cv2.circle(frame, center, 2, (0, 0, 255), 3)

                    if not found:
                        print str(video) + ";" + str(count) + ";?;?;?"

                except:
                    # print "nothing_here"
                    print str(video) + ";" + str(count) + ";?;?;?"

                # display current frame
                cv2.imshow(path, frame)

                # write frame
                out.write(frame)

                # write frame to jpg
                cv2.imwrite("out/" + path + "_frame_%d.jpg" % count, frame)
                cv2.imwrite("out/" + path + "_frame_%d_mask.jpg" % count, gray)

                #assemble our data into one
                detections[video] = images

                # wait for 'q' key to exit program
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # if video source stops than exit loop
            else:
                break

        # close properly resources
        cap.release()
        out.release()
        # out2.release()
        cv2.destroyAllWindows()
    else:
        continue

print "detected object coordinates"
print detections

trajectories = np.ones((height, width, 3), np.uint8) * 255

predictions = {}
for k1,t in detections.iteritems():
    # print t

    curTrajectory = np.ones((height, width, 3), np.uint8) * 255

    n = 0
    images = {}
    sorted_t = sorted(t.items(), key=operator.itemgetter(0))
    for k2, i in sorted_t:
        n += 1

        # write detected object trajectory
        if (i[0] in range(0,width)) & (i[1] in range(0,height)):
            coordinates = (i[0], i[1])
            trajectories = cv2.circle(trajectories, coordinates, 2, (0, 0, 255), 3)
            curTrajectory = cv2.circle(curTrajectory, coordinates, 2, (0, 0, 255), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(trajectories, str(k1) + ';' + str(k2), coordinates, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(curTrajectory, str(k1) + ';' + str(k2), coordinates, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if n > 1:
                trajectories = cv2.line(trajectories, coordinates, prevCoordinates, (255, 0, 0), 2)
                curTrajectory = cv2.line(curTrajectory, coordinates, prevCoordinates, (255, 0, 0), 2)
                # print k2

            #let's do some prediction
            if n > 1:
                pX = 2*coordinates[0] - prevCoordinates[0]
                pY = 2*coordinates[1] - prevCoordinates[1]
                predictedCoordinates = (pX, pY)
                images[k2+1] = predictedCoordinates

            #useful for computing prediction and drawing trajectory
            prevCoordinates = coordinates
    #break;
    cv2.imwrite("out/" + dataset + "/" + str(k1) + "_trajectory.jpg", curTrajectory)
    predictions[k1] = images

print "predicted object coordinates"
print predictions
for k1,t in predictions.iteritems():
    # print t

    curTrajectory = cv2.imread("out/" + dataset + "/" + str(k1) + "_trajectory.jpg")

    n = 0
    images = {}

    sorted_t = sorted(t.items(), key=operator.itemgetter(0))
    for k2, i in sorted_t:
        n += 1

        # write detected object trajectory
        if (i[0] in range(0,width)) & (i[1] in range(0,height)):
            coordinates = (i[0], i[1])
            trajectories = cv2.circle(trajectories, coordinates, 2, (0, 0, 0), 3)
            curTrajectory = cv2.circle(curTrajectory, coordinates, 2, (0, 0, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(trajectories, str(k1) + ';' + str(k2), coordinates, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(curTrajectory, str(k1) + ';' + str(k2), coordinates, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if n > 1:
                trajectories = cv2.line(trajectories, coordinates, prevCoordinates, (0, 255, 0), 2)
                curTrajectory = cv2.line(curTrajectory, coordinates, prevCoordinates, (0,255,0),2)

            # useful for computing prediction and drawing trajectory
            prevCoordinates = coordinates

    cv2.imwrite("out/" + dataset + "/" + str(k1) + "_prediction_trajectory.jpg", curTrajectory)

# Let's show our results
cv2.imshow("trajectory", trajectories)
cv2.imwrite("out/" + dataset + "/sum_trajectory.jpg", trajectories)

while True:
    # wait for 'q' key to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break