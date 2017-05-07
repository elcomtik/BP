# Roman Danko, 20.4.2017
#
# Object tracking(orange ball) implementation based on two principles combined together. Fist is used colour filtering
# of tracked object. Second is one named Moments, whic  found circles are written to image.

import os
import re
import operator
import cv2
import numpy as np

# directory = 'sikmy'
# directory = 'priamy'
directory = 'test'
width = 1280
height = 720
aspect_ratio = (width, height)

#select which trajectory calculate
# s = 3

video = 0
detections = {}
for filename in os.listdir(directory):


    if filename.endswith(".mp4"):

        path = os.path.join(directory, filename)
        #print(path)

        cap = cv2.VideoCapture(path)

        # define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # only this one currently working under windows

        name = re.sub('\mp4$', '', path)
        out = cv2.VideoWriter('out/' + name + "avi", fourcc, 50.0, aspect_ratio)  # 640,480 and 15fps for my webcam,   1280,720

        #print out.isOpened()

        video += 1
        #print video

        # if video != s:
        #     continue

        count = 0

        images = {}
        while cap.isOpened():

            # take each frame
            ret, frame = cap.read()
            if ret:

                # selected = False
                # while not selected:


                # Select ROI
                r = cv2.selectROI(path,frame)

                cX = r[0]+r[2]/2
                cY = r[1]+r[3]/2
                d = r[1]/2 if r[1]<r[3] else r[3]/2

                print str(video) + ";" + str(count) + ";" + str(cX) + ";" + str(cY) + ";" + str(d)

                count += 1

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

trajectory = np.zeros((height,width,3), np.uint8)

predictions = {}
for k1,t in detections.iteritems():
    print t

    n = 0
    images = {}
    sorted_t = sorted(t.items(), key=operator.itemgetter(0))
    for k2, i in sorted_t:
        n += 1

        # write detected object trajectory
        if (i[0] < width) & (i[1] < height):
            coordinates = (i[0], i[1])
            trajectory = cv2.circle(trajectory, coordinates, 2, (0, 0, 255), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(trajectory, str(k1) + ';' + str(k2), coordinates, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if n > 1:
                trajectory = cv2.line(trajectory, coordinates, prevCoordinates, (255,0,0),2)
                print k2

            #let's do some prediction
            if n > 1:
                pX = 2*coordinates[0] - prevCoordinates[0]
                pY = 2*coordinates[1] - prevCoordinates[1]
                predictedCoordinates = (pX, pY)
                images[k2+1] = predictedCoordinates

            #useful for computing prediction and drawing trajectory
            prevCoordinates = coordinates
    #break;
    predictions[k1] = images

print "predicted object coordinates"
print predictions
for k1,t in predictions.iteritems():
    print t

    n = 0
    images = {}

    sorted_t = sorted(t.items(), key=operator.itemgetter(0))
    for k2, i in sorted_t:
        n += 1

        # write detected object trajectory
        if (i[0] < width) & (i[1] < height):
            coordinates = (i[0], i[1])
            trajectory = cv2.circle(trajectory, coordinates, 2, (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(trajectory, str(k1) + ';' + str(k2), coordinates, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if n > 1:
                trajectory = cv2.line(trajectory, coordinates, prevCoordinates, (0,255,0),2)

            # useful for computing prediction and drawing trajectory
            prevCoordinates = coordinates

# Let's show our results
cv2.imshow("trajectory", trajectory)
cv2.imwrite("out/" + directory + "/sum_trajectory.jpg", trajectory)

while True:
    # wait for 'q' key to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break