# Roman Danko, 22.4.2017
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
# directory = 'test'
directory = 'test2'
# directory = 'test3'
width = 1280
height = 720
aspect_ratio = (width, height)

# Set up tracker.
# Instead of MIL(4), you can also use
# BOOSTING(1), KCF(2), TLD(2), MEDIANFLOW(1)
trackerType = "MIL"
trackerType = "BOOSTING"
trackerType = "KCF"
trackerType = "TLD"
trackerType = "MEDIANFLOW"

def detect(frame, pout):
    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of orange color in HSV
    # lower_orange = np.array([3, 100, 10]) #-15
    # upper_orange = np.array([33, 255, 255]) #+15
    lower_orange = np.array([3, 120, 10])  # -15
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
    gray = cv2.GaussianBlur(mask, (9, 9), 2, 2)

    # find circles by contours
    # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    img2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    # cv2.imshow("Contours", img2)

    x=y=w=h=0
    found = False
    if contours:
        # print len(contours)

        for cnt in contours:
            M = cv2.moments(cnt)
            # print M

            # if area of detected object is smaller than defined, we asume that it is on edge of screen so we
            # exclude this one
            # print M["m00"]
            if M["m00"] > 10000:
                found = True

                x, y, w, h = cv2.boundingRect(cnt)
                # bbox = (x,y,w,h)
                bbox = (x, y, w, h)
                # print bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # cv2.imshow("init bbox", frame)

                cX = x + w / 2
                cY = y + h / 2
                d = w / 2 if w < h else h / 2

                if pout:
                    a = 0 #placeholder
                    ###print str(video) + ";" + str(count) + ";" + str(cX) + ";" + str(cY) + ";" + str(d)
                # print "found"
                return found, frame, gray, x, y, w, h, cX, cY, M["m00"]

    if pout:
        a = 0 #placeholder
        ###print str(video) + ";" + str(count) + ";?;?;?"

    return found, frame, gray, x, y, w, h, 0, 0, 0

def track(tracker, frame, bbox, type):
    (x, y, w, h) = bbox
    cX = x + w / 2
    cY = y + h / 2
    d = w / 2 if w < h else h / 2

    if tracker:
        # Update tracker
        # print "update"
        ok, bbox = tracker.update(frame)
        (x, y, w, h) = bbox

        # Draw bounding box
        if ok:
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, p1, p2, (0, 0, 255))

            cX = x + w / 2
            cY = y + h / 2
            d = w / 2 if w < h else h / 2
            ### print str(video) + ";" + str(count) + ";" + str(cX) + ";" + str(cY) + ";" + str(d)
            return tracker, frame, cX, cY, d
        else:
            a = 0 #placeholder
            # print str(video) + ";" + str(count) + ";?;?;?"
            #if tracker dont return anything, we should return something meaning not ok
            return tracker, frame, -1, -1, -1
    else:
        # print "init"
        tracker = cv2.Tracker_create(type)

        ok = tracker.init(frame, bbox)
        return tracker, frame, cX, cY, d



video = 0
detections = {}
for filename in os.listdir(directory):

    if filename.endswith(".mp4"):

        path = os.path.join(directory, filename)
        # print(path)

        cap = cv2.VideoCapture(path)

        # define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # only this one currently working under windows

        name = re.sub('\mp4$', '', path)
        out = cv2.VideoWriter('out/' + trackerType + "_" + name + "avi", fourcc, 50.0, aspect_ratio)  # 640,480 and 15fps for my webcam,   1280,720
        #print out.isOpened()

        # Read frames until finds our object
        video += 1
        count = 0
        bbox = False
        images = {}

        found = False
        tracker = False
        #x, y, w, h

        # citame video subor
        while cap.isOpened():

            # snimok po snimku
            ret, frame = cap.read()
            if ret:

                count += 1
                # print "frame no: " + str(count)
                # print "my" + str(found)
                if not found:
                    tracker = False
                    found, oframe, gray, x, y, w, h, cX, cY, d = detect(frame, True)
                    if found:
                        images[count] = [int(cX), int(cY), int(d)]
                    bbox = (x, y, w, h)
                    ###cv2.imwrite("out/" + path + "_frame_%d_mask.jpg" % count, gray)

                if found:
                    tracker, oframe, cX, cY, d = track(tracker, frame, bbox, trackerType)
                    images[count] = [int(cX), int(cY), int(d)]

                    #kazdy 5ty obrazok urob preventivne detekciu
                    if count % 5 == 0:
                        found, oframe, gray, x, y, w, h, cX, cY, d = detect(frame, False)

                # zobrazime aktualny upraveny o detekciu
                cv2.imshow(path, oframe)

                # zapiseme do vystupneho videa
                ###out.write(oframe)

                # zapiseme do vystupneho obrazku
                cv2.imwrite("out/" + path + "_" + trackerType + "_frame_%d.jpg" % count, oframe)

                # spojime svoje data s detekcie do jednej premennej
                detections[video] = images

                # wait for 'q' key to exit program
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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
    # print t

    n = 0
    images = {}
    sorted_t = sorted(t.items(), key=operator.itemgetter(0))
    for k2, i in sorted_t:
        n += 1

        # write detected object trajectory
        if (i[0] in range(0,width)) & (i[1] in range(0,height)):
            coordinates = (i[0], i[1])
            trajectory = cv2.circle(trajectory, coordinates, 2, (0, 0, 255), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(trajectory, str(k1) + ';' + str(k2), coordinates, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if n > 1:
                trajectory = cv2.line(trajectory, coordinates, prevCoordinates, (255,0,0),2)
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
    predictions[k1] = images

print "predicted object coordinates"
print predictions
for k1,t in predictions.iteritems():
    # print t

    n = 0
    images = {}

    sorted_t = sorted(t.items(), key=operator.itemgetter(0))
    for k2, i in sorted_t:
        n += 1

        # write detected object trajectory
        if (i[0] in range(0,width)) & (i[1] in range(0,height)):
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
cv2.imwrite("out/" + directory + "/sum_trajectory_" + trackerType + ".jpg", trajectory)

while True:
    # wait for 'q' key to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break