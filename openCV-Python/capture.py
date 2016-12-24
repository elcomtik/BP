import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# print "width: " + str(cap.get(3)) + ", height: " + str(cap.get(4)) + ", fps: " + str(cap.get(5))
# for i in range(1, 18):
#     print str(cap.get(i))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  #only this one currently working under windows
out = cv2.VideoWriter('output.avi',fourcc, 15.0, (640,480))  #640,480 and 15fps for my webcam,   1280,720
print out.isOpened()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # write frame
        out.write(frame)

        #show frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()