import numpy as np
import cv2

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,600,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('Range','image',0,30,nothing)
cv2.createTrackbar('Sat min','image',0,255,nothing)
cv2.createTrackbar('Sat max','image',0,255,nothing)
cv2.createTrackbar('Value min','image',0,255,nothing)
cv2.createTrackbar('Value max','image',0,255,nothing)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    b = cv2.getTrackbarPos('B','image')
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    ran = cv2.getTrackbarPos('Range', 'image')
    sMin = cv2.getTrackbarPos('Sat min','image')
    sMax = cv2.getTrackbarPos('Sat max','image')
    vMin = cv2.getTrackbarPos('Value min','image')
    vMax = cv2.getTrackbarPos('Value max','image')


    img[:] = [b,g,r]

    orange = np.uint8([[[b, g, r ]]])
    hsv_orange = cv2.cvtColor(orange,cv2.COLOR_BGR2HSV)
    print hsv_orange

    hue = hsv_orange[0][0][0]

    # define range of orange color in HSV
    sat_int =  [sMin, sMax]
    value_int = [vMin, vMax]
    range = np.array([ran, 0, 0])

    lower_orange = np.array([hue, sat_int[0], value_int[0]]) - range  # -15
    upper_orange = np.array([hue, sat_int[1], value_int[1]]) + range  # +15

    print lower_orange
    print upper_orange


    # frame = cv2.imread("out/frames/frame15.jpg")
    # frame = cv2.imread("out/frames/raw/MVI_1157.MOV_frame2746.jpg")
    # frame = cv2.imread("out/frames/sikmy/sikmy_01.mp4_frame27.jpg")
    frame = cv2.imread("out/substraction/img_substracted.jpg")
    cv2.imshow('frame', frame)

    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # display mask and masked source image
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    # for better detection, ew need apply some blur (the best permformance provides for me gaussian)
    # without blur, multiple countours detected, for exapmle pointd from noise
    gray = cv2.GaussianBlur(mask, (25, 25), 2, 2)

    # display grayscale image before detection of circles
    cv2.imshow('gray', gray)

    # orange = np.uint8([[[0,153,255 ]]])
    # hsv_orange = cv2.cvtColor(orange,cv2.COLOR_BGR2HSV)
    # print hsv_orange

    # wait for 's' key to save current images
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("out/color_thresholding/mask.jpg", mask)
        cv2.imwrite("out/color_thresholding/res.jpg", res)
        cv2.imwrite("out/color_thresholding/gray.jpg", gray)


while True:
    # wait for 'q' key to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break