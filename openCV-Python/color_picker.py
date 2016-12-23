import numpy as np
import cv2

orange = np.uint8([[[0,153,255 ]]])
hsv_orange = cv2.cvtColor(orange,cv2.COLOR_BGR2HSV)
print hsv_orange