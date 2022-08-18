import numpy as np
import cv2

img = cv2.imread('tetris_blocks.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=100)

# find the keypoints with ORB
kps = orb.detect(img_gray, None)

# compute the descriptors with ORB
kps, des = orb.compute(img_gray, kps)

for kp in kps:
    print(kp.response)
    r = int(0.1 * kp.size)
    (x, y) = np.int0(kp.pt)
    cv2.circle(img, (x, y), r, (0, 0, 0), 1)

# display the image
cv2.imshow("Keypoints", img)
cv2.waitKey(0)