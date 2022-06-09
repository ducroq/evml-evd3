import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("hand.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest
# one
cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
contour = max(cnts, key=cv2.contourArea)

# get characteristic contour length
area = cv2.contourArea(contour)
equi_diameter = np.sqrt(4*area/np.pi)

# find convex hull and convexity defects
hull = cv2.convexHull(contour, returnPoints=False)
defects = cv2.convexityDefects(contour, hull)

# filter points with small distance between contour and hull
defects = defects.squeeze()
defects = defects[defects[:,-1] > equi_diameter]

# the histogram of the data
counts, bins = np.histogram(defects[:,-1])


# draw the outline of the object
cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)
for s,e,f,d in defects:
    start = tuple(contour[s][0])
    end = tuple(contour[e][0])
    far = tuple(contour[f][0])
    cv2.line(image,start,end,[0,255,0],2)
    cv2.circle(image,far,5,[0,0,255],-1)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

# plot a feature histogram
fig, ax = plt.subplots()
ax.hist(bins[:-1], bins, weights=counts)
ax.set_xlabel('Max contour to hull distance')
ax.set_ylabel('Count')
ax.set_title(r'Histogram of convexity defect distances')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

fig.savefig('histogram.jpg')
