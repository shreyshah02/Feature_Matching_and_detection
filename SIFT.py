import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

I_left_org = cv.imread('IMG-6986.jpg')
I_left = I_left_org
gray = cv.cvtColor(I_left,cv.COLOR_BGR2GRAY)

I_st_org = cv.imread('IMG-6987.jpg')

I_st = I_st_org
gray_st = cv.cvtColor(I_st,cv.COLOR_BGR2GRAY)

#sift = cv.xfeatures2d.SIFT_create()
sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.16, sigma= 1.2)
kp = sift.detect(gray, None)
kp_st = sift.detect(gray_st, None)

I_left = cv.drawKeypoints(gray, kp, I_left)
I_st = cv.drawKeypoints(gray_st, kp_st, I_st)

kp, des = sift.compute(gray, kp)
kp_st, des_st = sift.compute(gray_st, kp_st)

I_left = cv.drawKeypoints(gray, kp, I_left, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
I_st = cv.drawKeypoints(gray_st, kp_st, I_st, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#plt.imshow(cv.cvtColor(I_left,cv.COLOR_BGR2RGB)), plt.show()
#plt.imshow(cv.cvtColor(I_st,cv.COLOR_BGR2RGB)), plt.show()

#plt.imshow(I_left), plt.show()
#plt.imshow(I_st), plt.show()

cv.imwrite('Features_sift_statue_left.png', I_left)
cv.imwrite('Features_sift_statue_st.png', I_st)

bf = cv.BFMatcher()
matches = bf.match(des, des_st)

matches = sorted(matches, key=lambda x:x.distance)

I_match = cv.drawMatches(I_left_org, kp, I_st_org, kp_st, matches[:100], None, flags =2)
plt.imshow(cv.cvtColor(I_match,cv.COLOR_BGR2RGB)), plt.title('Matching'), plt.show()

cv.imwrite('Matching_SIFT_Statue.png', I_match)

