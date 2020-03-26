import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

I_left_org = cv.imread('IMG-6986.jpg')
I_st_org = cv.imread('IMG-6987.jpg')

I_st = I_st_org
I_left = I_left_org

orb = cv.ORB_create()
orb.setMaxFeatures(15000)

kp = orb.detect(I_left, None)
kp_st = orb.detect(I_st, None)

kp, des = orb.compute(I_left, kp)
kp_st, des_st = orb.compute(I_st, kp_st)

I_left = cv.drawKeypoints(I_left, kp, None, (0, 0, 255), 0)
I_st = cv.drawKeypoints(I_st, kp_st, None, (0, 0, 255), 0)
print('No. of Keypoints in Left: {}'.format(len(kp)))
print('No. of Keypoints in st: {}'.format(len(kp_st)))

plt.imshow(cv.cvtColor(I_left, cv.COLOR_BGR2RGB)), plt.title('Orb_Features with draw flag = 0, Left Image'), plt.show()
plt.imshow(cv.cvtColor(I_st, cv.COLOR_BGR2RGB)), plt.title('Orb_features with draw flag = 0, St Image'), plt.show()

cv.imwrite('Features_Orb_Statue_Left.png', I_left)
cv.imwrite('Features_Orb_Statue_St.png', I_st)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
matches = bf.match(des, des_st)

matches = sorted(matches, key = lambda x:x.distance)

Match_I = cv.drawMatches(I_left_org, kp, I_st_org, kp_st, matches[:100], None, matchColor=(0, 255, 0), flags=2)

plt.imshow(cv.cvtColor(Match_I,cv.COLOR_BGR2RGB)), plt.title('Matching'), plt.show()
cv.imwrite('Matching_Statue_ORB.png', Match_I)