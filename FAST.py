import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

I_left_org = cv.imread('IMG-6986.jpg')
I_st_org = cv.imread('IMG-6987.jpg')

I_left = I_left_org
I_st = I_st_org

fast = cv.FastFeatureDetector_create()
fast.setThreshold(80)

kp = fast.detect(I_left, None)
kp_st = fast.detect(I_st, None)

I_left = cv.drawKeypoints(I_left, kp, None, (0, 0, 255), 4)
I_st = cv.drawKeypoints(I_st, kp_st, None, (0, 0, 255), 4)

# FAST is a keypoint detector and hence does not compute descriptors
# Descriptors need to be computed by some other function
# We will try Brief here

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp, des = brief.compute(I_left, kp)
kp_st, des_st = brief.compute(I_st, kp_st)

print('Threshold :{}'.format(fast.getThreshold()))
print('Total Keypoints left : {}'.format(len(kp)))
print('Neighborhood: {}'.format(fast.getType()))
print('Total Keypoints st : {}'.format(len(kp_st)))
print('Non Maximal suppression: {}'.format(fast.getNonmaxSuppression()))

plt.imshow(cv.cvtColor(I_left, cv.COLOR_BGR2RGB)), plt.title('Threshold = {}'.format(fast.getThreshold())), plt.show()
plt.imshow(cv.cvtColor(I_st, cv.COLOR_BGR2RGB)), plt.title('St_Image, Threshold= {}'.format(fast.getThreshold())), plt.show()

cv.imwrite('Features_FAST_statue_left.png', I_left)
cv.imwrite('Features_FAST_statue_st.png', I_st)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
matches = bf.match(des, des_st)

matches = sorted(matches, key = lambda x:x.distance)

Match_I = cv.drawMatches(I_left_org, kp, I_st_org, kp_st, matches[:100], None, matchColor=(0, 255, 0), flags=2)

plt.imshow(cv.cvtColor(Match_I,cv.COLOR_BGR2RGB)), plt.title('Matching'), plt.show()
cv.imwrite('Matching_Statue_FAST.png', Match_I)

# fast.setNonmaxSuppression(0)
# kp = fast.detect(I_left, None)
# I_left = cv.drawKeypoints(I_left, kp, None, (0, 0, 255), 4)
#
# plt.imshow(cv.cvtColor(I_left, cv.COLOR_BGR2RGB)), plt.show()
