import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

I_left_org = cv.imread('IMG-6986.jpg')
I_st_org = cv.imread('IMG-6987.jpg')

I_left = I_left_org
I_st = I_st_org

star = cv.xfeatures2d.StarDetector_create()

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

kp = star.detect(I_left, None)
kp_st = star.detect(I_st, None)

I_left = cv.drawKeypoints(I_left, kp, None, (0, 0, 255), 4)
I_st = cv.drawKeypoints(I_st, kp_st, None, (0, 0, 255), 4)

kp, des = brief.compute(I_left, kp)
kp_st, des_st = brief.compute(I_st, kp_st)

print('Descriptor Size from Brief: {}'.format(brief.descriptorSize()))
print('Descriptor shape for left Image: {}'.format(des.shape))
print('Descriptor shape for st Image: {}'.format(des_st.shape))

plt.imshow(cv.cvtColor(I_left, cv.COLOR_BGR2RGB)), plt.title('Left Image with KeyPoints'), plt.show()
plt.imshow(cv.cvtColor(I_st, cv.COLOR_BGR2RGB)), plt.title('St Image with KeyPoints'), plt.show()

cv.imwrite('Features_Star_Brief_left_Statue.png', I_left)
cv.imwrite('Features_Star_Brief_st_Statue.png', I_st)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des, des_st)

matches = sorted(matches, key=lambda x:x.distance)

Match_I = cv.drawMatches(I_left_org, kp, I_st_org, kp_st, matches[:100], None, matchColor=(0, 255, 0), flags=2)

plt.imshow(cv.cvtColor(Match_I,cv.COLOR_BGR2RGB)), plt.title('Matching'), plt.show()
cv.imwrite('Matching_Statue_Brief_Star.png', Match_I)

