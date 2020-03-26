import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

I_left_org = cv.imread('IMG-6986.jpg')
I_st_org = cv.imread('IMg-6987.jpg')

I_left = I_left_org
I_st = I_st_org

surf = cv.xfeatures2d.SURF_create()
surf.setHessianThreshold(5000)

print('Hessian threshold: {}'.format(surf.getHessianThreshold()))

kp, des = surf.detectAndCompute(I_left, None)
kp_st, des_st = surf.detectAndCompute(I_st, None)

I_left = cv.drawKeypoints(I_left, kp, None, (0, 0, 255), 4)
I_st = cv.drawKeypoints(I_st, kp_st, None, (0, 0, 255), 4)

#plt.imshow(I_left), plt.show()
#plt.imshow(I_st), plt.show()

plt.imshow(cv.cvtColor(I_left,cv.COLOR_BGR2RGB)), plt.show()
plt.imshow(cv.cvtColor(I_st,cv.COLOR_BGR2RGB)), plt.show()

cv.imwrite('Features_SURF_statue_Left.png', I_left)
cv.imwrite('Features_SURF_statue_st.png', I_st)

bf = cv.BFMatcher()
matches = bf.match(des, des_st)

matches = sorted(matches, key=lambda x:x.distance)

I_match = cv.drawMatches(I_left_org, kp, I_st_org, kp_st, matches[:100], None, flags =2)
plt.imshow(cv.cvtColor(I_match,cv.COLOR_BGR2RGB)),plt.title('Matching'), plt.show()
cv.imwrite('Matching_statue_SURF.png', I_match)