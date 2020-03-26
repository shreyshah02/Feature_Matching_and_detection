import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

I_left_org = cv.imread('IMG-6986.jpg')
I_st_org = cv.imread('IMG-6987.jpg')

I_left = I_left_org
I_st = I_st_org

gray = cv.cvtColor(I_left, cv.COLOR_BGR2GRAY)
gray_st = cv.cvtColor(I_st, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 1000, 0.1, 50)
corners_st = cv.goodFeaturesToTrack(gray_st, 1000, 0.1, 50)

print(corners_st.shape)

corners = np.int0(corners)
corners_st = np.int0(corners_st)

for i in corners:
    x,y = i.ravel()
    cv.circle(I_left,(x,y), 10,(0, 0, 255), 4)

for i in corners_st:
    x,y = i.ravel()
    cv.circle(I_st, (x,y), 10, (0, 0, 255), 4)

plt.imshow(cv.cvtColor(I_left,cv.COLOR_BGR2RGB)), plt.show()
plt.imshow(cv.cvtColor(I_st,cv.COLOR_BGR2RGB)), plt.show()

cv.imwrite('Features-Shi-left.png', I_left)
cv.imwrite('Features-Shi-St.png', I_st)
# plt.imsave('image',I_left)

diam = 7
nbhd = np.zeros((diam, diam))
nbhd_st = np.zeros((diam,diam))
n_start = np.floor(diam/2)
n_stop = np.floor((diam+1)/2)
xdist = 200
ydist = 70
minsim = 0.7

#corners = corners.reshape(corners.shape[0], (corners.shape[1]*corners.shape[2]))

left_keypoints = []
right_keypoints = []

left_descriptors = []
right_descriptors = []

n = len(corners)
n_st = len(corners_st)

for c in range(0,n):
    x = corners[c,0,0]
    y = corners[c,0,1]
    if x>diam and y >diam:
        left_keypoints.append(cv.KeyPoint(x, y, _size = diam, _response = 0))
        nbhd = I_left[np.int32(y-n_start):np.int32(y+n_stop), np.int32(x-n_start):np.int32(x+n_stop)]
        desc = nbhd.flatten()-np.mean(nbhd)
        left_descriptors.append(desc)

for c in range(0,n_st):
    x = corners_st[c,0,0]
    y = corners_st[c,0,1]
    if x>=diam and y >= diam:
        right_keypoints.append(cv.KeyPoint(x, y, _size = diam, _response = 0))
        nbhd_st = I_st[np.int32(y-n_start):np.int32(y+n_stop), np.int32(x-n_start):np.int32(x+n_stop)]
        desc = nbhd_st.flatten()-np.mean(nbhd_st)
        right_descriptors.append(desc)

right_descriptors = np.array(right_descriptors)
left_descriptors = np.array(left_descriptors)

nlkp = len(left_keypoints)
nrkp = len(right_keypoints)

kp_sim_mat = np.zeros((nlkp, nrkp), dtype=np.float32)

for lkp in range(nlkp):
    lx, ly = np.int32(left_keypoints[lkp].pt)
    left_desc = left_descriptors[lkp]
    left_desc = left_desc[:100]
    left_norm = np.linalg.norm(left_desc)


    for rkp in range(nrkp):
        rx, ry = np.int32(right_keypoints[rkp].pt)
        if np.abs(ry - ly)<ydist and np.abs(rx-lx)<xdist:
            # Can get the dimensions of the smaller and make adjustments accordingly
            right_desc = right_descriptors[rkp]
            right_desc = right_desc[:100]
            right_norm = np.linalg.norm(right_desc)
            sim = np.dot(left_desc, right_desc)/(left_norm*right_norm)
            if sim>minsim:
                kp_sim_mat[lkp,rkp] = sim

lefts_best_right = np.zeros((nlkp, 2))
rights_best_left = np.zeros((nrkp,2))

for lkp in range(nlkp):
    req_row = kp_sim_mat[lkp, :]
    if np.any(req_row):
        ind = np.argmax(req_row)
        lefts_best_right[lkp,0] = ind
        lefts_best_right[lkp, 1] = req_row[ind]

for rkp in range(nrkp):
    req_col = kp_sim_mat[:, rkp]
    if np.any(req_col):
        ind = np.argmax(req_col)
        rights_best_left[rkp, 0] = ind
        rights_best_left[rkp, 1] = req_col[ind]

# Finding consistent matches

matches = []

for i in range(nlkp):
    m = lefts_best_right[i, 0]
    m = np.uint32(m)
    j = np.uint32(rights_best_left[m, 0])
    if j == i:
        matches.append(cv.DMatch(i, j, _distance=lefts_best_right[i,1]))

matches = sorted(matches, key = lambda x:x.distance, reverse=True)

Match_I = cv.drawMatches(I_left, left_keypoints, I_st, right_keypoints, matches[:100], None, matchColor=(0, 255, 0), flags=2)

plt.imshow(cv.cvtColor(Match_I,cv.COLOR_BGR2RGB)), plt.title('Matching'), plt.show()
cv.imwrite('Matching_Statue_Shi_Tomasi.png', Match_I)
