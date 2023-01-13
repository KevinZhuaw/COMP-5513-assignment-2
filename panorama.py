
import cv2
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, draw_match, transform
from utils import read_video_frames, write_and_show, destroyAllWindows, imshow

frms, fps = utils.read_video_frames('image/wall_paint.mov')
f1rm0 = frms[0].tolist()
nx0, ny0 = len(f1rm0[0]), len(f1rm0)

nx02, ny02 = int(nx0/2), int(ny0/2)

ny_pano, nx_pano = ny0*2, nx0*8
s2tack = np.zeros((ny_pano, nx_pano, 3), np.uint8)
s2tack[ny02:ny02+ny0, :nx0, :] = frms[0]

l2ast = s2tack
for k,c2ur in enumerate(frms[1:]):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(l2ast, mask=None)
    keypoints2, descriptors2 = sift.detectAndCompute(c2ur, mask=None)
    matcher = cv2.BFMatcher_create(crossCheck=True)
    match = matcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
    matcher = cv2.FlannBasedMatcher_create()
    match = matcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
    best_2 = matcher.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
    ratio = 0.7
    match = []
    for m, n in best_2:
        if m.distance < ratio * n.distance:
            match.append(m)
    match = sorted(match, key=lambda x: x.distance)
    match = match[:36]
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])
    src, dst = c2ur, l2ast
    src_kps, dst_kps = (keypoints2, keypoints1)
    T, status = cv2.findHomography(
        srcPoints=src_kps,
        dstPoints=dst_kps,
        method=cv2.USAC_ACCURATE,
        ransacReprojThreshold=3)

    new_img2 = cv2.warpPerspective(
        src=c2ur,
        M=T,
        dsize=(nx_pano, ny_pano))
    cnt = np.zeros([ny_pano, nx_pano, 1]) + 1e-10
    cnt += (new_img2 != 0).any(2, keepdims=True)
    cnt += (s2tack != 0).any(2, keepdims=True)
    s2tack = np.float32(s2tack)
    new_img2 = np.float32(new_img2)
    s2tack = np.round((new_img2 + s2tack) / cnt).astype(np.uint8)
    l2ast = s2tack

cv2.imwrite(f'./panorama{k+1}.jpg', s2tack)