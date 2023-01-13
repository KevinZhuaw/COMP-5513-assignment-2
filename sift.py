import cv2
import numpy as np
from utils import imread, imshow, write_and_show, destroyAllWindows

def keypoint_match(img1, img2, max_n_match=100, draw=True):
    # make sure they are of dtype uint8
    img1 = cv2.imread('image/left2.jpg')
    img2 = cv2.imread('image/right2.jpg')
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: convert to grayscale by `cv2.cvtColor`
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img1_gray',img1_gray)
    cv2.imshow('img2_gray', img2_gray)
    cv2.imwrite('img1_gray.jpg',img1_gray)
    cv2.imwrite('img2_gray.jpg', img2_gray)
    # TODO: detect keypoints and generate descriptor by `sift.detectAndCompute`
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1,mask = None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2,mask = None)


    # draw keypoints
    if draw:
        # TODO: draw keypoints on image1 and image2 by `cv2.drawKeypoints`
        img1_keypoints = cv2.drawKeypoints(
            image = img1,
            keypoints = keypoints1,
            outImage = None,
            flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite('img1_keypoints.jpg',img1_keypoints)
        img2_keypoints = cv2.drawKeypoints(
            image=img2,
            keypoints = keypoints2,
            outImage = None,
            flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite('img2_keypoints.jpg', img2_keypoints)
        pass


    # TODO: Knn match and Lowe's ratio test

    matcher = cv2.BFMatcher_create(crossCheck = True)
    match = matcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
    matcher = cv2.FlannBasedMatcher_create()
    match = matcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
    best_2 = matcher.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
    ratio = 0.7
    match = []
    for m, n in best_2:
        if m.distance < ratio * n.distance:
            match.append(m)
    # TODO: select best `max_n_match` matches
    match = sorted(match, key=lambda x: x.distance)

    match = match[:100]
    return keypoints1, keypoints2, match


def draw_match(img1, keypoints1, img2, keypoints2, match, savename):
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: draw matches by `cv2.drawMatches`
    match_draw = cv2.drawMatches(
        img1        = img1,
        keypoints1  = keypoints1,
        img2        = img2,
        keypoints2  = keypoints2,
        matches1to2 = match,
        outImg      = None,
        flags       = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('match.jpg', match_draw)


def transform(img, img_kps, dst_kps, H, W):
'''
Transfrom img such `img_kps` are aligned with `dst_kps`.
H: height of output image
W: width of output image
'''

    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])
    src, dst = img2, img1
    src_kps, dst_kps = (keypoints2, keypoints1)
    # TODO: get transform matrix by `cv2.findHomography`
    T, status = cv2.findHomography(
                    srcPoints = src_kps,
                    dstPoints = dst_kps,
                    method    = cv2.USAC_ACCURATE,
                    ransacReprojThreshold = 3)


    # TODO: apply transform by `cv2.warpPerspective`
    H, W, _ = img2.shape
    W = W*2
    transformed2 = cv2.warpPerspective(
                    src   = img2,
                    M     = T,
                    dsize = (W, H),
                    dst   = np.zeros_like(img2),
                    borderMode = cv2.BORDER_TRANSPARENT)
    cv2.imwrite('transformed.jpg',transformed2)

    return transformed


if __name__ == '__main__':
    ## read images
    img1 = imread('image/left2.jpg')
    img2 = imread('image/right2.jpg')

    ## find keypoints and matches
    keypoints1, keypoints2, match = keypoint_match(img1, img2, max_n_match=1000)

    draw_match(img1, keypoints1, img2, keypoints2,
               match, savename='match.jpg')

    # get all matched keypoints
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])

    ## Align img2 to img1
    H, W = img1.shape[:2]
    W = W*2
    new_img2 = transform(img2, keypoints2, keypoints1, H, W)
    write_and_show('transformed.jpg', new_img2)

    # resize img1
    transformed1 =  np.hstack([img1, np.zeros_like(img1)])
    direct_mean = transformed1 / 2 + transformed2 / 2

    # TODO: average `new_img1` and `new_img2`
    cnt = np.zeros([H,W,1]) + 1e-10
    cnt += (transformed2 != 0).any(2, keepdims=True)
    cnt += (transformed1 != 0).any(2, keepdims=True)
    transformed1 = np.float32(transformed1)
    transformed2 = np.float32(transformed2)
    stack = (transformed2 + transformed1)/cnt
    cv2.imwrite('stack.jpg', stack)


    destroyAllWindows()
