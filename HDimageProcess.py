# coding=UTF-8
import os
import cv2

import math
import numpy
import numpy as np
from math import ceil, floor

import tensorflow as tf

Iterator = tf.data.Iterator
from utils import compute_nc, compute_psnr, compute_ssim, compute_ber


class Point:

    def __init__(self, x=0, y=0):
        self.x = x    # 名称
        self.y = y     # 尺寸


class Region:

    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0):
        self.xmin = xmin   # 名称
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def computCornersNoRotation(ptCenter, edge):
    ptCorners = []
    diag = edge / 2
    pt0 = Point(ptCenter.x - diag, ptCenter.y - diag)
    ptCorners.append(pt0)
    pt1 = Point(ptCenter.x + diag, ptCenter.y - diag)
    ptCorners.append(pt1)
    pt2 = Point(ptCenter.x + diag, ptCenter.y + diag)
    ptCorners.append(pt2)
    pt3 = Point(ptCenter.x - diag, ptCenter.y + diag)
    ptCorners.append(pt3)
    return ptCorners


def getCentroid(img):
    x = 0.0
    y = 0.0
    sum = cv2.sumElems(img)
    dwidth = img.shape[1]
    dheight = img.shape[0]
    for i in range(0, dheight):
        for j in range(0, dwidth):
            x = x + j * img[i, j]
            y = y + i * img[i, j]
    center = Point()
    center.x = int(round(x / sum[0]))
    center.y = int(round(y / sum[0]))
    return center


def nonMaxSuppression(keypoints, edge):
    thres = edge * 2
    thres = thres * thres
    lenk = len(keypoints)
    i = 1
    while i < lenk:
        for j in range(0, i):
            dx = keypoints[j].pt[0] - keypoints[i].pt[0]
            dy = keypoints[j].pt[1] - keypoints[i].pt[1]
            sdist = dx * dx + dy * dy
            if sdist < thres:
                keypoints.remove(keypoints[i])
                lenk = lenk - 1
                i = i - 1
                break
        i = i + 1
    return keypoints


def getEmbeddingRegion(img, key, num, mask=None):
    ref_edge = 96
#     half_edge = ref_edge / 2
    key = (key * 99991 + 97) % 1000000
    key_1 = (key / 1000) / 1000.0
    key_2 = (key % 1000) / 1000.0

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = math.sqrt(1024 * 1024.0 / img.shape[0] / img.shape[1])
    dwidth = int(round(img.shape[1] * scale))
    dheight = int(round(img.shape[0] * scale))
    image2 = cv2.resize(src, (dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    centroid = Point()
    centroid.x = dwidth / 2
    centroid.y = dheight / 2
    edge = ref_edge / scale
   # edge = int(round(edge / 4.0) * 4)
    edge = int(round(edge / 8.0) * 8)
    stdedg = edge * scale
    half_edge = stdedg / 2
#     centroid = getCentroid(image2)

    mask = np.zeros_like(image2)
    dx = int(round(dwidth * 0.1))
    dy = int(round(dheight * 0.1))
    mask[dy:dheight - dy, dx:dwidth - dx] = 255
    for row in range(centroid.y - ref_edge, centroid.y + ref_edge):
        dd = int(round(math.sqrt(
            ref_edge * ref_edge - (centroid.y - row) * (centroid.y - row))))
        mask[row, centroid.x - dd:centroid.x + dd - 1] = 0

#     cv2.imwrite('data/mask.jpg',
#                 (mask).astype(np.uint8))
    minHessian = 1500

    surf = cv2.xfeatures2d.SURF_create(minHessian, upright=True)

    keypoints = surf.detect(image2, mask)

    regions = []

    n = 0
#     keypoints = nonMaxSuppression(keypoints, stdedg)
    lenk = len(keypoints)
    # sort keypoints based on size, small size first

    for i in range(0, lenk):
        if n >= num:
            break
        ptx = int(round(keypoints[i].pt[0]))
        pty = int(round(keypoints[i].pt[1]))
#         if mask[pty, ptx] < 200:
#             continue

        dx = keypoints[i].pt[0] - centroid.x
        dy = keypoints[i].pt[1] - centroid.y

        dist = math.sqrt(dx * dx + dy * dy) / 2
        alpha = math.atan2(
            centroid.y - keypoints[i].pt[1], centroid.x - keypoints[i].pt[0])
        theta = 2 * math.pi * key_1 + alpha

        flagInside = False

        rad = dist * key_2

        sox =  keypoints[i].pt[0] + \
            rad * math.cos(theta)
        soy = keypoints[i].pt[1] + \
            rad * math.sin(theta)
        if sox + half_edge >= dwidth or sox - half_edge < 0 or soy + half_edge >= dheight or soy - half_edge < 0:
            sox = keypoints[i].pt[0] - rad * math.cos(theta)
            soy = keypoints[i].pt[1] - rad * math.sin(theta)

        region = Region(
            sox - half_edge, soy - half_edge, sox + half_edge, soy + half_edge)

        if (region.xmin >= 0 and region.xmax < dwidth
                and region.ymin >= 0 and region.ymax < dheight):
            flagInside = True

        if flagInside:
            overlap = False
            for k in range(0, n):
                if (region.xmin > regions[k].xmax or region.xmax < regions[k].xmin
                        or region.ymin > regions[k].ymax or region.ymax < regions[k].ymin):
                    continue
                else:
                    overlap = True
                    break
            if not overlap:

                regions.append(region)
                n = n + 1
    while n < num:
        regions.append(None)
        n = n + 1
    for i in range(0, num):
        if regions[i] is not None:
            regions[i].xmin = int(round(regions[i].xmin / scale))
            regions[i].xmax = int(round(regions[i].xmax / scale))
            regions[i].ymin = int(round(regions[i].ymin / scale))
            regions[i].ymax = int(round(regions[i].ymax / scale))

    return regions


def getRevealingRegion(img, key, num, mask=None):
    ref_edge = 96
#     half_edge = ref_edge / 2
    key = (key * 99991 + 97) % 1000000
    key_1 = (key / 1000) / 1000.0
    key_2 = (key % 1000) / 1000.0

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = math.sqrt(1024 * 1024.0 / img.shape[0] / img.shape[1])
    dwidth = int(round(img.shape[1] * scale))
    dheight = int(round(img.shape[0] * scale))
    image2 = cv2.resize(src, (dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    centroid = Point()
    centroid.x = dwidth / 2
    centroid.y = dheight / 2
#     centroid = getCentroid(image2)

    mask = np.zeros_like(image2)
    dx = int(round(dwidth * 0.1))
    dy = int(round(dheight * 0.1))
    mask[dy:dheight - dy, dx:dwidth - dx] = 255
    for row in range(centroid.y - ref_edge, centroid.y + ref_edge):
        dd = int(round(math.sqrt(
            ref_edge * ref_edge - (centroid.y - row) * (centroid.y - row))))
        mask[row, centroid.x - dd:centroid.x + dd - 1] = 0
    minHessian = 1500

    surf = cv2.xfeatures2d.SURF_create(minHessian, upright=True)

    keypoints = surf.detect(image2, mask)

    edge = ref_edge / scale
    edge = int(round(edge / 4.0) * 4)
    stdedg = edge * scale
    half_edge = stdedg / 2
    regions = []

    n = 0
#     keypoints = nonMaxSuppression(keypoints, stdedg)
    lenk = len(keypoints)
    # sort keypoints based on size, small size first

    for i in range(0, lenk):
        if n >= num:
            break
        ptx = int(round(keypoints[i].pt[0]))
        pty = int(round(keypoints[i].pt[1]))
#         if mask[pty, ptx] < 200:
#             continue

        dx = keypoints[i].pt[0] - centroid.x
        dy = keypoints[i].pt[1] - centroid.y

        dist = math.sqrt(dx * dx + dy * dy) / 2
        alpha = math.atan2(
            centroid.y - keypoints[i].pt[1], centroid.x - keypoints[i].pt[0])
        theta = 2 * math.pi * key_1 + alpha

        flagInside = False

        rad = dist * key_2

        sox =  keypoints[i].pt[0] + \
            rad * math.cos(theta)
        soy = keypoints[i].pt[1] + \
            rad * math.sin(theta)
        if sox + half_edge >= dwidth or sox - half_edge < 0 or soy + half_edge >= dheight or soy - half_edge < 0:
            sox = keypoints[i].pt[0] - rad * math.cos(theta)
            soy = keypoints[i].pt[1] - rad * math.sin(theta)

        region = Region(
            sox - half_edge, soy - half_edge, sox + half_edge, soy + half_edge)

        if (region.xmin >= 0 and region.xmax < dwidth
                and region.ymin >= 0 and region.ymax < dheight):
            flagInside = True

        if flagInside:
            regions.append(region)
            n = n + 1
    while n < num:
        regions.append(None)
        n = n + 1
    for i in range(0, num):
        if regions[i] is not None:
            regions[i].xmin = int(round(regions[i].xmin / scale))
            regions[i].xmax = int(round(regions[i].xmax / scale))
            regions[i].ymin = int(round(regions[i].ymin / scale))
            regions[i].ymax = int(round(regions[i].ymax / scale))

    return regions


def getEmbeddingRegionBK2(img, key, num, mask=None):
    ref_edge = 48
#     half_edge = ref_edge / 2
    key = (key * 99991 + 97) % 1000000
    key_1 = (key / 1000) / 1000.0
    key_2 = (key % 1000) / 1000.0

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = cv2.GaussianBlur(src, (5, 5), 0)
    scale = math.sqrt(1024 * 1024.0 / img.shape[0] / img.shape[1])
    dwidth = int(round(img.shape[1] * scale))
    dheight = int(round(img.shape[0] * scale))
    image2 = cv2.resize(src, (dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    centroid = getCentroid(image2)

    mask = np.zeros_like(image2)
    dx = int(round(dwidth * 0.1))
    dy = int(round(dheight * 0.1))
    mask[dy:dheight - dy, dx:dwidth - dx] = 255
    kp = cv2.goodFeaturesToTrack(
        image2, 10, 0.02, ref_edge, blockSize=37, mask=mask)
    keypoints = []
    if kp is not None and len(kp) > 0:
        for x, y in np.float32(kp).reshape(-1, 2):
            pt = Point()
            pt.x = x
            pt.y = y
            keypoints.append(pt)
    edge = ref_edge / scale
    edge = int(round(edge / 4.0) * 4)
    stdedg = edge * scale
    half_edge = stdedg / 2
    regions = []

    n = 0

    lenk = len(keypoints)
    # sort keypoints based on size, small size first

    for i in range(1, lenk):
        if n >= num:
            break
        for j in range(0, i):
            if n >= num or i >= lenk:
                break

            dx = keypoints[j].x - keypoints[i].x
            dy = keypoints[j].y - keypoints[i].y

            dist = math.sqrt(dx * dx + dy * dy)

            ox = (keypoints[j].x + keypoints[i].x) / 2
            oy = (keypoints[j].y + keypoints[i].y) / 2

            theta = math.pi - math.atan2(dx, dy)
            alpha = math.atan2(centroid.y - oy, centroid.x - ox)

            if math.cos(theta - alpha) < 0:
                theta = math.pi + theta
            flagInside = False
            while not flagInside:
                shiftx = dist * key_1
                shifty = dist * key_2
                sox = ox + \
                    (shiftx * math.cos(theta) + shifty * math.sin(theta))
                soy = oy + \
                    (shiftx * math.sin(theta) - shifty * math.cos(theta))

                if sox + half_edge > dwidth or sox - half_edge < 0 or soy + half_edge > dheight or soy - half_edge < 0:
                    sox = ox + \
                        (shiftx * math.cos(theta) - shifty * math.sin(theta))
                    soy = oy + \
                        (shiftx * math.sin(theta) + shifty * math.cos(theta))

                region = Region(
                    sox - half_edge, soy - half_edge, sox + half_edge, soy + half_edge)

                if (region.xmin >= 0 and region.xmax < dwidth
                        and region.ymin >= 0 and region.ymax < dheight):
                    flagInside = True
                elif dist > 128:
                    dist = dist / 2
                else:
                    break
            if flagInside:
                overlap = False
                for k in range(0, n):
                    if (region.xmin > regions[k].xmax or region.xmax < regions[k].xmin
                            or region.ymin > regions[k].ymax or region.ymax < regions[k].ymin):
                        continue
                    else:
                        overlap = True
                        break
                if not overlap:

                    regions.append(region)
                    n = n + 1
    while n < num:
        regions.append(None)
        n = n + 1
    for i in range(0, num):
        if regions[i] is not None:
            regions[i].xmin = int(round(regions[i].xmin / scale))
            regions[i].xmax = int(round(regions[i].xmax / scale))
            regions[i].ymin = int(round(regions[i].ymin / scale))
            regions[i].ymax = int(round(regions[i].ymax / scale))

    return regions


def getRevealingRegionBK2(img, key, num, mask=None):
    ref_edge = 48
#     half_edge = ref_edge / 2
    key = (key * 99991 + 97) % 1000000
    key_1 = (key / 1000) / 1000.0
    key_2 = (key % 1000) / 1000.0

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = cv2.GaussianBlur(src, (5, 5), 0)
    scale = math.sqrt(1024 * 1024.0 / img.shape[0] / img.shape[1])
    dwidth = int(round(img.shape[1] * scale))
    dheight = int(round(img.shape[0] * scale))
    src = cv2.medianBlur(src, 3)
    image2 = cv2.resize(src, (dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    centroid = getCentroid(image2)

    mask = np.zeros_like(image2)
    dx = int(round(dwidth * 0.1))
    dy = int(round(dheight * 0.1))
    mask[dy:dheight - dy, dx:dwidth - dx] = 255
    kp = cv2.goodFeaturesToTrack(
        image2, 10, 0.02, ref_edge, blockSize=37, mask=mask)
    keypoints = []
    if kp is not None and len(kp) > 0:
        for x, y in np.float32(kp).reshape(-1, 2):
            pt = Point()
            pt.x = x
            pt.y = y
            keypoints.append(pt)
    edge = ref_edge / scale
    edge = int(round(edge / 4.0) * 4)
    stdedg = edge * scale
    half_edge = stdedg / 2
    regions = []

    n = 0

    lenk = len(keypoints)
    # sort keypoints based on size, small size first

    for i in range(1, lenk):
        if n >= num:
            break
        for j in range(0, i):
            if n >= num or i >= lenk:
                break

            dx = keypoints[j].x - keypoints[i].x
            dy = keypoints[j].y - keypoints[i].y

            dist = math.sqrt(dx * dx + dy * dy)

            ox = (keypoints[j].x + keypoints[i].x) / 2
            oy = (keypoints[j].y + keypoints[i].y) / 2

            theta = math.pi - math.atan2(dx, dy)
            alpha = math.atan2(centroid.y - oy, centroid.x - ox)

            if math.cos(theta - alpha) < 0:
                theta = math.pi + theta

            flagInside = False
            while not flagInside:
                shiftx = dist * key_1
                shifty = dist * key_2
                sox = ox + \
                    (shiftx * math.cos(theta) + shifty * math.sin(theta))
                soy = oy + \
                    (shiftx * math.sin(theta) - shifty * math.cos(theta))

                if sox + half_edge > dwidth or sox - half_edge < 0 or soy + half_edge > dheight or soy - half_edge < 0:
                    sox = ox + \
                        (shiftx * math.cos(theta) - shifty * math.sin(theta))
                    soy = oy + \
                        (shiftx * math.sin(theta) + shifty * math.cos(theta))

                region = Region(
                    sox - half_edge, soy - half_edge, sox + half_edge, soy + half_edge)

                if (region.xmin >= 0 and region.xmax < dwidth
                        and region.ymin >= 0 and region.ymax < dheight):
                    flagInside = True
                elif dist > 128:
                    dist = dist / 2
                else:
                    break
            if flagInside:
                regions.append(region)
                n = n + 1
    while n < num:
        regions.append(None)
        n = n + 1
    for i in range(0, num):
        if regions[i] is not None:
            regions[i].xmin = int(round(regions[i].xmin / scale))
            regions[i].xmax = int(round(regions[i].xmax / scale))
            regions[i].ymin = int(round(regions[i].ymin / scale))
            regions[i].ymax = int(round(regions[i].ymax / scale))

    return regions


def getEmbeddingRegionbk1(img, key, num, mask=None):
    ref_edge = 48
#     half_edge = ref_edge / 2
    key = (key * 99991 + 97) % 1000000
    key_1 = (key / 1000) / 1000.0
    key_2 = (key % 1000) / 1000.0

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = math.sqrt(1024 * 1024.0 / img.shape[0] / img.shape[1])
    dwidth = int(round(img.shape[1] * scale))
    dheight = int(round(img.shape[0] * scale))
    image2 = cv2.resize(src, (dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    centroid = getCentroid(image2)

    minHessian = 1500
    surf = cv2.xfeatures2d.SURF_create(minHessian, upright=True)

    keypoints = surf.detect(image2, None)
    edge = ref_edge / scale
    edge = int(round(edge / 4.0) * 4)
    stdedg = edge * scale
    half_edge = stdedg / 2
    regions = []

    n = 0
    keypoints = nonMaxSuppression(keypoints, stdedg)
    lenk = len(keypoints)
    # sort keypoints based on size, small size first

    for i in range(1, lenk):
        if n >= num:
            break
        for j in range(0, i):
            if n >= num or i >= lenk:
                break

            dx = keypoints[j].pt[0] - keypoints[i].pt[0]
            dy = keypoints[j].pt[1] - keypoints[i].pt[1]

            dist = math.sqrt(dx * dx + dy * dy)
#             while dist < 3 and i < lenk:
#                 keypoints.remove(keypoints[i])
#                 dx = keypoints[j].pt[0] - keypoints[i].pt[0]
#                 dy = keypoints[j].pt[1] - keypoints[i].pt[1]
#                 lenk = lenk - 1
#
#                 dist = math.sqrt(dx * dx + dy * dy)

            ox = (keypoints[j].pt[0] + keypoints[i].pt[0]) / 2
            oy = (keypoints[j].pt[1] + keypoints[i].pt[1]) / 2

            theta = math.pi - math.atan2(dx, dy)
            alpha = math.atan2(centroid.y - oy, centroid.x - ox)

            if math.cos(theta - alpha) < 0:
                theta = math.pi + theta
            flagInside = False
            while not flagInside:
                shiftx = dist * key_1
                shifty = dist * key_2
                sox = ox + \
                    (shiftx * math.cos(theta) + shifty * math.sin(theta))
                soy = oy + \
                    (shiftx * math.sin(theta) - shifty * math.cos(theta))

                if sox + half_edge > dwidth or sox - half_edge < 0 or soy + half_edge > dheight or soy - half_edge < 0:
                    sox = ox + \
                        (shiftx * math.cos(theta) - shifty * math.sin(theta))
                    soy = oy + \
                        (shiftx * math.sin(theta) + shifty * math.cos(theta))

                region = Region(
                    sox - half_edge, soy - half_edge, sox + half_edge, soy + half_edge)

                if (region.xmin >= 0 and region.xmax < dwidth
                        and region.ymin >= 0 and region.ymax < dheight):
                    flagInside = True
                elif dist > 128:
                    dist = dist / 2
                else:
                    break
            if flagInside:
                overlap = False
                for k in range(0, n):
                    if (region.xmin > regions[k].xmax or region.xmax < regions[k].xmin
                            or region.ymin > regions[k].ymax or region.ymax < regions[k].ymin):
                        continue
                    else:
                        overlap = True
                        break
                if not overlap:

                    regions.append(region)
                    n = n + 1
    while n < num:
        regions.append(None)
        n = n + 1
    for i in range(0, num):
        if regions[i] is not None:
            regions[i].xmin = int(round(regions[i].xmin / scale))
            regions[i].xmax = int(round(regions[i].xmax / scale))
            regions[i].ymin = int(round(regions[i].ymin / scale))
            regions[i].ymax = int(round(regions[i].ymax / scale))

    return regions


def getRevealingRegionbk1(img, key, num, mask=None):
    ref_edge = 48
#     half_edge = ref_edge / 2
    key = (key * 99991 + 97) % 1000000
    key_1 = (key / 1000) / 1000.0
    key_2 = (key % 1000) / 1000.0

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = math.sqrt(1024 * 1024.0 / img.shape[0] / img.shape[1])
    dwidth = int(round(img.shape[1] * scale))
    dheight = int(round(img.shape[0] * scale))
    image2 = cv2.resize(src, (dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    centroid = getCentroid(image2)

    minHessian = 1500
    surf = cv2.xfeatures2d.SURF_create(minHessian, upright=True)

    keypoints = surf.detect(image2, None)
    edge = ref_edge / scale
    edge = int(round(edge / 4.0) * 4)
    stdedg = edge * scale
    half_edge = stdedg / 2
    regions = []

    n = 0
    keypoints = nonMaxSuppression(keypoints, stdedg)
    lenk = len(keypoints)
    # sort keypoints based on size, small size first

    for i in range(1, lenk):
        if n >= num:
            break
        for j in range(0, i):
            if n >= num or i >= lenk:
                break

            dx = keypoints[j].pt[0] - keypoints[i].pt[0]
            dy = keypoints[j].pt[1] - keypoints[i].pt[1]

            dist = math.sqrt(dx * dx + dy * dy)
#             while dist < 3 and i < lenk:
#                 keypoints.remove(keypoints[i])
#                 dx = keypoints[j].pt[0] - keypoints[i].pt[0]
#                 dy = keypoints[j].pt[1] - keypoints[i].pt[1]
#                 lenk = lenk - 1
#
#                 dist = math.sqrt(dx * dx + dy * dy)

            ox = (keypoints[j].pt[0] + keypoints[i].pt[0]) / 2
            oy = (keypoints[j].pt[1] + keypoints[i].pt[1]) / 2

            theta = math.pi - math.atan2(dx, dy)
            alpha = math.atan2(centroid.y - oy, centroid.x - ox)

            if math.cos(theta - alpha) < 0:
                theta = math.pi + theta

            flagInside = False
            while not flagInside:
                shiftx = dist * key_1
                shifty = dist * key_2
                sox = ox + \
                    (shiftx * math.cos(theta) + shifty * math.sin(theta))
                soy = oy + \
                    (shiftx * math.sin(theta) - shifty * math.cos(theta))

                if sox + half_edge > dwidth or sox - half_edge < 0 or soy + half_edge > dheight or soy - half_edge < 0:
                    sox = ox + \
                        (shiftx * math.cos(theta) - shifty * math.sin(theta))
                    soy = oy + \
                        (shiftx * math.sin(theta) + shifty * math.cos(theta))

                region = Region(
                    sox - half_edge, soy - half_edge, sox + half_edge, soy + half_edge)

                if (region.xmin >= 0 and region.xmax < dwidth
                        and region.ymin >= 0 and region.ymax < dheight):
                    flagInside = True
                elif dist > 128:
                    dist = dist / 2
                else:
                    break
            if flagInside:
                regions.append(region)
                n = n + 1
    while n < num:
        regions.append(None)
        n = n + 1
    for i in range(0, num):
        if regions[i] is not None:
            regions[i].xmin = int(round(regions[i].xmin / scale))
            regions[i].xmax = int(round(regions[i].xmax / scale))
            regions[i].ymin = int(round(regions[i].ymin / scale))
            regions[i].ymax = int(round(regions[i].ymax / scale))

    return regions


def drawKeypoint(img, mask=None, num=5):
    edge = 96
    ref_edge = edge

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = math.sqrt(1024 * 1024.0 / img.shape[0] / img.shape[1])
    dwidth = int(round(img.shape[1] * scale))
    dheight = int(round(img.shape[0] * scale))
    image2 = cv2.resize(src, (dwidth, dheight))
    centroid = Point()
    centroid.x = dwidth / 2
    centroid.y = dheight / 2
    mask = np.zeros_like(image2)
    dx = int(round(dwidth * 0.1))
    dy = int(round(dheight * 0.1))
    mask[dy:dheight - dy, dx:dwidth - dx] = 255
    for row in range(centroid.y - ref_edge, centroid.y + ref_edge):
        dd = int(round(math.sqrt(
            ref_edge * ref_edge - (centroid.y - row) * (centroid.y - row))))
        mask[row, centroid.x - dd:centroid.x + dd - 1] = 0


#     if mask is not None:
#         mask = cv2.resize(mask, (dwidth, dheight))
    minHessian = 1500
    surf = cv2.xfeatures2d.SURF_create(minHessian)
#     sift = cv2.xfeatures2d.SIFT_create()
    keypoints = surf.detect(image2, mask)
#     keypoints = nonMaxSuppression(keypoints, edge)

    dist = 0.0
#     i = 1
#     while dist < 30:
#         if i > 1:
#             keypoints.remove(keypoints[1])
#         dx = keypoints[0].pt[0] - keypoints[1].pt[0]
#         dy = keypoints[0].pt[1] - keypoints[1].pt[1]
#
#         dist = math.sqrt(dx * dx + dy * dy)
#
#         i = i + 1

    pt = []
    for i in range(num):
        newpt = Point()
        newpt.x = (int)(keypoints[i].pt[0] / scale)
        newpt.y = int(keypoints[i].pt[1] / scale)
        pt.append(newpt)
    delta = 0  # 255 / num
    for i in range(num):
        r = 255 - i * delta
        b = i * delta
        cv2.line(img, (pt[i].x - 80, pt[i].y),
                 (pt[i].x + 80, pt[i].y), (r, r, r), 10)
        cv2.line(img, (pt[i].x, pt[i].y - 80),
                 (pt[i].x, pt[i].y + 80), (r, r, r), 10)
        cv2.circle(img, (pt[i].x, pt[i].y), 200, (r, r, r), 10)
#         if i == 2:
#             cv2.imwrite('data/tt.jpg',
#                         (img).astype(np.uint8))

    return img


def findRegionBasedTwoPoints(pt1, pt2, dwidth, dheight, scale, key, edge):
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]

    dist = math.sqrt(dx * dx + dy * dy)

    r = dist * key / 1000 / 250.0

    stdedg = dist
    if dist > edge:
        while dist > edge:
            dist = dist / 2
        if dist * 2 - edge > edge - dist:
            stdedg = dist
        else:
            stdedg = dist * 2
    elif dist < edge:
        while dist < edge:
            dist = dist * 2
        if dist - edge > edge - dist / 2:
            stdedg = dist / 2
        else:
            stdedg = dist
    edge = stdedg / scale
    edge = int(round(edge / 4.0) * 4)
    stdedg = edge * scale

    beta = math.atan(
        (pt1[1] - pt2[1]) / (pt2[0] - pt1[0]))
    # left plane
    if pt2[0] < pt1[0]:
        beta = beta + numpy.pi
#     alpha = math.atan(sy / sx)
    alpha = key % 1000 / 500.0 * numpy.pi
    findRegion = False

    while not findRegion:
        ptCenter = []

        ptCenter.append(
            Point(pt1[0] + r * math.cos(alpha + beta), pt1[1] - r * math.sin(alpha + beta)))

        ptCenter.append(
            Point(pt1[0] - r * math.sin(beta + alpha), pt1[1] - r * math.cos(beta + alpha)))

        ptCenter.append(
            Point(pt1[0] - r * math.cos(alpha + beta), pt1[1] + r * math.sin(alpha + beta)))

        ptCenter.append(
            Point(pt1[0] + r * math.sin(alpha + beta), pt1[1] + r * math.cos(alpha + beta)))

        flagin = False
        for k in range(0, 4):
            if flagin:
                break
            flagin = True
            ptCorners = computCornersNoRotation(ptCenter[k], stdedg)
            for i in range(0, 4):
                if ptCorners[i].x < 0 or ptCorners[i].x >= dwidth or \
                        ptCorners[i].y < 0 or ptCorners[i].y >= dheight:
                    flagin = False
                    break
        if k >= 3 and not flagin:
            print 'out of range'
            if r < stdedg:
                return None
            else:
                r = r / 2
        else:
            k = k - 1
            findRegion = True

    for i in range(0, 4):
        ptCorners[i].x = int(round(ptCorners[i].x / scale))
        ptCorners[i].y = int(round(ptCorners[i].y / scale))
    return Region(ptCorners[0].x, ptCorners[0].y, ptCorners[2].x, ptCorners[2].y)


def normlizeWatermark(secretPath, edge):
    wm = cv2.imread(
        secretPath, cv2.IMREAD_GRAYSCALE)
    edgeinner = int(edge * 0.6)
    image2 = cv2.resize(wm, (edgeinner, edgeinner))
    startidx = int(round((edge - edgeinner) / 2))

    wm = numpy.zeros((edge, edge), dtype=int)
    wm[startidx:startidx + edgeinner,
        startidx:startidx + edgeinner] = image2
    return wm


def replaceStegoRegion(img, stego, ptCorners):
    xstart = ptCorners.xmin + 3
    ystart = ptCorners.ymin + 3
    xend = ptCorners.xmax - 3
    yend = ptCorners.ymax - 3
    img[ystart:yend, xstart:xend] = stego[
        3:(stego.shape[0] - 3), 3:(stego.shape[1] - 3)]
    return img


def rotateImg(img, angle):
    rows, cols = img.shape[:2]
#     mask = numpy.zeros((rows, cols), numpy.uint8)
#     mask[64:rows - 64, 64:cols - 64] = 255
    affineTrans = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    costheta = math.cos(angle * numpy.pi / 180)
    sintheta = math.sin(angle * numpy.pi / 180)
    if sintheta < 0:
        sintheta = -sintheta
    newwidth = (int)((cols * costheta - rows * sintheta) /
                     (costheta * costheta - sintheta * sintheta))
    newheight = (int)((rows * costheta - cols * sintheta) /
                      (costheta * costheta - sintheta * sintheta))
    dst = cv2.warpAffine(
        img, affineTrans, (cols, rows))
    dst = cv2.warpAffine(
        img, affineTrans, (cols, rows),  borderMode=cv2.BORDER_REPLICATE)
# mask = cv2.warpAffine(mask, affineTrans, (cols,
# rows))flags=cv2.INTER_CUBIC,
    valid_dst = dst[rows / 2 - newheight / 2:rows / 2 + newheight / 2 - 1, cols / 2 - newwidth / 2:cols / 2 + newwidth /
                    2 - 1]
    return dst, valid_dst


def rotateWm(img, angle):
    rows, cols = img.shape[:2]
#     mask = numpy.zeros((rows, cols), numpy.uint8)
#     mask[64:rows - 64, 64:cols - 64] = 255
    affineTrans = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    costheta = math.cos(angle * numpy.pi / 180)
    sintheta = math.sin(angle * numpy.pi / 180)
    if sintheta < 0:
        sintheta = -sintheta
    newwidth = (int)((cols * costheta - rows * sintheta) /
                     (costheta * costheta - sintheta * sintheta))
    newheight = (int)((rows * costheta - cols * sintheta) /
                      (costheta * costheta - sintheta * sintheta))
    dst = cv2.warpAffine(
        img, affineTrans, (cols, rows))

    return dst

# def test_embedding(imgPath, key=123456, num=4):
#     log_path = 'logs/0710-2105'
#     train_model = Model()
#     input_placeholder = tf.placeholder(
#         shape=[None, None, None, 3], dtype=tf.float32)
#     reveal_output_op = train_model.get_reveal_network_op(
#         input_placeholder, is_training=False, transform=True)
#     # # reveal_output_op = self.get_reveal_network_op(self.stego_tensor, is_training=False, transform=True)
#     # reveal_output_op = self.get_reveal_network_op(self.stego_tensor, is_training=False)
#     reveal_output_op = tf.nn.sigmoid(reveal_output_op)
#
#     _, hidden = train_model.get_hiding_network_op(
#         cover_tensor=train_model.cover_tensor, secret_tensor=train_model.secret_tensor, is_training=False)
#
#     if os.path.exists(log_path + '.meta'):
#         loader = log_path
#     else:
#         loader = tf.train.latest_checkpoint(log_path)
# #     global_variables = tf.global_variables()
# #     encode_vars = [
# #         i for i in global_variables if i.name.startswith('encode')]
#     train_model.sess.run(tf.global_variables_initializer())
#     train_model.saver = tf.train.Saver()
#     train_model.saver.restore(train_model.sess, loader)
#     print('load model %s' % loader)
#     img = cv2.imread(imgPath)
#     if img is None or img.size <= 0:
#         os._exit(0)
#     regions = getEmbeddingRegion(img, key, num)
#     edge = regions[0].xmax - regions[0].xmin
#     secret_origin = normlizeWatermark(
#         'data/secret.jpg', edge)
#     for i in range(0, key):
#
#         embeddingRegion = img[
#             regions[i].ymin:regions[i].ymax, regions[i].xmin:regions[i].xmax]
#         cv2.imwrite('data/coverregion%s.jpg' % i,
#                     embeddingRegion)
#         dir = imgPath[0:imgPath.rindex('/')]
#
#         extract_image = embeddingRegion / 255.0
#         secret_image = (
#             np.expand_dims(secret_origin, 0) > 128).astype(np.uint8)
#         secret_image = np.expand_dims(secret_image, -1)
#         # stego = train_model.get_stego(
#         #    'logs/0621-0153', np.expand_dims(extract_image, 0).astype(np.float32), secret_image.astype(np.float32))
#         stego = train_model.sess.run(
#             hidden,
#             feed_dict={train_model.secret_tensor: secret_image.astype(np.float32),
#                        train_model.cover_tensor: np.expand_dims(extract_image, 0).astype(np.float32)})
#
#         stego = (np.clip(stego, 0, 1) * 255).astype(np.uint8)
#
#         cv2.imwrite('data/regionstego%s.jpg' % i,
#                     stego[0])
#         img = replaceStegoRegion(img, stego[0], regions[i])
#     cv2.imwrite('data/stego.jpg',
#                 img)
#     img = cv2.imread('data/stego.jpg')
#     nc = 0.0
#     regions = getEmbeddingRegion(img, key)
#
#     i = 0
#     while i < key:
#         if regions[i] is not None:
#             edge = regions[i].xmax - regions[i].xmin
#             embeddingRegion = img[
#                 regions[i].ymin:regions[i].ymax, regions[i].xmin:regions[i].xmax]
#             cv2.imwrite('data/redetectedstego%s.jpg' % i,
#                         embeddingRegion)
#             stego = np.expand_dims(embeddingRegion, 0)
#     #     stego = cv2.imread(
#     #         '/media/lqzhu/d/ISGAN-master/data/regionstego.jpg')
#     #     # stego = cv2.imread('/home/gl/Watermark_paper/test/-230000/cover/stego.jpg')
#     #     stego = np.expand_dims(stego, 0)
#             secret_reveal = train_model.sess.run(
#                 reveal_output_op, feed_dict={input_placeholder: stego / 255.})
#             secret_image = (
#                 np.clip(secret_reveal, 0, 1) * 255).astype(np.uint8)
#
#             #secret_image = train_model.get_secret('logs/0621-0153', stego / 255.)
#             edgeinner = int(edge / math.sqrt(2.0))
#             startidx = int(round((edge - edgeinner) / 2))
#
#             wm = secret_image[0]
#             secret_origin = cv2.imread(
#                 'data/secret.jpg', cv2.IMREAD_GRAYSCALE)
#             secret_origin = cv2.resize(secret_origin, (edgeinner, edgeinner))
#
#             nc = compute_nc(secret_origin, wm[startidx:startidx + edgeinner,
#                                               startidx:startidx + edgeinner], edgeinner)
#             if nc > 0.6:
#                 cv2.imwrite('data/revealedsecret.jpg',
#                             (secret_image[0]).astype(np.uint8))
#         i = i + 1


def drawRegion(imgPath, key=123456, num=4):
    img = cv2.imread(imgPath)
    regions = getEmbeddingRegion(img, key, num)
    delta = 255 / num

    for i in range(0, num):
        if regions[i] is not None:
            cv2.imwrite('data/IMG6948region' + str(i) + '.jpg',
                        img[regions[i].ymin:regions[i].ymax, regions[i].xmin:regions[i].xmax])
            r = 255  # - i * delta
            b = i * delta
            cv2.line(img, (int(regions[i].xmin), int(regions[i].ymin)),
                     (int(regions[i].xmax), int(regions[i].ymin)), (r, r, r), 10)
            cv2.line(img, (int(regions[i].xmax), int(regions[i].ymin)),
                     (int(regions[i].xmax), int(regions[i].ymax)), (r, r, r), 10)
            cv2.line(img, (int(regions[i].xmax), int(regions[i].ymax)),
                     (int(regions[i].xmin), int(regions[i].ymax)), (r, r, r), 10)
            cv2.line(img, (int(regions[i].xmin), int(regions[i].ymax)),
                     (int(regions[i].xmin), int(regions[i].ymin)), (r, r, r), 10)

    return img


#     test_embedding(
#         'data/DSC_0309.JPG')
