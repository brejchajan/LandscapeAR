# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:47:00+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:43:47+02:00
# @License: Copyright 2020 CPhoto@FIT, Brno University of Technology,
# Faculty of Information Technology,
# Božetěchova 2, 612 00, Brno, Czech Republic
#
# Redistribution and use in source code form, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# 3. Redistributions must be pursued only for non-commercial research
#    collaboration and demonstration purposes.
#
# 4. Where separate files retain their original licence terms
#    (e.g. MPL 2.0, Apache licence), these licence terms are announced, prevail
#    these terms and must be complied.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import pose_estimation.FUtil as FUtil

from skimage.util.shape import view_as_windows
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import gzip
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
import os
import re
from timeit import default_timer as timer
from tqdm import tqdm
import skimage

import cv2
cv2.setNumThreads(0)


def loadDepth(img_name):
    img = []
    with gzip.open(img_name, 'rb') as input:
        for line in input.readlines():
            img.append(np.fromstring(line, sep=' '))
    return np.array(img)


def savePointCloudToPly(pt_3d, pt_c, filename):
    pt_3d_c = np.concatenate((pt_3d, pt_c), axis=1)
    pt_3d_ct = [tuple(l) for l in pt_3d_c.tolist()]
    pt_3d_ct = np.array(pt_3d_ct, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('red', 'u1'),
                                         ('green', 'u1'),
                                         ('blue', 'u1')])
    el = PlyElement.describe(pt_3d_ct, 'vertex')
    PlyData([el], text=True).write(filename)


def unproject(coords_2d, depth, modelview, projection):
    """Unprojects 2d image coordinates to 3D based on a depth map.
    @coords_2d [numpy.array] of shape Nx2, where coordinates are stored
    as Nx(x,y)
    @depth [numpy.array] is the depth map of the source image. Must have the
    same size as the source image.
    @param modelview numpy array 4x4 of GL modelview matrix to define
    transformation from 3D to camera coordinates.
    @param projection numpy array 4x4 of GL projection matrix to define
    transformation from camera coordinates to image coordinates.

    @return [np.array] Nx4 array of 3D points in homogeneous coordinates (w=1).
    """

    h = depth.shape[0]
    w = depth.shape[1]
    intr = FUtil.projectiveToIntrinsics(projection, w, h)
    intr_inv = np.linalg.inv(intr)
    mv_inv = np.linalg.inv(modelview)

    #coords_2d = np.array([[400, 300]])

    z1 = np.ones((coords_2d.shape[0], 1))
    c2d = coords_2d.astype(int)
    depths = (depth[c2d[:, 1], c2d[:, 0]]).reshape(-1, 1)
    #print("depths")
    #print(depths)
    depths = np.hstack([depths, depths, depths])
    coords_im = np.hstack([c2d, z1])
    coords_im = (coords_im * depths)
    #print("coords im depths")
    #print(coords_im)
    coords_3d = (np.dot(intr_inv, coords_im.transpose()))
    #print("intr inv")
    #print(coords_3d)
    coords_3d = np.concatenate((coords_3d,
                                np.ones((1, coords_3d.shape[1]))), axis=0)
    coords_3d = np.dot(mv_inv, coords_3d).transpose()
    #print(coords_3d)
    #c = np.random.randint(0, 255, (coords_3d.shape[0], 3))
    #savePointCloudToPly(coords_3d[:, :3], c, "mat_desktop.ply")

    return coords_3d


def projectWithIntrinsics(pt_4d, h, w, modelview, intr, radial=np.array([0, 0])):
    pt_3d = np.dot(modelview, pt_4d.transpose())
    pt_3d = pt_3d / pt_3d[3, :]
    pt_3d = pt_3d[:3, :]
    pt_2d = np.dot(intr, pt_3d)
    pt_2d = (pt_2d[:2, :] / pt_2d[2, :]).transpose()

    # radial distortion
    pt_2d = pt_2d + intr[:2, 2] #translate to optical center
    pt_2d = pt_2d / np.diag(intr)[:2] #divide by focal length

    k1 = radial[0]
    k2 = radial[1]
    n = np.sum(pt_2d**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    pt_2d = pt_2d * r[:, None]

    pt_2d = pt_2d * np.diag(intr)[:2] #multiply by focal
    pt_2d = pt_2d - intr[:2, 2] #translate back

    return pt_2d

def projectWithIntrinsicsAndZ(pt_4d, h, w, modelview, intr, radial=np.array([0, 0])):
    pt_3d = np.dot(modelview, pt_4d.transpose())
    pt_3d = pt_3d / pt_3d[3, :]
    pt_3d = pt_3d[:3, :]
    pt_2d = np.dot(intr, pt_3d)
    pt_2d = (pt_2d[:2, :] / pt_2d[2, :]).transpose()

    # radial distortion
    pt_2d = pt_2d + intr[:2, 2] #translate to optical center
    pt_2d = pt_2d / np.diag(intr)[:2] #divide by focal length

    k1 = radial[0]
    k2 = radial[1]
    n = np.sum(pt_2d**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    pt_2d = pt_2d * r[:, None]

    pt_2d = pt_2d * np.diag(intr)[:2] #multiply by focal
    pt_2d = pt_2d - intr[:2, 2] #translate back

    return pt_2d, pt_3d[2, :]


def project(pt_4d, h, w, modelview, projection):
    intr = FUtil.projectiveToIntrinsics(projection, w, h)
    pt_3d = np.dot(modelview, pt_4d.transpose())
    pt_3d = pt_3d / pt_3d[3, :]
    pt_3d = pt_3d[:3, :]
    pt_2d = np.dot(intr, pt_3d)
    pt_2d = (pt_2d / pt_2d[2, :]).transpose()
    return pt_2d[:, :2]


def unproject_image(img, depth, modelview, projection):
    """ Unprojects image coordinates to 3D based on GL modelview and projection
        matrices and depth in meters.
        @param img numpy array HxWx3 of image to be unprojected. Image shape is
        used to determine local image coordinatesself.
        @param depth numpy array HxWx1 with depth stored in meters.
        @param modelview numpy array 4x4 of GL modelview matrix to define
        transformation from 3D to camera coordinates.
        @param projection numpy array 4x4 of GL projection matrix to define
        transformation from camera coordinates to image coordinates.

        return  coords_3d numpy array of shape (H*W)x3 of 3D points
        corresponding to 2D pixels in the image.
                colors pixel colors corresponding to each 3D point taken from
                the img.
    """
    h = img.shape[0]
    w = img.shape[1]

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    x1, y1 = np.meshgrid(x, y)
    coords_2d = np.hstack([x1.reshape((w * h), -1), y1.reshape((w * h), -1)])
    coords_3d = unproject(coords_2d, depth, modelview, projection)
    return coords_3d, img.reshape((h * w, 3))


def findIndicesOfCorresponding3DPoints(img1_3d, img2_3d, threshold=10):
    """ Calculates which pixel in img1 corresponds to in img2 based on
    their distance in 3D.
    @return indices of 3D points in view1 and in view2.
    """
    # select 3D points which are in both views
    kdt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(img1_3d)
    dist, nn_idx = kdt.kneighbors(img2_3d)

    v2_idx = np.arange(0, img2_3d.shape[0])
    v2_idx = v2_idx[dist[:, 0] < threshold]
    v1_idx = (nn_idx[v2_idx])[:, 0]

    return v1_idx, v2_idx


def findIndicesOfCorresponding3DPointsWithDist(img1_3d, img2_3d, threshold=10):
    """ Calculates which pixel in img1 corresponds to in img2 based on
    their distance in 3D.
    @return indices of 3D points in view1 and in view2.
    """
    # select 3D points which are in both views
    kdt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(img1_3d)
    dist, nn_idx = kdt.kneighbors(img2_3d)

    v2_idx = np.arange(0, img2_3d.shape[0])
    sel = dist[:, 0] < threshold
    v2_idx = v2_idx[sel]
    v1_idx = (nn_idx[v2_idx])[:, 0]

    return v1_idx, v2_idx, dist[sel, 0]


def genRandomCorrespondingPoints2D(img1_shape, img2_shape,
                                   fov_1, fov_2, v1_idx, v2_idx,
                                   radius=32, N=100, maxres=4096):
    h1, w1 = img1_shape[0:2]
    h2, w2 = img2_shape[0:2]

    v1_idx = v1_idx
    v2_idx = v2_idx

    j1 = (v1_idx / w1).astype(int)
    j2 = (v2_idx / w2).astype(int)

    i1 = (v1_idx - (j1 * w1))
    i2 = (v2_idx - (j2 * w2))

    # store coords 2D before scaling fov
    coords1 = np.concatenate((j1.reshape(-1, 1), i1.reshape(-1, 1)), axis=1)
    coords2 = np.concatenate((j2.reshape(-1, 1), i2.reshape(-1, 1)), axis=1)

    # scale coords according to fov just to select correct random patches
    nw_1 = (fov_1 / np.pi) * maxres
    scale1 = (nw_1 / w1)
    nh_1 = int(scale1 * h1)
    nw_1 = int(nw_1)

    nw_2 = (fov_2 / np.pi) * maxres
    scale2 = (nw_2 / w2)
    nh_2 = int(scale2 * h2)
    nw_2 = int(nw_2)

    j1 = (scale1 * j1).astype(int)
    i1 = (scale1 * i1).astype(int)
    j2 = (scale2 * j2).astype(int)
    i2 = (scale2 * i2).astype(int)

    # select keypoints inside padding by radius
    seli = np.all([i1 > radius, i1 < nw_1 - radius,
                   i2 > radius, i2 < nw_2 - radius], axis=0)
    selj = np.all([j1 > radius, j1 < nh_1 - radius,
                   j2 > radius, j2 < nh_2 - radius], axis=0)
    sel = np.all([seli, selj], axis=0)
    i1 = i1[sel]
    i2 = i2[sel]
    j1 = j1[sel]
    j2 = j2[sel]

    fin_idx = np.arange(coords1.shape[0])
    fin_idx = fin_idx[sel]

    # select N random points
    sel = np.arange(i1.shape[0])

    np.random.shuffle(sel)
    sel = sel[0:N]

    fin_idx = fin_idx[sel]

    return coords1[fin_idx], coords2[fin_idx], fin_idx


# @profile
def generatePatchesFast(img1, img2, img1_shape, img2_shape, coords1, coords2,
                        fov_1, fov_2,
                        show=False, radius=32, radius2=32,
                        maxres=4096, maxres2=4096, needles=0):

    orig_maxres1 = maxres
    orig_maxres2 = maxres2

    all_patches1 = []
    all_patches2 = []
    pad_mode = 'constant' if needles <= 0 else 'edge'

    if needles > 0:
        img1_orig = img1.copy()
        img2_orig = img2.copy()
    else:
        img1_orig = img1
        img2_orig = img2

    for i in range(0, needles + 1):
        j1 = coords1[:, 0].copy()
        i1 = coords1[:, 1].copy()
        j2 = coords2[:, 0].copy()
        i2 = coords2[:, 1].copy()

        if needles > 0:
            maxres = (orig_maxres1 / needles) * (i + 1)
            maxres2 = (orig_maxres2 / needles) * (i + 1)

        # resize images and indices to unit FOV so that extracted patches have
        # the same scale
        h1, w1 = img1_shape[0:2]
        h2, w2 = img2_shape[0:2]

        nw_1 = (fov_1 / np.pi) * maxres
        scale1 = (nw_1 / w1)
        nh_1 = int(scale1 * h1)
        nw_1 = int(nw_1)

        nw_2 = (fov_2 / np.pi) * maxres2
        scale2 = (nw_2 / w2)
        nh_2 = int(scale2 * h2)
        nw_2 = int(nw_2)
        K = int(radius)
        K2 = int(radius2)

        if not img1.shape[1] != nw_1 or img1.shape[0] != nh_1:
            img1 = cv2.resize(img1_orig, (nw_1, nh_1), interpolation=cv2.INTER_AREA)
        if not img2.shape[1] != nw_2 or img2.shape[0] != nh_2:
            img2 = cv2.resize(img2_orig, (nw_2, nh_2), interpolation=cv2.INTER_AREA)

        img1 = np.pad(img1, ((K, K), (K, K), (0, 0)), pad_mode)
        img2 = np.pad(img2, ((K2, K2), (K2, K2), (0, 0)), pad_mode)

        j1 = (scale1 * j1).astype(int)
        i1 = (scale1 * i1).astype(int)
        j2 = (scale2 * j2).astype(int)
        i2 = (scale2 * i2).astype(int)
        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]

        indices_1 = np.vstack((j1 + K, i1 + K))
        indices_2 = np.vstack((j2 + K2, i2 + K2))

        img1ch = img1.shape[2]
        img2ch = img2.shape[2]
        radius = int(radius)
        radius2 = int(radius2)
        radius_2 = int(radius * 2)

        radius2_2 = int(radius2 * 2)

        patches_1 = view_as_windows(
            img1, (radius_2, radius_2, img1ch)
        )[indices_1[0] - radius, indices_1[1] - radius, 0, ...]

        patches_2 = view_as_windows(
            img2, (radius2_2, radius2_2, img2ch)
        )[indices_2[0] - radius2, indices_2[1] - radius2, 0, ...]

        patches_1 = np.ascontiguousarray(
            patches_1.transpose((0, 3, 1, 2))
        ).astype(np.float32)
        patches_2 = np.ascontiguousarray(
            patches_2.transpose((0, 3, 1, 2))
        ).astype(np.float32)
        # patches_1[:, :3, :, :] = patches_1[:, :3, :, :] / 255.0
        # patches_2[:, :3, :, :] = patches_2[:, :3, :, :] / 255.0

        all_patches1.append(patches_1)
        all_patches2.append(patches_2)

    all_patches1 = np.concatenate(all_patches1, axis=1)
    all_patches2 = np.concatenate(all_patches2, axis=1)
    return all_patches1, all_patches2


def getSizeFOV(w, h, fov_1, maxres=4096):
    nw = (fov_1 / np.pi) * maxres
    scale1 = (nw / w)
    nh = int(scale1 * h)
    nw = int(nw)
    return nw, nh, scale1


def generatePatchesFastImg(img1, img1_shape, coords1,
                           fov_1, show=False, radius=32, maxres=4096,
                           needles=0):
    orig_maxres1 = maxres

    all_patches1 = []
    pad_mode = 'constant' if needles <= 0 else 'edge'

    if needles > 0:
        img1_orig = img1.copy()
    else:
        img1_orig = img1

    for i in range(0, needles + 1):
        j1 = coords1[:, 0].copy()
        i1 = coords1[:, 1].copy()

        if needles > 0:
            maxres = (orig_maxres1 / needles) * (i + 1)

        # resize images and indices to unit FOV so that extracted patches have the
        # same scale
        h1, w1 = img1_shape[0:2]

        nw_1, nh_1, scale1 = getSizeFOV(w1, h1, fov_1, maxres)

        K = int(radius)

        if not img1.shape[1] != nw_1 or img1.shape[0] != nh_1:
            if img1_orig.shape[2] <= 3:
                # opencv resize seems to work better, but is limited to 3
                # channels
                img1 = cv2.resize(img1_orig, (nw_1, nh_1), interpolation=cv2.INTER_AREA)
            else:
                # skimage resize works for arbitrary number of channels, but
                # performs worse with our method.
                img1 = skimage.transform.resize(img1_orig, (nh_1, nw_1), anti_aliasing=True)

        img1 = np.pad(img1, ((K, K), (K, K), (0, 0)), pad_mode)

        j1 = (scale1 * j1).astype(int)
        i1 = (scale1 * i1).astype(int)
        h1, w1 = img1.shape[0:2]

        radius_2 = int(radius * 2)

        indices_1 = np.vstack((j1 + K, i1 + K))

        img1ch = img1.shape[2]
        radius_2 = radius * 2
        patches_1 = view_as_windows(img1, (radius_2, radius_2, img1ch))[indices_1[0]-radius, indices_1[1]-radius,0,...]
        patches_1 = np.ascontiguousarray(patches_1.transpose((0,3,1,2))).astype(np.float32)

        all_patches1.append(patches_1)

    all_patches1 = np.concatenate(all_patches1, axis=1)
    all_patches1[:, :(3 * (needles + 1))] = all_patches1[:, :(3 * (needles + 1))] / 255.0

    return all_patches1


def generatePatchesFastImgNoscale(img1, img1_shape, coords1,
                                  show=False, radius=32, maxres=4096):
    j1 = coords1[:, 0]
    i1 = coords1[:, 1]

    # resize images and indices to unit FOV so that extracted patches have the
    # same scale
    h1, w1 = img1_shape[0:2]

    K = int(radius)

    img1 = np.pad(img1, ((K, K), (K, K), (0, 0)), 'constant')

    j1 = (j1).astype(int)
    i1 = (i1).astype(int)
    h1, w1 = img1.shape[0:2]

    radius_2 = int(radius * 2)

    indices_1 = np.vstack((j1 + K, i1 + K))

    img1ch = img1.shape[2]
    radius_2 = radius * 2
    patches_1 = view_as_windows(img1, (radius_2, radius_2, img1ch))[indices_1[0]-radius, indices_1[1]-radius,0,...]
    patches_1 = np.ascontiguousarray(patches_1.transpose((0,3,1,2))).astype(np.float32) / 255.0

    return patches_1


def generatePatchesImgScale(img1, img1_shape, coords1, diameters,
                       show=False, patch_radius=32, maxres=4096):

    j1 = coords1[:, 0]
    i1 = coords1[:, 1]

    radii = np.floor((diameters / 2.0) * patch_radius).astype(int)
    # radii[radii > 256] = 256
    radius = int(np.ceil(np.max(radii)))
    # pad the image with the largest radius
    img1 = np.pad(
        img1, ((radius, radius), (radius, radius), (0, 0)), 'constant'
    )

    j1 = (j1).astype(int) + radius
    i1 = (i1).astype(int) + radius

    h1, w1 = img1.shape[0:2]

    if show:
        colors = np.random.rand(i1.shape[0], 3)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.scatter(i1[:100], j1[:100], c=colors[:100])

    patches_stack_1 = []
    patch_radius_2 = patch_radius * 2

    for idx in range(0, i1.shape[0]):
        radius = radii[idx]
        radius_2_sq = radius * radius * 4
        y1l = int(max(j1[idx] - radius, 0))
        y1u = int(min(j1[idx] + radius, h1))
        x1l = int(max(i1[idx] - radius, 0))
        x1u = int(min(i1[idx] + radius, w1))

        patch1 = np.copy(img1[y1l:y1u, x1l:x1u, :])

        p1wh = patch1.shape[0] * patch1.shape[1]
        if (p1wh != radius_2_sq):
            continue

        patch1 = cv2.resize(patch1, (patch_radius_2, patch_radius_2))
        patches_stack_1.append(patch1)

    patches_stack_1 = np.array(patches_stack_1).transpose((0,3,1,2)).astype(np.float32) / 255.0
    return patches_stack_1


def generatePatchesImg(img1, img1_shape, coords1, fov_1,
                       show=False, radius=32, maxres=4096):


    j1 = coords1[:, 0]
    i1 = coords1[:, 1]
    N = coords1.shape[0]

    # resize images and indices to unit FOV so that extracted patches have the
    # same scale
    h1, w1 = img1_shape[0:2]

    nw_1 = (fov_1 / np.pi) * maxres
    scale1 = (nw_1 / w1)
    nh_1 = int(scale1 * h1)
    nw_1 = int(nw_1)
    #radius = int((1.0 / scale1) * radius)
    #scale1 = 1.0
    #print("maxres", maxres)

    img1 = transform.resize(img1, (nh_1, nw_1), mode='constant') #transform makes image in range 0,1 float!!!
    img1 = np.pad(img1, ((radius,radius),(radius,radius),(0,0)), 'constant')

    j1 = (scale1 * j1).astype(int) + radius
    i1 = (scale1 * i1).astype(int) + radius

    h1, w1 = img1.shape[0:2]

    if show:
        colors = np.random.rand(i1.shape[0], 3)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.scatter(i1, j1, c=colors)

    radius_2 = int(radius * 2)
    side = int(np.sqrt(N))
    side_px = int(side * radius_2)
    patches1 = np.zeros((side_px, side_px, 3), dtype=np.uint8)

    radius_2_sq = radius_2 * radius_2

    patches_stack_1 = []

    for idx in range(0, i1.shape[0]):
        py = int(idx / side)
        px = int(idx - (py * side))
        px = px * radius_2
        py = py * radius_2
        y1l = int(max(j1[idx] - radius, 0))
        y1u = int(min(j1[idx] + radius, h1))
        x1l = int(max(i1[idx] - radius, 0))
        x1u = int(min(i1[idx] + radius, w1))

        patch1 = img1[y1l:y1u, x1l:x1u, :]

        p1wh = patch1.shape[0] * patch1.shape[1]
        if (p1wh != radius_2_sq):
            continue

        #if show:
            #print(idx, px, px+radius_2, py, py+radius_2)
            #patches1[py:py+radius_2, px:px+radius_2, :] = patch1
        patches_stack_1.append(patch1)

    i1 = i1.reshape(-1, 1)
    j1 = j1.reshape(-1, 1)

    return patches_stack_1


#@profile
def generatePatches(img1, img2, img1_shape, img2_shape, coords1, coords2,
                    fov_1, fov_2,
                    show=False, radius=32, maxres=4096):
    """Generates patches from original images. Input are original images, which
    will be scaled according to their FOV so that extraxted patches have the
    same scale. Image size used for reprojecting the pixels to 3D are needed
    (img1_shape, img2_shape), so that corresponding pixel coordinates might be
    calculated accordingly from pixel indices v1_idx, v2_idx.
    @param img1 original dataset image 1
    @param img2 original dataset image 2 (second view)
    @param img1_shape shape (size) of the image to which image indices v1_idx,
           v2_idx are pointing to.
    @param img2_shape shape (size) of the image to which image indices v1_idx,
          v2_idx are pointing to.
    @param v1_idx indices of corresponding points in img1 of shape img1_shape.
    @param v2_idx indices of corresponding points in img2 of shape img2_shape.
    @param P1 OpenGL projection 4x4 matrix of img1
    @param P2 OpemGL projection 4x4 matrix of img2
    @param show True when the patches and selected points shall be plotted,
           default=False
    @param radius radius of the patch, patch side is 2 * radius
    @param N number of calculated patches, shall be N^2
    @param maxres resolution of image with FOV = pi. The input images will be
    rescaled based on their FOV taken from projection matrices P1, P2.

    @return patches1 patches from img1 represented by numpy array
    of shape (N, radius*2, radius*2, 3)
    @return patches2 patches from img1 represented by numpy array
            of shape (N, radius*2, radius*2, 3)
    @return sel indices to v1_idx, v2_idx of selected points.
    """
    j1 = coords1[:, 0]
    i1 = coords1[:, 1]
    j2 = coords2[:, 0]
    i2 = coords2[:, 1]
    N = coords1.shape[0]

    # resize images and indices to unit FOV so that extracted patches have the
    # same scale
    h1, w1 = img1_shape[0:2]
    h2, w2 = img2_shape[0:2]

    nw_1 = (fov_1 / np.pi) * maxres
    scale1 = (nw_1 / w1)
    nh_1 = int(scale1 * h1)
    nw_1 = int(nw_1)

    nw_2 = (fov_2 / np.pi) * maxres
    scale2 = (nw_2 / w2)
    nh_2 = int(scale2 * h2)
    nw_2 = int(nw_2)
    K = int(radius)

    img1 = cv2.resize(img1, (nw_1, nh_1), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (nw_2, nh_2), interpolation=cv2.INTER_AREA)

    img1 = np.pad(img1, ((K, K),(K, K),(0,0)), 'constant')
    img2 = np.pad(img2, ((K, K),(K, K),(0,0)), 'constant')
    j1 = (scale1 * j1).astype(int) + K
    i1 = (scale1 * i1).astype(int) + K
    j2 = (scale2 * j2).astype(int) + K
    i2 = (scale2 * i2).astype(int) + K
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    if show:
        colors = np.random.rand(i1.shape[0], 3)
        plt.figure(dpi=600)
        #plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.scatter(i1, j1, c=colors, s=1.0)
        plt.axis('off')

        plt.figure(dpi=600)
        #plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.scatter(i2, j2, c=colors, s=1.0)
        plt.axis('off')

    radius_2 = int(radius * 2)
    side = int(np.sqrt(N))
    side_px = int(side * radius_2)
    patches1 = np.zeros((side_px, side_px, 3), dtype=np.uint8)
    patches2 = np.zeros((side_px, side_px, 3), dtype=np.uint8)

    radius_2_sq = radius_2 * radius_2

    patches_stack_1 = []
    patches_stack_2 = []

    for idx in range(0, i1.shape[0]):
        py = int(idx / side)
        px = int(idx - (py * side))
        px = px * radius_2
        py = py * radius_2
        y1l = int(max(j1[idx] - radius, 0))
        y1u = int(min(j1[idx] + radius, h1))
        x1l = int(max(i1[idx] - radius, 0))
        x1u = int(min(i1[idx] + radius, w1))

        y2l = int(max(j2[idx] - radius, 0))
        y2u = int(min(j2[idx] + radius, h2))
        x2l = int(max(i2[idx] - radius, 0))
        x2u = int(min(i2[idx] + radius, w2))

        patch1 = img1[y1l:y1u, x1l:x1u, :]
        patch2 = img2[y2l:y2u, x2l:x2u, :]

        p1wh = patch1.shape[0] * patch1.shape[1]
        p2wh = patch2.shape[0] * patch2.shape[1]
        #if (p1wh != radius_2_sq or p2wh != radius_2_sq):
        #    continue

        if show:
            #print(idx, px, px+radius_2, py, py+radius_2)
            patches1[py:py+radius_2, px:px+radius_2, :] = patch1
            patches2[py:py+radius_2, px:px+radius_2, :] = patch2

        patches_stack_1.append(patch1.astype(np.float32) / 255.0)
        patches_stack_2.append(patch2.astype(np.float32) / 255.0)

    if show:
        plt.figure(dpi=600)
        plt.subplot(1, 2, 1)
        plt.imshow(patches1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(patches2)
        plt.axis('off')
        #plt.show()

    i1 = i1.reshape(-1, 1)
    j1 = j1.reshape(-1, 1)
    i2 = i2.reshape(-1, 1)
    j2 = j2.reshape(-1, 1)

    #print("generating patches done")
    patches_stack_1 = np.ascontiguousarray(np.asarray(patches_stack_1).transpose((0,3,1,2)))
    patches_stack_2 = np.ascontiguousarray(np.asarray(patches_stack_2).transpose((0,3,1,2)))
    return patches_stack_1, patches_stack_2


def calculateDistances3D(img1_3d, img2_3d, v1_idx, v2_idx, sel):
    """Calculates distances of corresponding points in 3D.
    """

    v1_3d = img1_3d[v1_idx[sel]]
    v2_3d = img2_3d[v2_idx[sel]]

    # calculate 3D distance of each point to each point
    dist = np.linalg.norm(v1_3d[:, None] - v2_3d, axis=2)
    # dist_idx = np.argsort(dist, axis=1)
    # dist_sorted = np.sort(dist, axis=1)
    # for i in range(0, dist_idx.shape[0]):
    #     row_dist_idx = dist_idx[i, dist_sorted[i, :] > pos_thr]
    #     print(row_dist_idx[:N])
    #     print(dist_sorted[i, :N])
    #     all_dist_idx.append(row_dist_idx[:N])
    # res = np.array(all_dist_idx)
    # return res
    return dist


def showNegatives(patches1, patches2, negatives,
                  img1, img2, v1_idx, v2_idx, sel):
    """Plots negatives for each patch."""

    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    j1 = (v1_idx[sel] / w1)
    j2 = (v2_idx[sel] / w2)

    i1 = (v1_idx[sel] - (j1 * w1))
    i2 = (v2_idx[sel] - (j2 * w2))

    plt.figure()
    for i in range(0, negatives.shape[0]):
        numneg = len(negatives[i])
        if (numneg > 0):
            plt.subplot(1, numneg + 2, 1)
            plt.imshow(patches1[i])
            plt.subplot(1, numneg + 2, 2)
            plt.imshow(patches2[i])
            for j in range(0, numneg):
                plt.subplot(1, numneg + 2, j + 3)
                plt.imshow(patches2[negatives[i, j]])

            colors = np.random.rand(i1.shape[0], 3)
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img1)
            plt.scatter(i1[i], j1[i], c='red')

            plt.subplot(1, 2, 2)
            plt.imshow(img2)
            plt.scatter(i2[negatives[i]], j2[negatives[i]],
                        c=colors)
            plt.scatter(i2[i], j2[i], c='red')

            plt.show()


def getPatches(img1_name, img2_name, show=False, max_w=1024,
               patch_size=64, npatches=100, measure=False):

    print("getPatches, ", img1_name, img2_name)
    img1_base, img1_ext = os.path.splitext(img1_name)
    img2_base, img2_ext = os.path.splitext(img2_name)

    img1_base = re.sub("_texture", "", img1_base)
    img2_base = re.sub("_texture", "", img2_base)

    img1_depth_name = img1_base + "_texture_depth.txt.gz"
    img2_depth_name = img2_base + "_texture_depth.txt.gz"

    MV1_name = img1_base + "_modelview.txt"
    MV2_name = img2_base + "_modelview.txt"

    P1_name = img1_base + "_projection.txt"
    P2_name = img2_base + "_projection.txt"

    RT1 = (FUtil.loadMatrixFromFile(MV1_name))
    RT2 = (FUtil.loadMatrixFromFile(MV2_name))

    P1 = FUtil.loadMatrixFromFile(P1_name)
    P2 = FUtil.loadMatrixFromFile(P2_name)

    img1_depth = loadDepth(img1_depth_name)
    img2_depth = loadDepth(img2_depth_name)

    img_1_depth_wrong = (np.min(img1_depth) == np.max(img1_depth))
    img_2_depth_wrong = (np.min(img2_depth) == np.max(img2_depth))
    if img_1_depth_wrong or img_2_depth_wrong:
        msg = "One of the two images has empty depth map."
        raise RuntimeError(msg)


    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)

    img1 = np.flip(img1, 2)
    img2 = np.flip(img2, 2)

    img1_aspect = float(img1.shape[0]) / img1.shape[1]
    img2_aspect = float(img2.shape[0]) / img2.shape[1]

    h1 = int(img1_aspect * max_w)
    h2 = int(img2_aspect * max_w)

    img1_resized = cv2.resize(img1, (max_w, h1))
    img2_resized = cv2.resize(img2, (max_w, h2))
    img1_depth = cv2.resize(img1_depth, (max_w, h1))
    img2_depth = cv2.resize(img2_depth, (max_w, h2))

    img1_3d, imgv1 = unproject_image(img1_resized, img1_depth, RT1, P1)
    img2_3d, imgv2 = unproject_image(img2_resized, img2_depth, RT2, P2)

    if measure:
        savePointCloudToPly(img1_3d[:, :3], imgv1[:, :],
                             'model_v1.ply')
        savePointCloudToPly(img2_3d[:, :3], imgv2[:, :],
                             'model_v2.ply')

    v1_idx, v2_idx = findIndicesOfCorresponding3DPoints(img1_3d, img2_3d)

    if v1_idx.shape[0] <= 0 or v2_idx.shape[0] <= 0:
        msg = "These two images do not contain any corresponding points."
        raise RuntimeError(msg)

    fov_1, fovy_1 = FUtil.projectiveToFOV(P1)
    fov_2, fovy_2 = FUtil.projectiveToFOV(P2)

    patch_radius = patch_size / 2
    coords1, coords2, sel = genRandomCorrespondingPoints2D(img1_resized.shape,
                                                           img2_resized.shape,
                                                           fov_1, fov_2,
                                                           v1_idx, v2_idx,
                                                           radius=patch_radius,
                                                           N=npatches)

    if measure:
        numiter = 1
        start = timer()
        for i in tqdm(range(0, numiter)):
            patches_1, patches_2 = generatePatchesFast(img1, img2, img1_resized.shape,
                                                   img2_resized.shape, coords1,
                                                   coords2, fov_1, fov_2, show,
                                                   patch_radius)
        end = timer()
        elapsed = end - start
        print("Total runtime patches fast: ", elapsed, "one iteration: ", elapsed / numiter)

        # grid1 = utils.make_grid(torch.from_numpy(patches_1).float() / 255.0, nrow=30)
        # grid2 = utils.make_grid(torch.from_numpy(patches_2).float() / 255.0, nrow=30)
        # fig = plt.figure(dpi=600)
        # plt.subplot(1,2,1)
        # plt.imshow(grid1.numpy().transpose((1, 2, 0)))
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.imshow(grid2.numpy().transpose((1, 2, 0)))
        # plt.axis('off')

        start = timer()
        for i in tqdm(range(0, numiter)):
            patches_1, patches_2 = generatePatches(img1, img2, img1_resized.shape,
                                                   img2_resized.shape, coords1,
                                                   coords2, fov_1, fov_2, show,
                                                   patch_radius)
        end = timer()
        elapsed = end - start
        print("Total runtime patches loop: ", elapsed, "one iteration: ", elapsed / numiter)

        # grid3 = utils.make_grid(torch.from_numpy(patches_1).float(), nrow=30)
        # grid4 = utils.make_grid(torch.from_numpy(patches_2).float(), nrow=30)
        # fig2 = plt.figure(dpi=600)
        # plt.subplot(1,2,1)
        # plt.imshow(grid3.numpy().transpose((1, 2, 0)))
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.imshow(grid4.numpy().transpose((1, 2, 0)))
        # plt.axis('off')
        # plt.show()
        # plt.close(fig)
        # plt.close(fig2)


    # dists = calculateDistances3D(img1_3d, img2_3d, v1_idx, v2_idx, sel)

    info = {'coords2d_1': coords1, 'coords2d_2': coords2,
            'coords3d_1': img1_3d[v1_idx[sel]],
            'coords3d_2': img2_3d[v2_idx[sel]],
            'img1_shape': img1_resized.shape, 'img2_shape': img2_resized.shape,
            'patch_radius': patch_radius, 'fov_1': fov_1, 'fov_2': fov_2,
            'img1_name': os.path.basename(img1_name),
            'img2_name': os.path.basename(img2_name)}

    # test if projected points correspond well to image coords
    # img1_2 = project(img1_3d[v1_idx[sel]], h2, max_w, RT2, P2)
    # img1_1 = project(img1_3d[v1_idx[sel]], h1, max_w, RT1, P1)
    # img2_1 = project(img2_3d[v2_idx[sel]], h1, max_w, RT1, P1)
    # img2_2 = project(img2_3d[v2_idx[sel]], h2, max_w, RT2, P2)
    # #
    # plt.figure()
    # plt.imshow(img2_resized)
    # plt.scatter(img1_2[:, 0], img1_2[:, 1], color='orange', s=1.0)
    # plt.scatter(img2_2[:, 0], img2_2[:, 1], color='blue', s=1.0)
    # plt.scatter(coords2[:, 1], coords2[:, 0], color='red', s=1.0)
    #
    # plt.figure()
    # plt.imshow(img1_resized)
    # plt.scatter(img2_1[:, 0], img2_1[:, 1], color='orange', s=1.0)
    # plt.scatter(img1_1[:, 0], img1_1[:, 1], color='blue', s=1.0)
    # plt.scatter(coords1[:, 1], coords1[:, 0], color='red', s=1.0)
    #plt.show()

    #showNegatives(patches_1, patches_2, negatives,
    #              img1_resized, img2_resized, v1_idx, v2_idx, sel)

    # join both point clouds  together for visualization
    # both_3d = np.concatenate((img1_3d[:, :3], img2_3d[:, :3]), axis=0)
    # both_c = np.concatenate((imgv1, imgv2), axis=0)
    # savePointCloudToPly(both_3d, both_c, 'model_real.ply')

    # save filtered 3D points
    if measure:
        savePointCloudToPly(img1_3d[v1_idx, :3], imgv1[v1_idx, :],
                            'model_v1_filtered.ply')
        savePointCloudToPly(img2_3d[v2_idx, :3], imgv2[v2_idx, :],
                            'model_v2_filtered.ply')

    #return patches_1, patches_2, info
    return info


if __name__ == "__main__":
    #img1_name = sys.argv[1]
    #img2_name = sys.argv[2]
    img1_name = "/mnt/matylda1/ibrejcha/adobe_intern/data/switzerland_wallis_30km_maximg/final_dataset/real/10036966853_349ae666d8_b.jpg"
    img2_name = "/mnt/matylda1/ibrejcha/adobe_intern/data/switzerland_wallis_30km_maximg/final_dataset/real/10049279564_7f2989c3b5_b_texture.jpg"

    #img1_name = "pose_estimation/10036966853_349ae666d8_b.jpg"
    #img2_name = "pose_estimation/10049279564_7f2989c3b5_b_texture.png"

    #img1_name = "/Users/janbrejcha/data/switzerland_wallis_30km_maximg_sample/final_dataset/real/10072177735_d857e011e5_b.png"
    #img2_name = "/Users/janbrejcha/data/switzerland_wallis_30km_maximg_sample/final_dataset/real/10072185566_47cc975c92_b_texture.png"

    info = getPatches(img1_name, img2_name, True, 512, 64, 100, measure=True)
