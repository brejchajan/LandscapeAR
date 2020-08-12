# @Date:   2020-08-06T16:24:58+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:46:49+02:00
# @License: Copyright 2020 Brno University of Technology,
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import numpy as np

# our code
import pose_estimation.FUtil as FUtil
from pose_estimation.patchSamplingDepth import unproject

from ransac.Ransac import Ransac
from ransac.EPNPEstimator import EPNPEstimator
from pypnp.P4PfEstimator import P4PfEstimator
import pose_estimation.BundleAdjustment as BundleAdjustment

def poseEPNPBAIterative(R, t, fov, mask, photo_shape, coords2d, coords3d, reprojection_error=10):
    num_inliers = 0
    cnt_same = 0
    while(True):
        if cnt_same > 5:
            break
        # bundle adjustment
        sel = mask[:, 0] == 1
        pose = np.ones([4, 4])
        pose[:3, :3] = R
        pose[:3, 3] = t
        intr1 = FUtil.fovToIntrinsics(fov, photo_shape[1], photo_shape[0])
        left_cam_idx = np.zeros(coords2d[sel].shape[0]).astype(np.int)
        left_cam_pt_idx = np.arange(0, coords2d[sel].shape[0]).astype(np.int)
        points2d = coords2d - np.array([[photo_shape[1] / 2.0, photo_shape[0] / 2.0]])
        res_R, res_t, res_intr, success = BundleAdjustment.bundleAdjustment([pose], [intr1], coords3d[sel], points2d[sel], left_cam_idx, left_cam_pt_idx, show=False)
        print("res intr BA", res_intr)
        if success:
            cnt_same_wrong = 0
            # update FOV with refined estimate
            fov = 2 * np.arctan2(photo_shape[1] / 2.0, res_intr[0])
            print("estim FOV BA:", (fov * 180) / np.pi)
            print("R BA:", res_R)
            print("t BA:", res_t)

            # measure number of inliers by runnong another
            # RANSAC round
            ret, R_epnp, t_epnp, mask_epnp = poseFrom2D3DWithFOV(coords2d, fov, photo_shape, coords3d, reprojection_error)
            new_num_inliers = np.sum(mask)
            print("EPNP R", R)
            print("EPNP t", t)

            print("EPNP num inliers", new_num_inliers)
            # count number of iterations number of inliers is the same
            if (new_num_inliers == num_inliers):
                cnt_same += 1
            else:
                # if the number of inliers increases or decreases, reset counter
                cnt_same = 0
            if new_num_inliers >= num_inliers:
                num_inliers = new_num_inliers
                R = R_epnp
                t = t_epnp
                mask = mask_epnp
            else:
                break
        else:
            break
    return R, t, fov, mask


def solveP4PfRansac(coords3d, src_pts, reprojection_error=10):
    '''
        Wrapper to the Ransac solver with P4Pf estimator so that we can call
        it the same way as we call openCV cv2.solvePnPRansac.
    '''
    R = None
    t = None
    f = None
    best_model_inliers_idx = None

    data = np.hstack([src_pts, coords3d])
    estimator = P4PfEstimator()
    ransac = Ransac(estimator, 4, reprojection_error)
    res, best_model, best_model_inliers_idx, loss = ransac.run(data, 0.99)

    if res:
        R = best_model[:9].reshape(3, 3)
        t = best_model[9:12]
        f = best_model[12]

    return res, R, t, f, best_model_inliers_idx


def solveEPNPRansac(coords3d, src_pts, intr, reprojection_error=10):
    '''
        Wrapper to the Ransac solver with P4Pf estimator so that we can call
        it the same way as we call openCV cv2.solvePnPRansac.
    '''
    R = None
    t = None
    best_model_inliers_idx = None

    data = np.hstack([src_pts, coords3d])
    estimator = EPNPEstimator(intr)
    ransac = Ransac(estimator, 5, reprojection_error)
    res, best_model, best_model_inliers_idx, loss = ransac.run(data, 0.99)

    if res:
        R = best_model[:9].reshape(3, 3)
        t = best_model[9:12]
    return res, R, t, best_model_inliers_idx


def poseFrom2D3DP4Pf(src_pts, coords_3d, reprojection_error=10):
    res, R, t, f, inliers_idx = solveP4PfRansac(
        coords_3d, src_pts, reprojection_error=reprojection_error
    )
    mask = np.zeros((src_pts.shape[0], 1)).astype(int)
    if res:
        R[1:3, :] = -R[1:3, :]
        t[1:3] = -t[1:3]
        mask[inliers_idx] = int(1)
    return res, R, t, f, mask


def poseFrom2D3DWithFOVEPNPOurRansac(src_pts, fov, img1_shape, coords3d, reprojection_error=10):
    """Estimates the pose using 2D-3D correspondences, EPnP algorithm and
       our implementation of RANSAC in python.
       src_pts, fov [radians] correspond to left view, which should be
                              the photgraph
       dst_pts, intr2, and img2_depth correspond to the rendered image.
       @type array @param src_pts 2D points in the photograph
       @type array @param projection1 OpenGL projection matrix of the camera
                          which took the photo.
       @type array @param dst_pts 2D points in the rendered image
       @type array @param projection2 OpenGL projection matrix of the camera
                          which rendered the synthetic image.
       @type array @param MV2 modelview matrix of the camera which rendered the
                          synthetic image.
       @type array @param img2_depth depth map of the rendered image.
    """
    intr1 = FUtil.fovToIntrinsics(fov, img1_shape[1], img1_shape[0])
    intr1[:, 1:3] = -intr1[:, 1:3]
    ret, R, t, inliers_idx = solveEPNPRansac(
        coords3d, src_pts, intr1, reprojection_error=reprojection_error
    )
    mask = np.zeros((src_pts.shape[0], 1)).astype(int)
    if ret:
        R[1:3, :] = -R[1:3, :]
        t[1:3] = -t[1:3]
        mask[inliers_idx] = int(1)

    return ret, R, t, mask


def poseFrom2D3DWithFOV(src_pts, fov, img1_shape, coords3d, reprojection_error=10, iterationsCount=100000, pnp_flags=cv2.SOLVEPNP_EPNP, use_ransac=True):
    """Estimates the pose using 2D-3D correspondences and PnP algorithm.
       src_pts, fov [radians] correspond to left view, which should be
                              the photgraph
       dst_pts, intr2, and img2_depth correspond to the rendered image.
       @type array @param src_pts 2D points in the photograph
       @type array @param projection1 OpenGL projection matrix of the camera
                          which took the photo.
       @type array @param dst_pts 2D points in the rendered image
       @type array @param projection2 OpenGL projection matrix of the camera
                          which rendered the synthetic image.
       @type array @param MV2 modelview matrix of the camera which rendered the
                          synthetic image.
       @type array @param img2_depth depth map of the rendered image.
    """
    print("fov", fov)
    print("src pts", src_pts.shape, coords3d.shape)
    intr1 = FUtil.fovToIntrinsics(fov, img1_shape[1], img1_shape[0])
    intr1[:, 1:3] = -intr1[:, 1:3]
    dist = np.array([])
    if use_ransac:
        ret, Rvec, t, inliers = cv2.solvePnPRansac(coords3d, src_pts, intr1, dist, reprojectionError=reprojection_error, iterationsCount=iterationsCount, flags=pnp_flags) #,
    else:
        inliers = np.ones((src_pts.shape[0], 1)).astype(np.int)
        ret, Rvec, t = cv2.solvePnP(coords3d, src_pts, intr1, dist, flags=pnp_flags) #,
    R = cv2.Rodrigues(Rvec, jacobian=0)[0]
    R[1:3, :] = -R[1:3, :]
    t[1:3] = -t[1:3]
    t = t.reshape(-1)
    mask = np.zeros((src_pts.shape[0], 1)).astype(int)
    if ret:
        mask[inliers[:, 0]] = int(1)
    return ret, R, t, mask

def poseFrom2D3D(src_pts, projection1, img1_shape,
                 dst_pts, projection2, MV2, img2_depth, pnp_flags=cv2.SOLVEPNP_EPNP):
    """Estimates the pose using 2D-3D correspondences and PnP algorithm.
       src_pts, intr1 correspond to left view, which should be the photgraph
       dst_pts, intr2, and img2_depth correspond to the rendered image.
       @type array @param src_pts 2D points in the photograph
       @type array @param projection1 OpenGL projection matrix of the camera
                          which took the photo.
       @type array @param dst_pts 2D points in the rendered image
       @type array @param projection2 OpenGL projection matrix of the camera
                          which rendered the synthetic image.
       @type array @param MV2 modelview matrix of the camera which rendered the
                          synthetic image.
       @type array @param img2_depth depth map of the rendered image.
    """
    coords3d = unproject(dst_pts, img2_depth, MV2, projection2)[:, :3]
    intr1 = FUtil.projectiveToIntrinsics(projection1, img1_shape[1],
                                         img1_shape[0])
    intr1[:, 1:3] = -intr1[:, 1:3]
    dist = np.array([])
    ret, Rvec, t, inliers = cv2.solvePnPRansac(coords3d, src_pts, intr1, dist, reprojectionError=10, iterationsCount=100000, flags=pnp_flags) #,
    R = cv2.Rodrigues(Rvec, jacobian=0)[0]
    R[1:3, :] = -R[1:3, :]
    t[1:3] = -t[1:3]
    t = t.reshape(-1)
    mask = np.zeros((src_pts.shape[0], 1))

    if ret:
        mask[inliers[:, 0]] = 1
    return ret, R, t, mask


def poseFromMatches(good, intr1, intr2, src_pts, dst_pts):
    # add one to third dimension
    src_pts = np.concatenate((src_pts, np.ones((src_pts.shape[0], 1))), axis=1)
    dst_pts = np.concatenate((dst_pts, np.ones((src_pts.shape[0], 1))), axis=1)

    # transfrom the points to unit plane
    # (calibrated camera with unit camera matrix)
    src_pts = np.dot(np.linalg.inv(intr1), src_pts.transpose()).transpose()
    src_pts = np.dot(intr2, src_pts.transpose()).transpose()
    # dst_pts = np.dot(np.linalg.inv(intr2), dst_pts.transpose()).transpose()
    # print(src_pts, np.min(src_pts[:, :2]), np.max(src_pts[:, :2]))
    # print("intrinsics", intr1)

    F, mask = cv2.findFundamentalMat(src_pts[:, :2], dst_pts[:, :2],
                                     cv2.FM_RANSAC, 3.0)

    E = np.dot(np.dot(intr2.transpose(), F), intr2)
    # print(F, E, mask)
    src_recover = src_pts[mask[:, -1].astype(np.bool), :2]
    dst_recover = dst_pts[mask[:, -1].astype(np.bool), :2]

    ngood = []
    for i in range(0, mask.shape[0]):
        if mask[i] != 0:
            g = good[i]
            ngood.append(g)
    num_inliers, R, t, mask = cv2.recoverPose(E, src_recover, dst_recover,
                                              intr2)
    return R, t, mask, ngood
