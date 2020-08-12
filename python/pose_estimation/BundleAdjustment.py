# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-12T11:43:08+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:43:21+02:00
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




import numpy as np
from scipy.sparse import lil_matrix
import cv2
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    points_proj[:, 0] = -points_proj[:, 0]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, points_3d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    #points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    #points_2d = params[n_cameras * 9:].reshape((n_observations, 2))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    residual = (points_proj - points_2d).ravel()
    return residual

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def bundleAdjustment(cam_poses, cam_intr, pt_3d, pt_2d, cam_ind, pt_3d_ind, show=False):
    n_cameras = len(cam_poses)
    n_points = pt_3d.shape[0]
    n_observations = pt_2d.shape[0]
    camera_params = []
    for idx in range(0, n_cameras):
        pose = cam_poses[idx]
        intr = cam_intr[idx]
        rotvec = cv2.Rodrigues(pose[:3, :3], jacobian=0)[0].reshape(3)
        x1 = rotvec
        x2 = pose[:3, 3]
        x3 = np.array([intr[0, 0]])
        x4 = np.array([0])
        x5 = np.array([0])
        cam_param = np.concatenate([x1, x2, x3, x4, x5])
        camera_params.append(cam_param)
    camera_params = np.concatenate(camera_params)
    print(camera_params.shape)

    #x0 = np.hstack((camera_params.ravel(), pt_3d.ravel()))
    #x0 = camera_params.ravel()
    x0 = np.hstack((camera_params.ravel()))
    f0 = fun(x0, n_cameras, n_points, cam_ind, pt_3d_ind, pt_2d, pt_3d)

    #A = BundleAdjustment.bundle_adjustment_sparsity(n_cameras, n_points, cam_ind, pt_3d_ind)
    t0 = time.time()
    res = least_squares(fun, x0, verbose=2, jac='3-point', x_scale='jac', method='trf', #ftol=1e-4
                        args=(n_cameras, n_points, cam_ind, pt_3d_ind, pt_2d, pt_3d))
    t1 = time.time()

    res_rotvec = res.x[:3]
    res_t = res.x[3:6]
    intr = res.x[6:9]
    res_R = cv2.Rodrigues(res_rotvec, jacobian=0)[0]

    if show:
        print("Optimization took {0:.0f} seconds".format(t1 - t0))
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(f0)

        plt.subplot(2,1,2)
        plt.plot(res.fun)
        #plt.show()
    return res_R, res_t, intr, res.success
