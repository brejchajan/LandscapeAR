# @Date:   2020-08-06T16:16:58+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:48:46+02:00
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

import numpy as np
import cv2
from pypnp.p4pf import p4pf
from pose_estimation.patchSamplingDepth import projectWithIntrinsics



class EPNPEstimator(object):
    def __init__(self, intr):
        self.intr = intr

    def estimate(self, data):
        numpts = data.shape[0]
        if (numpts != 5):
            raise RuntimeError("Invalid number of points sampled for P4P.")
        m2D = data[:, 0:2].reshape(-1, 1, 2).astype(np.float32)
        m3D = data[:, 2:5].reshape(-1, 3).astype(np.float32)
        dist = np.array([])
        ret, Rvec, t = cv2.solvePnP(m3D, m2D, self.intr, dist, flags=cv2.SOLVEPNP_EPNP)
        models = []
        if not ret:
            return None
        R = cv2.Rodrigues(Rvec, jacobian=0)[0]
        t = t.reshape(-1)

        model = np.hstack([R.reshape(-1), t.reshape(-1)])
        models.append(model)
        return tuple(models)

    def computeResiduals(self, model, data):
        m2D = data[:, 0:2]
        m3D = data[:, 2:5]

        # retrieve model params
        R = model[:9].reshape(3, 3)
        t = model[9:12]

        # project 3D points to 2D
        p4d = np.concatenate((m3D, np.ones((m3D.shape[0], 1))), axis=1)
        MV = np.ones((4, 4))
        MV[:3, :3] = R
        MV[:3, 3] = t

        m2D_p = projectWithIntrinsics(p4d, 0, 0, MV, self.intr)

        residuals = np.linalg.norm(m2D - m2D_p, axis=1)
        return residuals
