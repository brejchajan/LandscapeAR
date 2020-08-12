# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:47:28+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:44:19+02:00
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
from pypnp.p4pf import p4pf
from pose_estimation.patchSamplingDepth import projectWithIntrinsics


class P4PfEstimator(object):
    def __init__(self):
        pass

    def estimate(self, data):
        numpts = data.shape[0]
        if (numpts != 4):
            raise RuntimeError("Invalid number of points sampled for P4P.")
        m2D = data[:, 0:2].transpose().astype(np.float64)
        m3D = data[:, 2:5].transpose().astype(np.float64)
        R, t, f = p4pf(m2D, m3D)
        models = []
        if f is None:
            return None
        for i in range(0, len(f)):
            RR = R[i]
            tt = t[i]
            #RR[1:3, :] = -RR[1:3, :]
            #tt[1:3] = -tt[1:3]
            model = np.hstack([RR.reshape(-1), tt.reshape(-1), f[i]])
            models.append(model)
            #print("estimated model ", R[i], t[i], f[i])
        return tuple(models)

    def computeResiduals(self, model, data):
        m2D = data[:, 0:2]
        m3D = data[:, 2:5]


        # retrieve model params
        R = model[:9].reshape(3, 3)
        t = model[9:12]
        f = model[12]

        # project 3D points to 2D
        p4d = np.concatenate((m3D, np.ones((m3D.shape[0], 1))), axis=1)
        MV = np.ones((4, 4))
        MV[:3, :3] = R
        MV[:3, 3] = t

        intr = K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
        m2D_p = projectWithIntrinsics(p4d, 0, 0, MV, intr)

        residuals = np.linalg.norm(m2D - m2D_p, axis=1)
        return residuals
