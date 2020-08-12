# @Date:   2020-08-04T10:43:33+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:48:49+02:00
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


class LineEstimator(object):
    def __init__(self):
        pass

    def estimate(self, data):
        if data.shape[0] > 2:
            raise RuntimeError("Obtained non minimal sample.")
        # parametric line, u is directional vector, A is a point, p = A + tu,
        # t is a real parameter.
        A = data[0]
        B = data[1]
        u = B - A
        i = 0
        #if np.linalg.norm(u) < 0.0001:
        #    raise RuntimeError("Wrongly sampled points. Two points cannot be the same.")
        #    return
        while np.linalg.norm(B - A) < 1:
            B = A + u * 10 * i
            i += 1

        return tuple([np.array([A, B])])

    def computeResiduals(self, model, data):
        # compute the distance of a point from a line
        eps = 1e-10
        p1 = model[0]
        p2 = model[1]
        p3 = data
        residuals = np.abs(np.cross(p2 - p1, p3 - p1) / (np.linalg.norm(p2 - p1)))
        return residuals
