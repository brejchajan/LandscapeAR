# @Date:   2020-08-04T10:42:48+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:48:41+02:00
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
from p4pf import p4pf

# P4P + unknown focal length testing app
# given a set of 4x 2D<->3D correspondences, calculate camera pose and the
# camera focal length.
#
# rewritten by Jan Brejcha (Jun 2020) from the
# original Matlab code by Martin Bujnak, (c)apr2008
#


# ground truth focal, image resolution [-0.5;0.5]x[-0.5;0.5]
fgt = 1.5

# 4x 2D - 3D correspondences
if 0:
    # planar
    M = np.array([ -4.16909046323065, -21.76924382754407, -11.28160103570576,  -18.70673643641682,
           8.76573361388879,   2.54916911753407, -15.64179144355760,   -7.12884738225938,
                  0,                  0,                  0,                  0]).reshape(3, 4)

    m =  np.array([ -0.01685480263817,  -0.30156127000652,  -0.11261034380081,    -0.25658711285962,
           -0.05066846248374,   0.01061608945225,   0.17825476446826,     0.09480347286640]).reshape(2, 4)

    # ground truth orientation + position
    Rgt = np.array(
            [0.99441452977752,  -0.10554498077766,                  0,
             -0.05095105295313,  -0.48004620390991,  -0.87576231496602,
             0.09243231669889,   0.87087077063381,  -0.48274254803710]
          ).reshape(3, 3)

    Tgt = np.array([  3.98991000654439,  0.74564356260581,  88.96209555860348])

else:
    # general scene
    M =  np.array([ -3.33639834336120, -23.35549638285873, -13.18941519576778,  6.43164913914748,
            0.65948286155096,  -4.90376715918747,   1.17103701629876,  0.14580433383203,
           -8.46658219501120,  -3.99876939947909,  -3.02248927651177,  -22.16086539862748]).reshape(3, 4)

    m = np.array([ 0.11009888473695,   0.39776592879400,   0.28752996253253,   -0.05017617729940,
          0.03882059658299,  -0.17303453640632,  -0.05791310109713,    0.19297848817239]).reshape(2, 4)

    # ground truth orientation + position
    Rgt = np.array([ -0.94382954756954,   0.33043272406750,                 0,
             0.27314119331857,   0.78018522420862,  -0.56276540799790,
            -0.18595610677571,  -0.53115462041845,  -0.82661665574857]).reshape(3, 3)

    Tgt = np.array([ 2.85817409844554, -2.17296255562889, 77.54246130780075])

# type "python3 setup.py build_ext --inplace" to compile mex helper file

R, t, f = p4pf(m, M)

# solutions test
for i in range(0, len(f)):
    Rrel = np.dot(np.linalg.inv(Rgt), R[i])
    Trel = Tgt - t[i]
    trace = np.minimum((np.trace(Rrel) - 1 ) / 2, 1.0)
    dangle = np.linalg.norm(np.arccos(trace)) / np.pi * 180

    # print errors
    print('focal err: {}   rotation err: {}   translation err: {}'.format((fgt-f[i]), dangle, np.linalg.norm(Trel)))
