# @Date:   2020-08-04T10:37:16+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:46:43+02:00
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


def loadMatrixFromFile(filename):
    """Loads numpy matrix from text file."""

    with open(filename, "r") as f:
        lines = f.readlines()
        mat = []
        line_arr = []
        for line in lines:
            for mat_line in line.strip().split("\n"):
                for x in mat_line.split(" "):
                    try:
                        num = float(x)
                        line_arr.append(num)
                        if (len(line_arr) == 4):
                            mat.append(line_arr)
                            line_arr = []
                    except ValueError:
                        pass
    return np.array(mat)


def fovToIntrinsics(fov, width, height):
    #fovy = (fov / width) * height

    cx = (width / 2.0)
    cy = (height / 2.0)

    fx = cx / np.tan(fov / 2.0)
    #fy = cy / np.tan(fovy / 2.0)
    K = np.array([[fx, -0, -cx], [0, -fx, -cy], [0, -0, -1]])
    return K


def intrinsicsToFov(intr):
    fx = intr[0, 0]
    cx = intr[0, 2]
    fov = np.arctan2(cx, fx) * 2.0
    if fov < 0:
        fov = -fov
    return fov


def projectiveToIntrinsics(Proj, width, height):
    """Converts OpenGL projective matrix created by gluPerspective to
    corresponding intrinsics matrix K.
    @param Proj projective matrix created by openGL function gluPerspective
    @param width image width in pixels
    @param height image height in pixels
    @return 3x3 intrinsic matrix in the same coordinate system as the input
    Proj matrix (is compatible with modelview.)
    """

    f_w = Proj[0, 0]
    f_h = Proj[1, 1]

    cx = (width / 2.0)
    cy = (height / 2.0)
    fx = ((width / 2.0) * f_w)
    fy = ((height / 2.0) * f_h)

    K = np.array([[fx, -0, -cx], [0, -fy, -cy], [0, -0, -1]])
    return K


def projectiveToFOV(Proj):
    """Converts OpenGL projective matrix created by
    gluPerspective to FOV and FOVY.
    @param Proj projective matrix created by openGL function gluPerspective
    @return tuple FOV, FOVY in radians."""
    # arccotg(x) = pi/2 - arctg(x)
    fov = 2.0 * ((np.pi / 2.0) - np.arctan(Proj[0, 0]))
    fovy = 2.0 * ((np.pi / 2.0) - np.arctan(Proj[1, 1]))
    return fov, fovy


def getRotScale(M):
    """Assuming M is 3x3 matrix"""
    sx = np.linalg.norm(M[:3, 0])
    sy = np.linalg.norm(M[:3, 1])
    sz = np.linalg.norm(M[:3, 2])
    scale = np.array([sx, sy, sz])
    scale_M = np.tile(scale, [3, 1])
    R = M/scale_M
    return R, scale


def fromTwoCameras(P1, P2, K1, K2):
    """Calculates Fundamental matrix from two relative camera poses P1, P2
    and intrinsics K1, K2."""
    # calculate P2 relative to P1
    #P1 = np.linalg.inv(P1)
    #P2 = np.linalg.inv(P2)
    #tmp = P1
    #P1 = P2
    #P2 = tmp

    R1 = P1[:3, :3]
    R2 = P2[:3, :3]
    R = np.dot(np.transpose(R2), R1)
    t = np.dot(np.transpose(R2), P1[:3, 3] - P2[:3, 3])


    # according to Hartley and Zissermann eq. 9.4, p. 241
    #epipole e
    F1 = np.dot(np.dot(np.linalg.inv(K2).transpose(), R), K1.transpose())
    e = np.dot(np.dot(K1, R.transpose()), t)
    ex = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    F = np.dot(F1, ex)

    #e1 = np.dot(K2, t2)
    #epipole e'
    #e = np.dot(K2, t2)
    #ex = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    #F1 = np.dot(np.dot(K2, R2), np.linalg.inv(K1))
    #F = np.dot(ex, F1)

    F = F / F[2, 2]
    return F
