# @Date:   2020-08-11T17:46:12+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:46:34+02:00
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
import numpy.linalg
import math

def vec(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);

def inZero2Pi(a):
    while (a < 0):
        a += 2 * math.pi
    while (a >= 2 * math.pi):
        a -= 2 * math.pi
    return a

def rotationZ(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotationY(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotationX(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])

def angles(R):
    alpha = 0
    beta = 0
    gamma = 0
    v1 = np.matrix([[0,0,1]]).transpose()
    A = -vec(R.dot(v1), v1)
    if (np.linalg.norm(A) < 1e-15):
        gamma = 0
    else:
        A = np.linalg.inv(R).dot(A.reshape(3,1))
        gamma = -np.arctan2(-A.item(0), A.item(1))
    gamma = inZero2Pi(gamma)
    R = R.dot(rotationZ(-gamma))
    A = np.linalg.inv(R).dot(np.matrix([[0,0,1]]).transpose())
    beta = -np.arctan2(A.item(0), A.item(2))
    beta = inZero2Pi(beta)
    R = R.dot(rotationY(-beta))

    A = R.dot(np.array([[1,0,0]]).transpose())
    alpha = np.arctan2(A.item(1), A.item(0))
    alpha = inZero2Pi(alpha)
    return alpha, beta, gamma

def matrix(alpha, beta, gamma):
    return rotationZ(alpha) * rotationY(beta) * rotationZ(gamma)
