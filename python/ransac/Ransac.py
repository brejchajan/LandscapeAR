# @Date:   2020-08-04T10:43:33+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:48:51+02:00
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


class Ransac(object):

    def __init__(self, estimator, n_pts, t_dist, min_inliers=10,
                 max_iterations=100000):
        self.estimator = estimator
        self.n_pts = n_pts
        self.t_dist = t_dist
        self.min_inliers = np.max(np.array([min_inliers, n_pts]) + 1)
        self.max_iterations = max_iterations

    def run(self, data, probability=0.99):
        iter = 0
        best_model = None
        best_model_inliers_idx = 0
        best_model_inliers = 0
        best_loss = 1e100
        K = 1e100

        while K > iter and iter < self.max_iterations:
            numpts = data.shape[0]
            sel_idx = np.random.permutation(numpts)[:self.n_pts]
            models = self.estimator.estimate(data[sel_idx])
            if models is None:
                continue

            # choose best model out from all current solutions
            for model_idx in range(0, len(models)):
                residuals = self.estimator.computeResiduals(
                    models[model_idx], data
                )
                inliers_sel = residuals < self.t_dist
                inliers_idx = np.where(inliers_sel)[0]
                loss = np.mean(residuals[inliers_idx])
                inliers = np.sum(inliers_sel)
                if inliers > self.min_inliers and inliers > best_model_inliers:
                    # and loss < best_loss:
                    best_model_inliers_idx = inliers_idx
                    best_model = models[model_idx]
                    best_model_inliers = inliers
                    best_loss = loss
                w = best_model_inliers / numpts
                K = (np.log(1 - probability)
                     / np.log(1 - np.power(w + 1.0/numpts, self.n_pts)))
                if K < 0:
                    K = 1e100
                # print("K", K, "bm inliers", best_model_inliers,
                #      inliers, loss, best_loss)

            iter += 1
        print("RANSAC finished after ", iter, "steps.")
        res = True
        if best_model is None:
            res = False
        return res, best_model, best_model_inliers_idx, best_loss


def samplePointsOnLineGaussianNoise(A, B, npts, sigma):
    u = (B - A).reshape(-1, 1)
    A = A.reshape(-1, 1)
    t = np.random.uniform(0, 1, npts)
    gaussian_noise = sigma * np.random.randn(2, npts)
    pts = (A + u * t) + gaussian_noise
    return pts.transpose()


def testLineFit():
    import matplotlib.pyplot as plt

    # generate data
    num_inliers = 100
    num_outliers = 2000
    min = 0
    max = 10
    A = np.array([1, 8])
    B = np.array([10, 2])
    pts = samplePointsOnLineGaussianNoise(A, B, num_inliers, 0.05)
    outliers = np.random.uniform(min, max, [2, num_outliers]).transpose()
    allpts = np.vstack([pts, outliers])
    plt.figure()
    print(pts.shape)
    # plt.scatter(allpts[:, 0], allpts[:, 1], c='green', s=0.5)
    plt.scatter(pts[:, 0], pts[:, 1], c='green', s=0.5)
    plt.scatter(outliers[:, 0], outliers[:, 1], c='blue', s=0.5)

    # fit the model using ransac
    from LineEstimator import LineEstimator
    estimator = LineEstimator()
    ransac = Ransac(estimator, 2, 0.05, 0)
    res, best_model, best_model_inliers_idx, loss = ransac.run(allpts, 0.9999)
    print("best model: ", best_model, "num inliers: ",
          best_model_inliers_idx.shape, "loss", loss)
    print("best_model 0", best_model[0])
    print("best_model 1", best_model[1])
    Af = best_model[0]
    Bf = best_model[1]
    plt.plot([Af[0], Bf[0]], [Af[1], Bf[1]], c='r')
    plt.show()


if __name__ == "__main__":
    testLineFit()
