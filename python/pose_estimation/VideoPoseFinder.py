# @Date:   2020-08-08T17:06:11+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:47:06+02:00
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
import os
import pymap3d
import numpy as np
import shutil

# our code
import pose_estimation.PoseFinder as fp
from pose_estimation import FUtil
import pose_estimation.eulerZYZ as eulerZYZ


class PoseFinderArgs(object):
    pass


class VideoPoseFinder(object):
    def __init__(self, video_path, initial_gps, fov, working_directory, snapshot, model, earth_file, log_dir, cuda=True, normalize_output=True):
        self.video_path = video_path
        self.initial_gps = initial_gps
        self.fov = fov
        self.working_directory = working_directory
        self.video_frames_dir = os.path.join(self.working_directory, "video_frames")
        if not os.path.isdir(self.video_frames_dir):
            os.makedirs(self.video_frames_dir)

        self.video_output_dir = os.path.join(self.working_directory, "video_output")
        if not os.path.isdir(self.video_output_dir):
            os.makedirs(self.video_output_dir)

        self.snapshot = snapshot
        self.model = model
        self.log_dir = log_dir
        self.normalize_output = normalize_output
        self.cuda = cuda
        self.earth_file = earth_file
        self.video = cv2.VideoCapture(self.video_path)
        self.curr_gps = self.initial_gps
        self.framenum = 0

        cmd = "itr --pose-from-gps " + str(self.initial_gps[0]) + " " + str(self.initial_gps[1]) + " initcam " + self.earth_file
        os.system(cmd)

        init_pose = FUtil.loadMatrixFromFile("initcam_modelview.txt")
        R = init_pose[:3, :3]
        self.scene_center = np.dot(-R.transpose(), init_pose[:3, 3])
        self.curr_center = np.zeros((3, 1))

        success, image = self.video.read()
        frame_base = "frame%d" % self.framenum
        frame_name = frame_base + ".jpg"
        frame_name = os.path.join(self.video_frames_dir, frame_name)
        if not os.path.isfile(frame_name):
            cv2.imwrite(frame_name, image)

        pfargs = self.buildPFArgs(frame_name, self.curr_gps, self.fov)
        self.poseFinder = fp.PoseFinder(pfargs)

        self.KF = self.initKalmanFilter()


    def buildPFArgs(self, query_image, gps, fov):
        args = PoseFinderArgs()
        args.query_image = query_image
        args.snapshot = self.snapshot
        args.working_directory = self.working_directory
        args.earth_file = self.earth_file
        args.log_dir = self.log_dir
        args.model_name = self.model
        args.cuda = self.cuda
        args.normalize_output = self.normalize_output
        args.grid_options = [0, 1]
        args.no_voting = False
        args.gps = gps
        args.best_buddy_refine = True
        args.voting_cnt = 5
        args.use_depth = False
        args.use_normals = False
        args.use_silhouettes = False
        args.fov = fov
        self.matching_dir = "matching_" + args.snapshot + "_" + args.model_name
        args.matching_dir = self.matching_dir
        args.use_hardnet = False
        args.maxres = 4096
        args.fcn_keypoints = False
        args.fcn_keypoints_multiscale = False
        args.d2net = False
        args.ncnet = False
        args.dense_uniform_keypoints = False
        args.dense_halton_keypoints = False
        args.stride = None
        return args


    def processVideo(self):
        success = True
        while success:
            success = self.processNextFrame()

    def initKalmanFilter(self):
        # see https://docs.opencv.org/3.3.0/dc/d2c/tutorial_real_time_pose.html
        fps = self.video.get(cv2.CAP_PROP_FPS)
        dt = 1.0/fps
        KF = cv2.KalmanFilter(18, 6, 0, cv2.CV_64F)
        print("process noise cov", KF.processNoiseCov.shape)
        print("measurement noise cov", KF.measurementNoiseCov.shape)
        print("error cov post", KF.errorCovPost.shape)
        cv2.setIdentity(KF.processNoiseCov, 2000)
        cv2.setIdentity(KF.measurementNoiseCov, 2000)
        cv2.setIdentity(KF.errorCovPost, 2000)

        # position
        KF.transitionMatrix = np.eye(18)
        KF.transitionMatrix[0, 3] = dt
        KF.transitionMatrix[1, 4] = dt
        KF.transitionMatrix[2, 5] = dt
        KF.transitionMatrix[3, 6] = dt
        KF.transitionMatrix[4, 7] = dt
        KF.transitionMatrix[5, 8] = dt
        KF.transitionMatrix[0, 6] = 0.5 * dt * dt;
        KF.transitionMatrix[1, 7] = 0.5 * dt * dt;
        KF.transitionMatrix[2, 8] = 0.5 * dt * dt;

        #orientation
        KF.transitionMatrix[9, 12] = dt
        KF.transitionMatrix[10, 13] = dt
        KF.transitionMatrix[11, 14] = dt
        KF.transitionMatrix[12, 15] = dt
        KF.transitionMatrix[13, 16] = dt
        KF.transitionMatrix[14, 17] = dt
        KF.transitionMatrix[9, 15] =  0.5 * dt * dt;
        KF.transitionMatrix[10, 16] =  0.5 * dt * dt;
        KF.transitionMatrix[11, 17] =  0.5 * dt * dt;

        # measurement model
        KF.measurementMatrix = np.zeros(KF.measurementMatrix.shape)
        KF.measurementMatrix[0, 0] = 1
        KF.measurementMatrix[1, 1] = 1
        KF.measurementMatrix[2, 2] = 1
        KF.measurementMatrix[3, 9] = 1
        KF.measurementMatrix[4, 10] = 1
        KF.measurementMatrix[5, 11] = 1

        return KF


    def getKalmanMeasurement(self, R, t):
        a, b, g = eulerZYZ.angles(R)
        return np.array([[t[0], t[1], t[2], a, b, g]]).transpose()

    def updateKalmanFilter(self, measurement):
        pred = self.KF.predict()
        est = self.KF.correct(measurement)
        #print("pred", measurement, pred, est)
        t_est = np.array([est[0, 0], est[1, 0], est[2, 0]])
        #print("t est", t_est)
        rot_est = np.array(eulerZYZ.matrix(est[9, 0], est[10, 0], est[11, 0]))
        #rot_est[1:3] = -rot_est[1:3]
        #print("rot est", est[9, 0], est[10, 0], est[11, 0])
        return rot_est, t_est


    def processNextFrame(self):

        success, image = self.video.read()
        frame_base = "frame%d" % self.framenum
        frame_name = frame_base + ".jpg"
        frame_name = os.path.join(self.video_frames_dir, frame_name)
        if not os.path.isfile(frame_name):
            cv2.imwrite(frame_name, image)
        self.poseFinder.query_img_path = frame_name
        self.poseFinder.gps = self.curr_gps

        res, pose, center = self.poseFinder.findPose()
        print("got pose:", pose, center)
        if (res):
            # pose successfull, get camera center
            R = pose[:3, :3]
            C = np.dot(-R.transpose(), pose[:3, 3]) + center
            C = C - self.scene_center
            t = np.dot(-R, C)

            # no kalmann filter
            R_est = R
            C_est = C
            t_est = t
            #print("t", t)
            #measurement = self.getKalmanMeasurement(R, t)
            #R_est, t_est = self.updateKalmanFilter(measurement)
            #C_est = np.dot(-R_est.transpose(), t_est)
            dist = np.linalg.norm(self.curr_center - C_est)
            C_est_glob = C_est + self.scene_center
            lat, lon, alt = pymap3d.ecef2geodetic(C_est_glob[0], C_est_glob[1], C_est_glob[2])
            C_est_glob -= center
            t_est_glob = np.dot(-R_est, C_est_glob)

            #save estimated pose as a new bestpose
            bestpose_filename = os.path.join(self.working_directory, self.matching_dir, frame_base, frame_base + "_bestpose_kalman_pose.txt")
            photo_estimated_MV = np.zeros((4,4))
            photo_estimated_MV[3, 3] = 1
            photo_estimated_MV[:3, :3] = R_est
            photo_estimated_MV[:3, 3] = t_est_glob
            #print("kalman mv", photo_estimated_MV)
            #np.savetxt(bestpose_filename, photo_estimated_MV)

            #re-export to NVM using poseFinder
            #self.poseFinder.findPose()

            render = False
            if render:
                #render found pose
                if not os.path.isfile(os.path.join(self.video_output_dir, 'fixed', "image_frame%d.png" % self.framenum)):
                    scene_path = os.path.join(self.working_directory, self.matching_dir, frame_base, "nvm_export")
                    cmd = "itr --egl 0 --directory " + scene_path + " --render-photoviews 1024" + " --output " + self.video_output_dir + " " + self.earth_file
                    os.system(cmd)

            print("Distance from previous frame:", dist)
            if dist > 2000 and dist < 10000:
                #update current center
                self.curr_center = C_est
                #we moved significantly, update the position for rendering
                self.curr_gps = [lat, lon, alt]
            elif dist <= 2000 or dist >= 10000:
                # move render of current frame so that it is not rendered again
                print("Render moved to a new frame.")
                shutil.move(os.path.join(self.working_directory, frame_base), os.path.join(self.working_directory, "frame%d" % (self.framenum + 1)))

        self.framenum += 1
        return success
