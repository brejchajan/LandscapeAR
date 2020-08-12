# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:51:49+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:47:12+02:00
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

import argparse as ap
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import exiftool
import datetime

# our code
from pose_estimation.PoseFinder import PoseFinder
from pose_estimation import FUtil
from trainPatchesDescriptors import calculateErrors
from training.MultimodalPatchesDataset import MultimodalPatchesDataset

def buildArgumentParser():
    parser = ap.ArgumentParser()
    parser.add_argument("directory", help="The directory containing renders \
                    and matched results in subdirectory \
                    \"matching (default, override by --matching_dir_name)\". \
                    These can be calculated using the findPose.py script.")
    parser.add_argument("dataset_dir", help="The directory that contains the \
                    dataset of rendered photoviews created by \
                    itr.")
    parser.add_argument("--matching_dir_name", help="Name of the directory \
                        containing matched images inside the <directory>.",
                        default="matching")
    parser.add_argument("--original_images", help="directory containing the \
                        original photographs. This will be used to get the \
                        original timestamps.")
    return parser


def movePoseToCenter(pose, center):
    R = pose[:3, :3]
    t = np.dot(-R.transpose(), pose[:3, 3]) + center
    pose[:3, 3] = t
    return pose


def getGPSPose(image_name, dataset_dir):
    dataset_path = os.path.join(dataset_dir, "real_gps")
    if os.path.isdir(dataset_path):
        pose_name = os.path.join(dataset_path, image_name + "_modelview.txt")
        pose = FUtil.loadMatrixFromFile(pose_name)
        R = pose[:3, :3]
        t = pose[:3, 3]
        center = np.dot(-R.transpose(), t)
        pose[:3, 3] = center
        return pose
    return None


def getRenderedPose(image_name, result_dir):
    rendered_dir = os.path.join(result_dir, image_name)
    pose_name = os.path.join(rendered_dir, "*_0.000000_pose.txt")
    pose_files = glob.glob(pose_name)

    if not pose_files:
        return None, None

    if not os.path.isfile(pose_files[0]):
        raise RuntimeError("Unable to find rendered pose: " + pose_files[0])
    scene_info_name = os.path.join(rendered_dir, "scene_info.txt")
    center = MultimodalPatchesDataset.getSceneCenter(scene_info_name)
    pose = FUtil.loadMatrixFromFile(pose_files[0])
    pose = movePoseToCenter(pose, center)
    return pose, center


def getGroundTruthPose(image_name, dataset_dir):
    dataset_path = os.path.join(dataset_dir, "real")
    pose_name = os.path.join(dataset_path, image_name + "_modelview.txt")
    projection_name = os.path.join(dataset_path, image_name + "_projection.txt")
    img_name = os.path.join(dataset_path, image_name + ".png")
    if not os.path.isfile(img_name):
        img_name = os.path.join(dataset_path, image_name + ".jpg")
    if not os.path.isfile(img_name):
        raise RuntimeError("Unable to find the query image.", image_name)

    scene_info_name = os.path.join(dataset_dir, "scene_info.txt")

    pose = FUtil.loadMatrixFromFile(pose_name)

    if not os.path.isfile(scene_info_name):
        R = pose[:3, :3]
        t = pose[:3, 3]
        center = np.dot(-R.transpose(), t)
        pose[:3, 3] = pose[:3, 3] - np.dot(-R, center)
        #print("Loaded center from pose, pose: ", pose, "center", center)
    else:
        center = MultimodalPatchesDataset.getSceneCenter(scene_info_name)

    pose = movePoseToCenter(pose, center)
    projection = FUtil.loadMatrixFromFile(projection_name)
    img = cv2.imread(img_name)
    intrinsics = FUtil.projectiveToIntrinsics(projection, img.shape[1], img.shape[0])
    return pose, intrinsics


def getPoseCenter(pose):
    pose_R = pose[:3, :3]
    pose_t = pose[:3, 3]
    pose_C = np.dot(-pose_R.transpose(), pose_t)
    return pose_C


def calculatePosErrAlongCameraAxis(gt_pos, est_pose, est_intr):
    """ est_pose assumes pose in format RC, not Rt."""
    est_pose = np.copy(est_pose)
    est_pose_c = np.copy(est_pose)
    t = -np.dot(est_pose[:3, :3], est_pose[:3, 3])
    est_pose[:3, 3] = t
    view = np.array([0, 0, 1])
    intr_inv = np.linalg.inv(est_intr)
    mv_inv = np.linalg.inv(est_pose)
    view_3d = np.dot(intr_inv, view)
    view_3d = np.concatenate((view_3d, [1]))
    x2 = np.dot(mv_inv, view_3d).transpose()
    x2 = x2[:3]
    x1 = est_pose_c[:3, 3]
    x0 = gt_pos
    x2x1 = x2 - x1
    tpar = -np.dot(x1 - x0, x2 - x1) / np.dot(x2x1, x2x1)
    xinter = x1 + ((x2 - x1) * tpar)

    y = np.linalg.norm(gt_pos - xinter)
    x = np.linalg.norm(xinter - x1)
    d = np.linalg.norm(gt_pos - x1)
    # x is distance from camera position along camera axis, y is distance
    # perpendicular to the camera axis towards the ground truth
    return x, y


def getResultForImage(image_name, result_dir, dataset_dir, matching_dir_name):
    # get translation and rotation result for best pose
    bp_R_gt_err = 180
    bp_t_gt_err = 100000
    bp_t_ren_err = 100000
    rp_R_gt_err = 180
    rp_t_gt_err = 100000
    rp_t_ren_err = 100000
    gps_t_gt_err = 100000
    rendered_pose = None
    gps_pose = None
    gt_fov = -1
    bp_fov = -1
    bp_fov_ba = -1
    rp_fov = -1
    rp_fov_ba = -1

    bp_min_err = -1
    bp_num_inliers = -1
    rp = None
    rp_min_err = -1
    rp_num_inliers = -1
    gt_t_ren_err = 100000
    matched_dist = np.array([-1])
    all_reproj_dist = np.array([-1])
    rp_err_axis = -1
    rp_err_perp = -1
    bp_filename = None

    gps_pose = getGPSPose(image_name, dataset_dir)

    rendered_pose, ren_center = getRenderedPose(image_name, result_dir)
    #if rendered_pose is None:
    #    rendered_pose = rendered_pose_gt

    if rendered_pose is None and ren_center is None:
        return bp_R_gt_err, bp_t_gt_err, bp_t_ren_err, bp_min_err, bp_num_inliers, rp_R_gt_err, rp_t_gt_err, rp_t_ren_err, rp_min_err, rp_num_inliers, gt_t_ren_err, gps_t_gt_err, rp_err_axis, rp_err_perp, gt_fov, bp_fov, bp_fov_ba, rp_fov, rp_fov_ba, matched_dist, all_reproj_dist, bp_filename
    try:
        gt_pose, gt_intrinsics = getGroundTruthPose(image_name, dataset_dir)
    except Exception:
        print("Ground truth not found, skipping.")
        return bp_R_gt_err, bp_t_gt_err, bp_t_ren_err, bp_min_err, bp_num_inliers, rp_R_gt_err, rp_t_gt_err, rp_t_ren_err, rp_min_err, rp_num_inliers, gt_t_ren_err, gps_t_gt_err, rp_err_axis, rp_err_perp, gt_fov, bp_fov, bp_fov_ba, rp_fov, rp_fov_ba, matched_dist, all_reproj_dist, bp_filename

    gt_fov = FUtil.intrinsicsToFov(gt_intrinsics)

    matching_dir = os.path.join(result_dir, matching_dir_name)
    result_img_dir = os.path.join(matching_dir, image_name)
    bp, bp_min_err, bp_num_inliers, bp_filename, bp_fov, bp_fov_ba = PoseFinder.findBestPose(result_img_dir)

    if bp_fov is None:
        bp_fov = -1
    if bp_fov_ba is None:
        bp_fov_ba = -1
    rp_path = os.path.join(result_img_dir, image_name + "_bestpose_pose.txt")
    rp_info_path = os.path.join(result_img_dir, image_name + "_bestpose_info.txt")
    rp = None
    rp_min_err = -1
    rp_num_inliers = -1
    if os.path.isfile(rp_path) and os.path.isfile(rp_info_path):
        rp = np.loadtxt(rp_path)
        rp_min_err, rp_num_inliers, rp_fov, rp_fov_ba= PoseFinder.parseInfoFile(rp_info_path)
        if rp_fov is None:
            rp_fov = -1
        if rp_fov_ba is None:
            rp_fov_ba = -1

    gt_t_ren_err = np.linalg.norm(gt_pose[:3, 3] - rendered_pose[:3, 3])
    if bp is not None:
        bp = movePoseToCenter(bp, ren_center)
        bp_R_gt_err, bp_t_gt_err = calculateErrors(gt_pose[:3, :3], gt_pose[:3, 3], bp[:3, :3], bp[:3, 3])
        bp_t_ren_err = np.linalg.norm(bp[:3, 3] - rendered_pose[:3, 3])
    else:
        bp_min_err = -1
        bp_num_inliers = -1

    # get translation and rotation result for refined pose
    matched_dist = np.array([-1])
    all_reproj_dist = np.array([-1])
    rp_err_axis = -1
    rp_err_perp = -1
    if rp is not None and not np.any(np.isnan(rp[:3, 3])):
        reproj_3D_path = os.path.join(result_img_dir, image_name + "_reproj_pt3D.npy")
        matched_3D_path = os.path.join(result_img_dir, image_name + "_matched_pt3D.npy")
        matched_3D = np.load(matched_3D_path)

        R_rp = rp[:3, :3]
        rp_center = np.dot(-R_rp.transpose(), rp[:3, 3])
        matched_dist = np.linalg.norm(matched_3D - rp_center, axis=1)
        if os.path.isfile(reproj_3D_path):
            reproj_3D = np.load(reproj_3D_path)
            all_reproj_dist = np.linalg.norm(reproj_3D - rp_center, axis=1)
        else:
            all_reproj_dist = np.array([-1])
        rp = movePoseToCenter(rp, ren_center)

        rp_err_axis, rp_err_perp = calculatePosErrAlongCameraAxis(gt_pose[:3, 3], rp, gt_intrinsics)

        rp_R_gt_err, rp_t_gt_err = calculateErrors(gt_pose[:3, :3], gt_pose[:3, 3], rp[:3, :3], rp[:3, 3])
        rp_t_ren_err = np.linalg.norm(rp[:3, 3] - rendered_pose[:3, 3])
    if gps_pose is not None:
        gps_t_gt_err = np.linalg.norm(gps_pose[:3, 3] - gt_pose[:3, 3])

    return bp_R_gt_err, bp_t_gt_err, bp_t_ren_err, bp_min_err, bp_num_inliers, rp_R_gt_err, rp_t_gt_err, rp_t_ren_err, rp_min_err, rp_num_inliers, gt_t_ren_err, gps_t_gt_err, rp_err_axis, rp_err_perp, gt_fov, bp_fov, bp_fov_ba, rp_fov, rp_fov_ba, matched_dist, all_reproj_dist, bp_filename

def plotCumulativeHistogram(values, label, linestyle, thr, nbins=100000, color=None, linewidth=1):
    isnan = np.isnan(values.astype(np.float))
    values = values[~isnan]
    h_sift = np.histogram(values, bins=nbins)
    d_sift = h_sift[0].astype(np.float)
    bin_w = h_sift[1][1] - h_sift[1][0]
    d_sift /= (d_sift * bin_w).sum()
    cs_sift = np.cumsum(d_sift * bin_w)
    print(label, h_sift[1])
    bins_sift = h_sift[1][0:nbins+0]
    bins_sift = np.insert(bins_sift, 0, bins_sift[0] - 0.1)
    bins_sift = np.insert(bins_sift, 0, 0)
    cs_sift = np.insert(cs_sift, 0, 0)
    cs_sift = np.insert(cs_sift, 0, 0)
    sel = bins_sift <= thr
    if color:
        plt.plot(bins_sift[sel], cs_sift[sel], label=label, linestyle=linestyle, color=color, linewidth=linewidth)
    else:
        plt.plot(bins_sift[sel], cs_sift[sel], label=label, linestyle=linestyle, linewidth=linewidth)
    return bins_sift[sel], cs_sift[sel]


def getTimestampForImage(image_name, original_images_path):
    search_path = os.path.join(original_images_path, image_name + "*")
    orig_image = glob.glob(search_path)
    if len(orig_image) > 0:
        orig_image = orig_image[0]
    else:
        return None

    with exiftool.ExifTool() as et:
        res = et.execute_json("-CreateDate", orig_image)
        if len(res) > 0 and 'EXIF:CreateDate' in res[0]:
            try:
                raw_datetime = res[0]['EXIF:CreateDate']
                parts = raw_datetime.split('+')
                photo_datetime = datetime.datetime.strptime(
                    parts[0], '%Y:%m:%d %H:%M:%S'
                )
                #if (len(parts) > 1):
                #    hours = datetime.datetime.strptime(
                #        parts[1], '%H:%M'
                #    )
                #    photo_datetime += hours

                return photo_datetime
            except ValueError as ve:
                print(ve)
                print("image:", orig_image, "EXIF datetime", res[0]['EXIF:CreateDate'])

    return None

def getResults(args):
    result_dir = args.directory
    dataset_dir = args.dataset_dir
    matching_dir = os.path.join(result_dir, args.matching_dir_name)

    image_names = os.listdir(matching_dir)
    bp_filenames = []
    all_img_errors = []
    all_img_names = []
    all_matched_dists = []
    all_reproj_dists = []
    all_gt_fov = []
    timestamps = []
    for image_name in tqdm(image_names):
        if image_name == "no_orientation_gt":
            continue
        if os.path.isdir(os.path.join(matching_dir, image_name)):
            all_img_names.append(image_name)
            res = getResultForImage(image_name, result_dir, dataset_dir,
                                    args.matching_dir_name)

            all_gt_fov.append(res[14])
            img_errors = np.array(res[:19])
            all_matched_dists.append(res[19])
            all_reproj_dists.append(res[20])
            all_img_errors.append(img_errors)
            bp_filenames.append(res[21])

            if args.original_images:
                timestamp = getTimestampForImage(image_name, args.original_images)
                timestamps.append(timestamp)

            print(image_name, img_errors[5], img_errors[6], img_errors[0], img_errors[1], img_errors[4])
    all_img_errors = np.array(all_img_errors)
    all_reproj_dists = np.concatenate(all_reproj_dists, axis=0)
    all_matched_dists = np.concatenate(all_matched_dists, axis=0)
    all_gt_fov = np.array(all_gt_fov)

    #print(all_matched_dists.shape)
    #print(all_img_errors)
    sel = all_img_errors[:, 1] > 100000
    all_img_errors[:, 1][sel] = 100000
    sel = all_img_errors[:, 2] > 100000
    all_img_errors[:, 2][sel] = 100000
    sel = all_img_errors[:, 6] > 100000
    all_img_errors[:, 6][sel] = 100000
    sel = all_img_errors[:, 7] > 100000
    all_img_errors[:, 7][sel] = 100000
    sel = all_img_errors[:, 11] > 100000
    all_img_errors[:, 11][sel] = 100000

    thr_orient = 20
    plt.figure()
    plt.title("Cumulative orientation error [degrees]")
    plotCumulativeHistogram(all_img_errors[:, 0], label='best_pose_gt_Rot', linestyle='-', thr=thr_orient, nbins=5000)
    plotCumulativeHistogram(all_img_errors[:, 5], label='refined_pose_gt_Rot', linestyle='--', thr=thr_orient, nbins=5000)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlabel("Rotation magnitude [degrees]")
    plt.ylabel("Fraction of images")
    plt.savefig(os.path.join(matching_dir, "rot_err.pdf"))
    thr_t = 2000
    plt.figure()
    plt.title("Cumulative positional error [meters]")
    plotCumulativeHistogram(all_img_errors[:, 1], label='best_pose_gt_t', linestyle='-', thr=thr_t, nbins=5000)
    plotCumulativeHistogram(all_img_errors[:, 2], label='best_pose_gps', linestyle='-', thr=thr_t, nbins=5000)
    plotCumulativeHistogram(all_img_errors[:, 6], label='refined_pose_gt_t', linestyle='--', thr=thr_t, nbins=5000)
    plotCumulativeHistogram(all_img_errors[:, 7], label='refined_pose_gps', linestyle='--', thr=thr_t, nbins=5000)
    plotCumulativeHistogram(all_img_errors[:, 10], label='gt_t_gps', linestyle='-.', thr=thr_t, nbins=5000)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlabel("Distance [m]")
    plt.ylabel("Fraction of images")
    plt.savefig(os.path.join(matching_dir, "t_err.pdf"))
    plt.figure()
    plt.title("Histogram of positional errors")
    max_t_err = 2000
    sel = all_img_errors[:, 6] < max_t_err

    _, bins, _ = plt.hist(np.array(all_img_errors[sel, 6]), bins=500, label='refined_pose_gt_t', alpha=0.5)
    #plt.hist(np.array(all_img_errors[sel, 1]), bins=bins, label='best_pose_gt_t', alpha=0.5)
    sel = all_img_errors[:, 10] < max_t_err
    plt.hist(np.array(all_img_errors[sel, 10]), bins=500, label='gt_t_gps', alpha=0.5)
    #ticks = np.arange(np.min(all_img_errors[sel, 6]), np.max(all_img_errors[sel, 6]) + 1, 100.0)
    #plt.xticks(ticks)
    plt.xlabel("Distance [m]")
    plt.ylabel("Number of images")

    #plt.hist(all_img_errors[:, 10], label='gt_t_render_t', linestyle='--')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(matching_dir, "t_err_hist.pdf"))

    plt.figure()
    all_matched_dists = all_matched_dists[all_matched_dists <= 100000]
    all_reproj_dists = all_reproj_dists[all_reproj_dists <= 100000]
    all_hist, bins, _ = plt.hist(all_reproj_dists, bins=100, label='all_pt_dists', alpha=0.5, density=True)
    matched_hist, _, _ = plt.hist(all_matched_dists, bins=bins, label='matched_pt_dists', alpha=0.5, density=True)
    print(all_reproj_dists)
    plt.xlabel("Distance [m]")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(matching_dir, "point_distances.pdf"))

    #plt.figure()
    #proba_matched_dist = np.divide(matched_hist, all_hist)
    #print(proba_matched_dist)
    #plt.plot(bins[1:], proba_matched_dist)

    plt.figure()
    sel = all_img_errors[:, 6] < max_t_err
    axis_err = all_img_errors[:, 11]
    perp_err = all_img_errors[:, 12]
    axis_err = axis_err[np.logical_and(sel, axis_err > 0)]
    perp_err = perp_err[np.logical_and(sel, perp_err > 0)]
    plt.hist(axis_err, bins=100, label='refined_pose_axis_dist', alpha=0.5)
    plt.hist(perp_err, bins=100, label='refined_pose_perp_dist', alpha=0.5)
    plt.xlabel("Distance [m]")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(matching_dir, "t_err_perp_axis_parallel_axis.pdf"))

    sel = all_gt_fov > 0
    plt.figure()
    plt.scatter(all_img_errors[sel, 6].flatten(), (all_gt_fov[sel].flatten() * 180.0) / np.pi)
    plt.xlabel('refined_pose_gt_t')
    plt.ylabel('Field-of-view')
    plt.savefig(os.path.join(matching_dir, "t_err_fov_scatter.pdf"))

    plt.figure()
    ax = plt.gca()
    plt.scatter(all_img_errors[:, 4].flatten(), all_img_errors[:, 1].flatten())
    plt.xlabel('number of inliers')
    plt.ylabel('dist. of BP from GT')
    ax.set_yscale('log')
    plt.savefig(os.path.join(matching_dir, "num_inliers_gt_t_scatter.pdf"))

    # plot fov
    sel = all_img_errors[:, 14] > 0
    thr_fov = 20
    bp_fov_err = (np.abs(all_img_errors[sel, 15] - all_img_errors[sel, 14]) * 180) / np.pi
    bp_fov_ba_err = (np.abs(all_img_errors[sel, 16] - all_img_errors[sel, 14]) * 180) / np.pi
    rp_fov_err = (np.abs(all_img_errors[sel, 17] - all_img_errors[sel, 14]) * 180) / np.pi
    rp_fov_ba_err = (np.abs(all_img_errors[sel, 18] - all_img_errors[sel, 14]) * 180) / np.pi

    plt.figure()
    plotCumulativeHistogram(bp_fov_err, label='bp_fov_err', linestyle='-', thr=thr_fov, nbins=100000)
    plotCumulativeHistogram(bp_fov_ba_err, label='bp_fov_ba_err', linestyle='-', thr=thr_fov, nbins=100000)
    plotCumulativeHistogram(rp_fov_err, label='rp_fov_err', linestyle='-.', thr=thr_fov, nbins=100000)
    plotCumulativeHistogram(rp_fov_ba_err, label='rp_fov_ba_err', linestyle='-.', thr=thr_fov, nbins=100000)
    plt.xlabel('FOV err [deg]')
    plt.ylabel('fraction of images')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(matching_dir, "fov_err.pdf"))

    if args.original_images:
        # plot mean accuracy given the month
        months = []
        # allocate month buckets
        for m in range(0, 12):
            months.append([])
        # sort the results according to the month
        months_arr = np.arange(len(timestamps)).reshape(-1, 1).astype(np.int32)
        for idx in range(0, len(timestamps)):
            datetime = timestamps[idx]
            if datetime is not None:
                months[datetime.month - 1].append(all_img_errors[idx, :])
                months_arr[idx] = datetime.month
            else:
                months_arr[idx] = -1
        all_img_errors = np.concatenate([all_img_errors, months_arr], axis=1)
        mean_inliers_months = np.arange(0, 12)
        num_photos_months = np.arange(0, 12)
        for m in range(0, 12):
            month_data = np.array(months[m])
            num_photos_months[m] = month_data.shape[0]
            if month_data.shape[0] > 0:
                mean_inliers_months[m] = np.mean(month_data[:, 4])
            else:
                mean_inliers_months[m] = 0

        plt.figure()
        ax = plt.gca()
        plt.plot(np.arange(1,13), mean_inliers_months, marker='o', label='mean inliers')
        plt.plot(np.arange(1,13), num_photos_months, marker='o', label='num photos')
        plt.xlabel('month')
        plt.ylabel('mean distance to GT [m]')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(matching_dir, "gt_t_month_lineplot.pdf"))



    np.savetxt(os.path.join(matching_dir, "img_errors_only.txt"), all_img_errors)

    delim = ":"
    img_errors_outfile = os.path.join(matching_dir, "img_errors.txt")
    with open(img_errors_outfile, 'w') as f:
        for idx in range(0, all_img_errors.shape[0]):
            f.write(all_img_names[idx] + delim)
            bp_filename = bp_filenames[idx]
            if bp_filename is None:
                bp_filename = "None"
            f.write(os.path.basename(bp_filename) + delim)
            for jdx in range(0, all_img_errors.shape[1]):
                f.write('{:.3f}'.format(all_img_errors[idx, jdx]) + delim)

            f.write("\n")

    plt.show()

if __name__ == "__main__":
    parser = buildArgumentParser()
    args = parser.parse_args()

    getResults(args)
