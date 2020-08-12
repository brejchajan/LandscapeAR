# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:51:59+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:48:20+02:00
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
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
font_size = 25
plt.rcParams['font.size'] = font_size
plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = '\\usepackage{libertine},\\usepackage[libertine]{newtxmath},\\usepackage[T1]{fontenc}'

from getPoseResults import plotCumulativeHistogram
plt.ioff()

output_dir = "../latex/LandscapeAR_ECCV2020/eccv2020kit/graphs"
#matylda_path = "/Users/janbrejcha/matylda1"
matylda_path = "/mnt/matylda1/"
dataset_dir = "switzerland_wallis_30km_maximg_test"
results_path = os.path.join(matylda_path, "ibrejcha/data/matching/rendered_datasets/orig")
#experiment_path = os.path.join(results_path, dataset_dir)
experiment_path = os.path.join(results_path, 'geoPose3K_test')

dataset_list = ['geoPose3K_test', 'nepal', 'andes_huascaran', 'yosemite']
#'ETH_CH1', 'ETH_CH2', 'switzerland_wallis_30km_maximg_test' 'switzerland_wallis_30km_maximg_test'
# , 'nepal', 'andes_huascaran', 'yosemite'


def plotDatasetMethodComparison(result_dir, maxdist=2000, cache=False):
    num_datasets = len(dataset_list)
    dset_idx = 1
    allfig = plt.figure(figsize=(12,9))
    for dataset in dataset_list:
        fig = plt.subplot(2, 2, dset_idx)
        experiment_path = os.path.join(result_dir, dataset)
        for method in experiment_list:
            errors_path = os.path.join(experiment_path, method, 'img_errors_only.txt')
            cache_path = os.path.join(os.getcwd(), "exper_results", dataset, method)
            loaded = False
            errors = []
            cached_errors = os.path.join(cache_path, "img_errors_only.txt")
            if os.path.isfile(cached_errors):
                if cache:
                    errors = np.loadtxt(cached_errors)
                    loaded = True
            elif os.path.isfile(errors_path):
                errors = np.loadtxt(errors_path)
                loaded = True
                if cache:
                    if not os.path.isdir(cache_path):
                        os.makedirs(cache_path)
                    np.savetxt(cached_errors, errors)

            if loaded:
                type = 6
                if method == 'ncnet_pfpascal_matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000':
                    type = 1

                rp_t_err = errors[:, type] #6
                if True or dataset == "nepal" and method == "matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_siftdesc_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output":

                    print("num imgs for dataset and method", dataset, method, rp_t_err.shape)
                    plotCumulativeHistogram(rp_t_err, label_map[method], lstyl_map[method], maxdist, color=color_map[method], linewidth=3)
                ax = plt.gca()
                plt.legend(loc='lower right', fontsize=font_size * 0.5)
                title = " ".join([word.capitalize().replace("pose", "Pose").replace("3k", "3K") for word in dataset.replace("_", " ").replace("test", "").split(' ')])
                plt.title(title, fontsize=font_size * 0.8)
                maxyloc = 0.9 #np.max(locs) + 0.1
                plt.yticks(np.arange(0, maxyloc, step=0.1))
                plt.xticks(np.arange(0, maxdist+1, step=500))
                #ax.tick_params(axis='both', which='major', labelsize=font_size * 0.5)
                plt.xlabel('distance from ground truth [m]', fontsize=font_size * 1.2)
                plt.ylabel('fraction of images', fontsize=font_size * 1.2)
                plt.grid(True, linestyle='--')
            else:
                print("Could not load: ", errors_path)
        dset_idx = dset_idx + 1
    allfig.tight_layout()
    plt.savefig(os.path.join("./graphs", "four_datasets_t_err.pdf"))

def plotDatasetMethodComparisonInliersMonth(result_dir, maxdist=12, cache=False):
    num_datasets = len(dataset_list)
    dset_idx = 1
    allfig = plt.figure(figsize=(12,9))
    for dataset in dataset_list:
        fig = plt.subplot(2, 2, dset_idx)
        experiment_path = os.path.join(result_dir, dataset)
        midx = 0
        for method in experiment_list:
            midx += 1
            errors_path = os.path.join(experiment_path, method, 'img_errors_only.txt')
            cache_path = os.path.join(os.getcwd(), "exper_results", dataset, method)
            loaded = False
            errors = []
            cached_errors = os.path.join(cache_path, "img_errors_only.txt")
            if os.path.isfile(cached_errors):
                if cache:
                    errors = np.loadtxt(cached_errors)
                    loaded = True
            elif os.path.isfile(errors_path):
                errors = np.loadtxt(errors_path)
                loaded = True
                if cache:
                    if not os.path.isdir(cache_path):
                        os.makedirs(cache_path)
                    np.savetxt(cached_errors, errors)

            if loaded:
                type = 9
                if method == 'ncnet_pfpascal_matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000':
                    type = 4

                sel = errors[:, type] < 0
                errors[sel, type] = 0
                print(dataset, method)
                print("errors shape", errors.shape)
                timestamps = errors[:, 13]
                months = []
                # allocate month buckets
                for m in range(0, 12):
                    months.append([])
                for idx in range(0, len(timestamps)):
                    month = int(timestamps[idx])
                    if month >= 0:
                        months[month - 1].append(errors[idx, :])
                mean_inliers_months = np.arange(0, 12)
                num_photos_months = np.arange(0, 12)
                sum_inliers = 0
                for m in range(0, 12):
                    month_data = np.array(months[m])
                    if month_data.shape[0] > 0:
                        sum_inliers += np.sum(month_data[:, type])
                for m in range(0, 12):
                    month_data = np.array(months[m])
                    num_photos_months[m] = month_data.shape[0]
                    if month_data.shape[0] > 0:
                        mean_inliers_months[m] = np.mean(month_data[:, type])
                    else:
                        mean_inliers_months[m] = 0

                if midx == 1:
                    plt.plot(np.arange(1,13), num_photos_months, marker='o', label='num photos', linestyle='--')
                plt.plot(np.arange(1,13), mean_inliers_months, marker='o', label=label_map[method], color=color_map[method])
                ax = plt.gca()
                plt.legend(loc='upper right', fontsize=font_size * 0.5)
                title = " ".join([word.capitalize().replace("pose", "Pose").replace("3k", "3K") for word in dataset.replace("_", " ").replace("test", "").split(' ')])
                plt.title(title, fontsize=font_size * 0.8)
                #plt.yticks(np.arange(0, 0.8, step=0.25))
                plt.xticks(np.arange(0, maxdist+1, step=1))
                #ax.tick_params(axis='both', which='major', labelsize=font_size * 0.5)
                plt.xlabel('month', fontsize=font_size * 1.2)
                plt.ylabel('inliers', fontsize=font_size * 1.2)
                plt.grid(True, linestyle='--')
            else:
                print("Could not load: ", errors_path)

        dset_idx = dset_idx + 1
    allfig.tight_layout()
    plt.savefig(os.path.join("./graphs", "four_datasets_inliers_month.pdf"))


def plotDatasetMethodComparisonInliers(result_dir, maxdist=1000, cache=False):
    num_datasets = len(dataset_list)
    dset_idx = 1
    allfig = plt.figure(figsize=(12,9))
    for dataset in dataset_list:
        fig = plt.subplot(2, 2, dset_idx)
        experiment_path = os.path.join(result_dir, dataset)
        for method in experiment_list:
            errors_path = os.path.join(experiment_path, method, 'img_errors_only.txt')
            cache_path = os.path.join(os.getcwd(), "exper_results", dataset, method)
            loaded = False
            errors = []
            cached_errors = os.path.join(cache_path, "img_errors_only.txt")
            if os.path.isfile(cached_errors):
                if cache:
                    errors = np.loadtxt(cached_errors)
                    loaded = True
            elif os.path.isfile(errors_path):
                errors = np.loadtxt(errors_path)
                loaded = True
                if cache:
                    if not os.path.isdir(cache_path):
                        os.makedirs(cache_path)
                    np.savetxt(cached_errors, errors)

            if loaded:
                type = 9
                if method == 'ncnet_pfpascal_matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000':
                    type = 4
                sel = errors[:, type] < 0
                errors[sel, type] = 0
                mean_inliers = np.mean(errors[:, type].flatten())
                med_inliers = np.median(errors[:, type].flatten())
                plt.scatter(errors[:, type].flatten(), errors[:, 1].flatten(), alpha=0.6, label=label_map[method] + '; $\mu$ inl: ' + "{:.1f}".format(mean_inliers) + ", $\mathrm{M}$ inl: " + "{:.1f}".format(med_inliers), color=color_map[method])
                ax = plt.gca()
                plt.legend(loc='upper right', fontsize=font_size * 0.5)
                title = " ".join([word.capitalize().replace("pose", "Pose").replace("3k", "3K") for word in dataset.replace("_", " ").replace("test", "").split(' ')])

                print(title, label_map[method], mean_inliers, med_inliers, sep=";")
                plt.title(title, fontsize=font_size * 0.8)
                plt.yticks(np.arange(0, 0.8, step=0.25))
                plt.xticks(np.arange(0, maxdist+1, step=500))
                #ax.tick_params(axis='both', which='major', labelsize=font_size * 0.5)
                ax.set_yscale('log')
                plt.xlabel('number of inliers', fontsize=font_size * 1.2)
                plt.ylabel('distance from GT [m]', fontsize=font_size * 1.2)
                plt.grid(True, linestyle='--')
            else:
                print("Could not load: ", errors_path)
        dset_idx = dset_idx + 1
    allfig.tight_layout()
    plt.savefig(os.path.join("./graphs", "four_datasets_t_err_inliers.pdf"))


def printPercentageBarPlot(gt_gps_err, rp_t_err, maxdist=2000, step=100, xtickstep=500):
    all_improved = []
    all_worsen = []
    all_improved_cnt = []
    all_worsen_cnt = []
    all_failed = []
    all_failed_cnt = []
    bins = []
    counts = []
    maxdistgps =int(np.round(np.max(gt_gps_err)))
    for d in tqdm(range(0, maxdistgps, step)):
        sel_failed = np.logical_and(np.logical_and(gt_gps_err > d, gt_gps_err < (d + step)), rp_t_err > 2000)
        sel = np.logical_and(np.logical_and(gt_gps_err > d, gt_gps_err < (d + step)), rp_t_err < 2000)
        improved = np.sum(rp_t_err[sel] < gt_gps_err[sel])
        worsen = np.sum(rp_t_err[sel] > gt_gps_err[sel])
        failed = np.sum(sel_failed)
        all_improved_cnt.append(improved)
        all_worsen_cnt.append(worsen)
        all_failed_cnt.append(failed)
        sum = improved + worsen + np.sum(sel_failed)
        improved = improved / (sum + 0.000001)
        worsen = worsen / (sum + 0.000001)
        failed = failed / (sum + 0.000001)
        all_improved.append(improved)
        all_worsen.append(worsen)
        all_failed.append(failed)
        bins.append(d)
        counts.append(sum)
    all_improved = np.array(all_improved)
    all_worsen = np.array(all_worsen)
    all_failed = np.array(all_failed)
    all_improved_cnt = np.array(all_improved_cnt)
    all_worsen_cnt = np.array(all_worsen_cnt)
    all_failed_cnt = np.array(all_failed_cnt)
    counts = np.array(counts)
    bins = np.array(bins)
    plt.figure()
    improved_perc = (np.sum(all_improved_cnt) / np.sum(counts)) * 100
    worsen_perc = (np.sum(all_worsen_cnt) / np.sum(counts)) * 100
    failed_perc = (np.sum(all_failed_cnt) / np.sum(counts)) * 100
    plt.bar(bins, all_improved, label='{:.0f}'.format(improved_perc) + '\%', color='mediumseagreen', width=step, edgecolor='white')
    plt.bar(bins, all_worsen, bottom=all_improved, label='{:.0f}'.format(worsen_perc) + '\%', color='orange', width=step, edgecolor='white')
    plt.bar(bins, all_failed, bottom=all_worsen + all_improved, label='{:.0f}'.format(failed_perc) + '\%', color='indianred', width=step, edgecolor='white')
    plt.xticks(np.arange(0, maxdistgps+500, step=xtickstep), fontsize=12)
    plt.yticks(np.arange(0, 1+0.25, step=0.25), fontsize=12)
    for i in range(0, counts.shape[0]):
        plt.text(bins[i], 0.03, str(counts[i]), horizontalalignment='center', verticalalignment='center', fontdict=dict(fontsize = 0.5 * font_size))
    #    plt.text(bins[i], 1 - 0.04, str(all_worsen_cnt[i]), horizontalalignment='center', verticalalignment='center', fontdict=dict(fontsize = 0.5 * font_size))
    #plt.bar(bins, counts/np.max(counts), alpha=0.5, label='sum', width=step)
    print("Sum of improved and worsen", np.sum(counts))
    plt.legend(loc='lower right', fontsize=font_size * 0.8, ncol=3, numpoints=1, mode="expand", bbox_to_anchor=(-0.02, 1.02, 1.04, .102))
    #plt.xlabel('$||p_s - p_g|| [m]$', fontsize=font_size * 1.2)
    plt.xlabel('baseline [m]', fontsize=font_size * 1.2)
    plt.ylabel('fraction of images', fontsize=font_size * 1.2)
    plt.tight_layout()

def plotTranslationErrorTable(output_dir, name, xstep=500, maxdist=2000):
    # translation error plot
    fig = plt.figure()
    for exper in experiment_list:
        errors_path = os.path.join(experiment_path, exper, 'img_errors_only.txt')
        errors = np.loadtxt(errors_path)
        values = errors[:, 6]
        isnan = np.isnan(values.astype(np.float))
        values = values[~isnan]
        nbins = np.arange(100, np.max(values), 100)
        thr = maxdist
        h_sift = np.histogram(values, bins=nbins)
        d_sift = h_sift[0].astype(np.float)
        bin_w = h_sift[1][1] - h_sift[1][0]
        d_sift /= (d_sift * bin_w).sum()
        cs_sift = np.cumsum(d_sift * bin_w)
        bins_sift = nbins[:-1]#h_sift[1][nbins]
        sel = bins_sift <= thr
        xval = bins_sift[sel]
        yval = cs_sift[sel]

        np.set_printoptions(2)
        print("xval: ", label_map[exper], xval)
        print("yval: ", label_map[exper], yval)

def plotRotationErrorTable(output_dir, name, thr_orient=10, orient_step=5):
    #orientation error plot
    print("orientation")
    fig = plt.figure()
    for exper in experiment_list:
        errors_path = os.path.join(experiment_path, exper, 'img_errors_only.txt')
        errors = np.loadtxt(errors_path)
        values = errors[:, 5]
        isnan = np.isnan(values.astype(np.float))
        values = values[~isnan]
        nbins = np.arange(1, np.max(values), 1)
        thr = thr_orient
        h_sift = np.histogram(values, bins=nbins)
        d_sift = h_sift[0].astype(np.float)
        bin_w = h_sift[1][1] - h_sift[1][0]
        d_sift /= (d_sift * bin_w).sum()
        cs_sift = np.cumsum(d_sift * bin_w)
        bins_sift = nbins[:-1]#h_sift[1][nbins]
        sel = bins_sift <= thr
        xval = bins_sift[sel]
        yval = cs_sift[sel]

        np.set_printoptions(2)
        print("xval: ", label_map[exper], xval)
        print("yval: ", label_map[exper], yval)

def plotTranslationErrorPlot(output_dir, name, xstep=500, maxdist=2000):
    # translation error plot
    fig = plt.figure()
    for exper in experiment_list:
        errors_path = os.path.join(experiment_path, exper, 'img_errors_only.txt')
        errors = np.loadtxt(errors_path)
        rp_t_err = errors[:, 6]
        plotCumulativeHistogram(rp_t_err, label_map[exper], lstyl_map[exper], maxdist, color=color_map[exper], linewidth=3)
        plt.legend(loc='lower right', fontsize=font_size * 0.8)
        plt.xlabel('distance from ground truth [m]', fontsize=font_size * 1.2)
        plt.ylabel('fraction of images', fontsize=font_size * 1.2)
        plt.xticks(np.arange(0, maxdist+xstep, step=xstep))
        locs, labels = plt.yticks()
        maxyloc = 0.9 #np.max(locs) + 0.1
        plt.yticks(np.arange(0, maxyloc, step=0.1))
        plt.grid(True, linestyle='--')
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(os.path.join(output_dir, name))


def plotRotationErrorPlot(output_dir, name, thr_orient=20, orient_step=5):
    #orientation error plot
    fig = plt.figure()
    for exper in experiment_list:
        errors_path = os.path.join(experiment_path, exper, 'img_errors_only.txt')
        errors = np.loadtxt(errors_path)
        rp_ori_err = errors[:, 5]
        plotCumulativeHistogram(rp_ori_err, label_map[exper], lstyl_map[exper], thr_orient, color=color_map[exper], linewidth=3)
        plt.legend(loc='lower right', fontsize=font_size * 0.8)
        plt.xlabel('rotation error [°]', fontsize=font_size * 1.2)
        plt.ylabel('fraction of images', fontsize=font_size * 1.2)
        plt.xticks(np.arange(0, thr_orient+orient_step, step=orient_step))
        locs, labels = plt.yticks()
        #maxyloc = np.max(locs) + 0.1
        maxyloc = 0.9
        plt.yticks(np.arange(0, maxyloc, step=0.1))
        plt.grid(True, linestyle='--')
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(os.path.join(output_dir, name))

def plotRotationErrorPlotRefinedAndBestPose(output_dir, name, thr_orient=20, orient_step=5):
    #refined pose vs. best pose, distance to ground truth and render
    plt.figure()
    for exper in experiment_list:
        errors_path = os.path.join(experiment_path, exper, 'img_errors_only.txt')
        all_img_errors = np.loadtxt(errors_path)
        plotCumulativeHistogram(all_img_errors[:, 0], label=label_map[exper]+'-bp', linestyle='-', thr=thr_orient, linewidth=3, color=color_map[exper])
        plotCumulativeHistogram(all_img_errors[:, 5], label=label_map[exper]+'-rp', linestyle='--', thr=thr_orient, linewidth=3, color=color_map[exper])
        plt.legend(loc='lower right', fontsize=font_size * 0.8)
        plt.grid(True, linestyle='--')
        plt.xlabel("rotation error [°]", fontsize=font_size * 1.2)
        plt.ylabel("fraction of images", fontsize=font_size * 1.2)
        plt.xticks(np.arange(0, thr_orient+orient_step, step=orient_step))
        locs, labels = plt.yticks()
        #maxyloc = np.max(locs) + 0.1
        maxyloc = 0.9
        plt.yticks(np.arange(0, maxyloc, step=0.1))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(os.path.join(output_dir, name))

def plotTranslationErrorPlotRefinedAndBestPose(output_dir, name, thr_t=2000):
    plt.figure()
    for exper in experiment_list:
        errors_path = os.path.join(experiment_path, exper, 'img_errors_only.txt')
        all_img_errors = np.loadtxt(errors_path)
        plotCumulativeHistogram(all_img_errors[:, 1], label=label_map[exper]+'-bp', linestyle='-', thr=thr_t, linewidth=3, color=color_map[exper])
        #plotCumulativeHistogram(all_img_errors[:, 2], label='bp-gps', linestyle='-', thr=thr_t, linewidth=3)
        plotCumulativeHistogram(all_img_errors[:, 6], label=label_map[exper]+'-rp', linestyle='--', thr=thr_t, linewidth=3, color=color_map[exper])
        #plotCumulativeHistogram(all_img_errors[:, 7], label='rp-gps', linestyle='--', thr=thr_t, linewidth=3)
        #plotCumulativeHistogram(all_img_errors[:, 10], label='gps-gt', linestyle='-.', thr=thr_t, linewidth=3)
        plt.legend(loc='lower right', fontsize=font_size * 0.8, columnspacing=0.5)
        plt.grid(True, linestyle='--')
        plt.xlabel("distance [m]", fontsize=font_size * 1.2)
        plt.ylabel("fraction of images", fontsize=font_size * 1.2)
        plt.xticks(np.arange(0, thr_t+500, step=500))
        maxyloc = 0.9
        plt.yticks(np.arange(0, maxyloc, step=0.1))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(os.path.join(output_dir, name))

def plotTranslationErrorPlotNoisyPositions(experiment_list, output_dir, name, thr_t=2000):
    experiment_path = "/mnt/matylda1/ibrejcha/data/matching/rendered_datasets/gaussnoise_1000/geoPose3K_test/"
    plt.figure()
    exper = experiment_list[0]
    errors_path = os.path.join(experiment_path, exper, 'img_errors_only.txt')
    all_img_errors = np.loadtxt(errors_path)
    #plotCumulativeHistogram(all_img_errors[:, 1], label=label_map[exper]+'-bp', linestyle='-', thr=thr_t, linewidth=3, color=color_map[exper])
    #plotCumulativeHistogram(all_img_errors[:, 2], label='bp-gps', linestyle='-', thr=thr_t, linewidth=3)
    plotCumulativeHistogram(all_img_errors[:, 6], label=label_map[exper]+'-gt', linestyle='--', thr=thr_t, linewidth=3, color=color_map[exper])
    #plotCumulativeHistogram(all_img_errors[:, 7], label=label_map[exper]+'-sp', linestyle='--', thr=thr_t, linewidth=3)
    plotCumulativeHistogram(all_img_errors[:, 10], label='sp-gt', linestyle='-.', thr=thr_t, linewidth=3, color='orange')
    #plotCumulativeHistogram(all_img_errors[:, 11], label='gps-gt', linestyle='-.', thr=thr_t, linewidth=3, color='dodgerblue')
    print("min err", np.sum(all_img_errors[:, 6] < 200), all_img_errors[:, 6].shape)
    plt.legend(loc='lower right', fontsize=font_size * 0.8, columnspacing=0.5)
    plt.grid(True, linestyle='--')
    plt.xlabel("distance from ground truth [m]", fontsize=font_size * 1.2)
    plt.ylabel("fraction of images", fontsize=font_size * 1.2)
    plt.xticks(np.arange(0, thr_t+500, step=500))
    maxyloc = 1.1
    #plt.yticks(np.arange(0, maxyloc, step=0.1))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(os.path.join(output_dir, name))


if False:
    ### percentage of poses with more precise position than was the render position
    experiment_path = "/mnt/matylda1/ibrejcha/data/matching/rendered_datasets/gaussnoise_500/switzerland_wallis_30km_maximg_test/"
    errors_path = os.path.join(experiment_path, experiment_list[1], 'img_errors_only.txt')
    errors = np.loadtxt(errors_path)
    maxdist = 2000
    rp_t_err = errors[:, 6]
    sel = rp_t_err > 0 #< maxdist
    rp_t_err = errors[sel, 6]
    gt_gps_err = errors[sel, 10]
    diff = rp_t_err - gt_gps_err
    plt.figure()
    plt.hist(diff, 100)
    plt.figure()
    plt.hist(errors[:, 10], 100)
    plt.hist(errors[sel, 10], 100)
    printPercentageBarPlot(gt_gps_err, rp_t_err, step=75)
    plt.savefig(os.path.join(output_dir, "matterhorn-ours-conv7l-gaussnoise-500.pdf"))

def plotNoisyPercentage(experiment_list):
    ### percentage of poses with more precise position than was the render position
    experiment_path = "/mnt/matylda1/ibrejcha/data/matching/rendered_datasets/gaussnoise_1000/geoPose3K_test/"
    errors_path = os.path.join(experiment_path, experiment_list[0], 'img_errors_only.txt')
    errors = np.loadtxt(errors_path)
    rp_t_err = errors[:, 6]
    sel = rp_t_err > 0 #< maxdist
    rp_t_err = errors[sel, 6]
    gt_gps_err = errors[sel, 10]
    diff = rp_t_err - gt_gps_err
    plt.figure()
    plt.hist(diff, 100)
    plt.figure()
    plt.hist(errors[:, 10], 100)
    plt.hist(errors[sel, 10], 100)

    printPercentageBarPlot(gt_gps_err, rp_t_err, step=200, xtickstep=1000)
    plt.savefig(os.path.join(output_dir, "geoPose3K-ours-conv7l-gaussnoise-1000.pdf"))
    print("number of images ", errors.shape[0])

### the plotting

## comparison on four datasets
experiment_list = ['d2net_matcher_numfeat_lowinliers',
                   'ncnet_pfpascal_matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000',
                   'matching_hardnet_maxphotowidth_1200',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output',
                   #'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_MultimodalPatchNet5lShared2l_epoch_15_step_700000_normalize_output',
                   'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_siftdesc_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output']

label_map = {'d2net_matcher_numfeat_lowinliers': 'D2Net',
             'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_MultimodalPatchNet5lShared2l_epoch_15_step_700000_normalize_output': 'Ours-Conv7l',
             'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': 'Ours',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'Ours-aux',
             'matching_hardnet_maxphotowidth_1200': 'HardNet++',
             'ncnet_pfpascal_matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000': 'NCNet',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output': 'VGG-16-D2-FT',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_siftdesc_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output' : 'SIFT'
             }
lstyl_map = {'d2net_matcher_numfeat_lowinliers': '-',
             'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_MultimodalPatchNet5lShared2l_epoch_15_step_700000_normalize_output': '--',
             'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': '--',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': '--',
             'matching_hardnet_maxphotowidth_1200': '-',
             'ncnet_pfpascal_matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000': '-',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output': '--',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_siftdesc_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output' : '-'
             }
color_map = {'d2net_matcher_numfeat_lowinliers': 'mediumseagreen',
             'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_MultimodalPatchNet5lShared2l_epoch_15_step_700000_normalize_output': 'indianred',
             'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': 'darkslateblue',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'indianred',
             'matching_hardnet_maxphotowidth_1200': 'orange',
             'ncnet_pfpascal_matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000': 'darkgoldenrod',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output':'darkgreen',
             'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_siftdesc_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output' : 'saddlebrown'
             }

#plotDatasetMethodComparison(results_path)

#plotDatasetMethodComparisonInliers(results_path)

#plotDatasetMethodComparisonInliersMonth(results_path)



## comparison of normal vs aux vs locked singlemodal vs hardnet++ vs d2net
experiment_list = ['d2net_matcher_numfeat_lowinliers',
                   'matching_hardnet_maxphotowidth_1200',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output',
                   'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output',
                   ]

label_map = {'d2net_matcher_numfeat_lowinliers': 'D2Net',
                   'matching_hardnet_maxphotowidth_1200': 'HardNet++',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output': 'VGG-16-D2-FT',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'Ours-aux',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'Ours-aux-render',
                   'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': 'Ours',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output':'Ours-render'
                   }
lstyl_map = {'d2net_matcher_numfeat_lowinliers': '-',
                   'matching_hardnet_maxphotowidth_1200': '-',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output': '--',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': '--',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': '--',
                   'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': '--',
                   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output':'--'
                   }
color_map = {'d2net_matcher_numfeat_lowinliers': 'mediumseagreen',
                  'matching_hardnet_maxphotowidth_1200': 'orange',
                  'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output': 'darkgreen',
                  'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'indianred',
                  'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'lightcoral',
                  'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': 'darkslateblue',
                  'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_renderbranch_only_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output':'slateblue'
                  }

#plotTranslationErrorPlot(output_dir, "geopose-method-comparison-normal-aux-renderbranch.pdf")
plotRotationErrorPlot(output_dir, "geopose-rotation-method-comparison-normal-aux-renderbranch.pdf")

## comparison of refined vs. best pose
experiment_list = ['d2net_matcher_numfeat_lowinliers',
                   'matching_hardnet_maxphotowidth_1200',
                   #'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output',
                   #'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output',
                   'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output',
                   ]
#plotTranslationErrorPlotRefinedAndBestPose(output_dir, "geopose-method-rp-bp.pdf")
#plotRotationErrorPlotRefinedAndBestPose(output_dir, "geopose-rotation-method-rp-bp.pdf")

## comparison of random semihard, adaptive semihard, adaptive semihard with aux
experiment_list = [ 'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-11-15_15:53:44_MultimodalPatchNet5lShared2l_epoch_10_step_1170000_normalize_output',
                    #'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2020-02-06_14:56:39_MultimodalPatchNet5lShared2l_epoch_27_step_1140000_normalize_output',
                    'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output',
                    #'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output'
                  ]
label_map = {   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-11-15_15:53:44_MultimodalPatchNet5lShared2l_epoch_10_step_1170000_normalize_output': 'Ours-RSH',
                'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2020-02-06_14:56:39_MultimodalPatchNet5lShared2l_epoch_27_step_1140000_normalize_output': 'Ours-RSH-UN',
                'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': 'Ours-ASH',
                'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'Ours-ASH-aux'
            }
lstyl_map = {   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-11-15_15:53:44_MultimodalPatchNet5lShared2l_epoch_10_step_1170000_normalize_output': '-',
'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2020-02-06_14:56:39_MultimodalPatchNet5lShared2l_epoch_27_step_1140000_normalize_output': '-',
                'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': '--',
                'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': '--'
            }
color_map = {   'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-11-15_15:53:44_MultimodalPatchNet5lShared2l_epoch_10_step_1170000_normalize_output': 'darkseagreen',
                'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2020-02-06_14:56:39_MultimodalPatchNet5lShared2l_epoch_27_step_1140000_normalize_output': 'darkgreen',
                'matching_maxphotowidth_1024_allalpsdatasets_matcher_numfeat_lowinliers_2019-11-06_16:15:32_adaptive_semihard_ep1410000_semihard_coeff_0_23_ep2230000_ortler_pcbrejcha_MultimodalPatchNet5lShared2l_epoch_50_step_3070000_normalize_output': 'darkslateblue',
                'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'indianred'
            }

#plotTranslationErrorTable(output_dir, "geopose-method-comparison-random-adaptive-aux.pdf", xstep=250, maxdist=1000)
#plotTranslationErrorPlot(output_dir, "geopose-method-comparison-random-adaptive-aux.pdf", xstep=250, maxdist=1000)
#plotRotationErrorTable(output_dir, "geopose-rotation-method-comparison-random-adaptive-aux.pdf")


# plot noisy improved worsen
#experiment_list = ['matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output']

experiment_list = ['matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output']
label_map = {'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output': 'Ours-aux'}
color_map = {'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-17_14:53:30_MultimodalPatchNet5lShared2l_epoch_21_step_1210000_normalize_output':'indianred'}
#experiment_list = ['matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output']
#label_map = {'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output': 'VGG-16-D2-FT'}
#color_map = {'matching_maxphotowidth_1024_maxres_3000_allalpsdatasets_2019-12-09_14:08:25_LockedSinglemodalVGG16FinetunedShared3l_epoch_11_step_1030000_normalize_output': 'darkgreen'}

#plotNoisyPercentage(experiment_list)
#plotTranslationErrorPlotNoisyPositions(experiment_list, output_dir, 'geoPose3K-ours-conv7l-gaussnoise-1000-t-err.pdf')

plt.show()
