# @Date:   2020-08-06T16:21:25+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:52:01+02:00
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

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math


def convertHistogram(limits, values, nbins):
    """Converts the tensorflow histogram stored in bins with unequal width
    to bins with equal widths.
    @limits [numpy.array] bin edges
    @values [numpy.array] containing the bin values
    @nbins [int] number of bins to create.
    """
    bin_width = (limits[-1] - limits[0]) / nbins

    left = 0
    new_vals = []
    new_bins = []
    overflow = 0
    underflow = 0
    currbin = 0
    diff_overflow = 0

    for idx in range(0, limits.shape[0]):
        diff = limits[idx] - limits[left] + diff_overflow
        diff_overflow = diff - bin_width
        if diff_overflow >= 0:
            overflow_new = (diff_overflow/(limits[idx] - limits[idx - 1]))
            underflow = (1.0 - overflow_new) * values[idx]
            new_bin = np.sum(values[left:idx-1]) + underflow + overflow
            overflow = overflow_new * values[idx]
            new_vals.append(new_bin)
            new_bins.append(currbin)
            currbin = currbin + bin_width
            left = idx
            if diff_overflow >= bin_width:
                div = math.floor(diff_overflow / bin_width)
                frac = ((div * bin_width) / diff_overflow) * overflow
                overflow_new = overflow - frac
                for i in range(0, int(div)):
                    new_bin = frac / div
                    new_vals.append(new_bin)
                    new_bins.append(currbin)
                    currbin = currbin + bin_width

                overflow = overflow_new
                diff_overflow = diff_overflow - div * bin_width
                assert diff_overflow < bin_width
        else:
            diff_overflow = 0

    new_vals = np.array(new_vals)
    new_bins = np.array(new_bins)
    return new_vals, new_bins, bin_width

def plotHistogram(dists_file, label, linestyle, top=0):
    #plot sift histogram
    dists_sift = np.load(dists_file)
    h_sift = np.histogram(np.min(dists_sift[:, :(top+1)], axis=1), bins=nbins)
    d_sift = h_sift[0].astype(np.float)
    bin_w = h_sift[1][1] - h_sift[1][0]
    d_sift /= (d_sift * bin_w).sum()
    cs_sift = np.cumsum(d_sift * bin_w)
    bins_sift = h_sift[1][0:1000]
    sel = bins_sift < 50
    plt.plot(bins_sift[sel], cs_sift[sel], label=label, linestyle=linestyle)


def findBestTFEvents(tfevents, steps, name, colors=None, linestyles=None):
    print("Find best tfevents")
    if steps is None:
        plotAll = True
        steps = []
    idx = 0
    max_step = 0
    max_val = 0
    try:
        for e in tf.compat.v1.train.summary_iterator(tfevents):
            for v in e.summary.value:
                if v.tag == "dist2d_first":
                    print("step", e.step)
                    limits = np.array(v.histo.bucket_limit)
                    values = np.array(v.histo.bucket)
                    widths = limits[:-2] - limits[1:-1]
                    new_vals, new_bins, bin_width = convertHistogram(limits, values, nbins)
                    new_vals /= (new_vals * bin_width).sum()
                    #plt.bar(new_bins, new_vals * bin_width, width=bin_width)
                    sel = new_bins < 10
                    print(np.sum(sel))
                    cs = np.cumsum(new_vals * bin_width)
                    selected = cs[sel]
                    selected = selected[-1]
                    print(selected)
                    if max_val < selected:
                        max_val = selected
                        max_step = e.step
    except RuntimeError as re:
        print("ERROR DatasetError.", re.message)
    finally:
        return max_step, max_val


def plotHistogramsTFEvents(tfevents, steps, name, colors=None, linestyles=None):
    plotAll = False
    if steps is None:
        plotAll = True
        steps = []
    idx = 0
    try:
        for e in tf.train.summary_iterator(tfevents):
            if plotAll or e.step in steps:
                print("step", e.step)
                for v in e.summary.value:
                    if v.tag == "dist2d_first":
                        limits = np.array(v.histo.bucket_limit)
                        values = np.array(v.histo.bucket)
                        widths = limits[:-2] - limits[1:-1]
                        new_vals, new_bins, bin_width = convertHistogram(limits, values, nbins)
                        new_vals /= (new_vals * bin_width).sum()
                        #plt.bar(new_bins, new_vals * bin_width, width=bin_width)
                        sel = new_bins < 50
                        cs = np.cumsum(new_vals * bin_width)
                        if linestyles is not None:
                            if isinstance(linestyles, list):
                                lstyle = linestyles[idx]
                            else:
                                lstyle = linestyles

                        if linestyles is None and colors is not None:
                            plt.plot(new_bins[sel], cs[sel], label=name + "-step-" + str(e.step), color=colors[idx])
                        elif linestyles is None and colors is None:
                            plt.plot(new_bins[sel], cs[sel], label=name + "-step-" + str(e.step))
                        elif linestyles is not None and colors is None:
                            plt.plot(new_bins[sel], cs[sel], label=name + "-step-" + str(e.step), linestyle=lstyle)
                        else:
                            plt.plot(new_bins[sel], cs[sel], label=name + "-step-" + str(e.step), linestyle=lstyle, color=colors[idx])
                        idx += 1
    except:
        print("DatasetError.")


    plt.legend(loc='upper right')

if __name__ == "__main__":
    nbins = 1000
    plotAll = False

    # Example usage - existing tfevents must be used
    tfevents_15_35 = "tfevents_dir/events.out.tfevents.1550850057.supergpu7.fit.vutbr.cz"

    plotHistogramsTFEvents(tfevents_15_35, [], 'conv7_m_0_1_maxhard_0_35', linestyles='--')

    plt.legend(loc='upper right')
    plt.xlabel("2D distance [px]")
    plt.ylabel("fraction of queries")
    plt.show()
    plt.savefig("graphs/conv7_m_0_1_maxhard_0_35.pdf")
