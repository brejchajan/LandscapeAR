# @Date:   2020-08-08T17:04:58+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:51:16+02:00
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

import argparse as ap
import os

from pose_estimation.PoseFinder import PoseFinder


def buildArgumentParser():
    parser = ap.ArgumentParser()

    parser.add_argument("query_image", help="Path to the query image. The \
    image needs to contain GPS coordinates in exif which will be used as an \
    initial position estimate.")
    parser.add_argument("snapshot", help="""Restores model from
                        the snapshot with given name. If no model is specified
                        using -m, loads latest model.""")
    parser.add_argument("--working-directory", help="Directory which will \
    be used to store rendered images and results. Defaults to current \
    directory.", default=os.getcwd())
    parser.add_argument("--earth-file", help="Earth xml file to be used for \
    image rendering using itr. \
    Default: ../itr/example.earth",
    default="../itr/example.earth")
    parser.add_argument("-l", "--log_dir", help="""Directory used for saving
                        training progress logs and models. Defaults to cwd
                        when not specified.""", default=os.getcwd())
    parser.add_argument("-m", "--model_name", help="""Specify exact model name
                        to be restored using -r option.""")
    parser.add_argument("-c", "--cuda", action="store_true", help="If this \
                        flag is used, cuda will be used for neural network\
                        processing.")
    parser.add_argument("-no", "--normalize_output", action="store_true",
                        help="If set, the descriptors are normalized to unit \
                        hypersphere.")
    parser.add_argument("--grid-options", help="Specify grid radius and offset.",
                        nargs=2, default = [0, 1], type=int)
    parser.add_argument("--no-voting", action='store_true', help="Disables \
                        the voting stage and does exhaustive matching \
                        instead.")
    parser.add_argument("--gps", nargs=2, type=float, help="\
                        If GPS <lat> <lon> is defined, it will be used instead \
                        of the GPS tag in the photo.")
    parser.add_argument("--best-buddy-refine", action='store_true',
                        help="Use best buddy matching during the final \
                        refinement matching using 2D and all reprojected 3D \
                        points.")
    parser.add_argument("--voting-cnt", default=3, help="Specify the number \
                        of top candidates for which the pose estimation will \
                        be run after the voting step.", type=int)
    parser.add_argument("--use-depth", action='store_true')
    parser.add_argument("--use-normals", action='store_true')
    parser.add_argument("--use-silhouettes", action='store_true')
    parser.add_argument("--fov", help="Set field-of-view in degrees. \
                        Overrides FOV from exif, or from dataset.", type=float)
    parser.add_argument("--matching-dir", default="matching", help="specify \
                        the name of the output directory. Default='matching'.")
    parser.add_argument("--use-hardnet", action='store_true',
                        help='Uses original hardnet for patch description.')
    parser.add_argument("--maxres", type=int, default=4096,
                        help="Resolution for recalculating images size \
                        w.r.t its FOV. Maxres corrsponds to FOV=180deg.")
    parser.add_argument("--fcn-keypoints", action='store_true', help="Use \
                        fully convolutional variant of our keypoint \
                        two-branch net. This allows dense extraction of \
                        keypoints from the whole image at one step. Uses \
                        FOV of the image to detect keypoints at single \
                        scale")
    parser.add_argument("--fcn-keypoints-multiscale",
                        action='store_true', help="Use \
                        fully convolutional variant of our keypoint \
                        two-branch net. This allows dense extraction of \
                        keypoints from the whole image at one step. Does not \
                        use FOV and detects keypoints on multiple scales.")
    parser.add_argument("--dense-uniform-keypoints", action='store_true',
                        help="Use densely uniformly sampled keypoints \
                        instead of any other keypoint detector.")
    parser.add_argument("--dense-halton-keypoints", action='store_true',
                        help="Use densely uniformly sampled keypoints \
                        instead of any other keypoint detector.")
    parser.add_argument("--saddle-keypoints", action='store_true')
    parser.add_argument("--stride", help="Stride used to compute the dense \
                        representations. Default=12", default=12, type=int)
    parser.add_argument("--d2net", action='store_true', help=" \
                        User original D2Net keypoints and descriptors for \
                        matching.")
    parser.add_argument("--ncnet", action='store_true', help=" \
                        Uses ncnet matching for pose estimation step.")
    parser.add_argument("--sift-descriptors", help="Uses SIFT descriptors \
                        for matching, neural descriptors are not used.",
                        action='store_true')
    parser.add_argument("--p4pf", help="P4Pf and our own RANSAC will be used \
                        for matching. No FOV initialization is needed, since \
                        P4Pf estimates the focal length.", action='store_true')
    parser.add_argument("--p4pf-epnp-iterative",
                        help="P4Pf and our own RANSAC will be used \
                        to estimate initial solution of camera pose and FOV. \
                        The estimated FOV is then used to re-estimate the \
                        camera pose iteratively using \
                        epnp + ransac + bundle adjustment until the number of \
                        inliers keeps improving.",
                        action='store_true')
    parser.add_argument("--epnpor", help="EPNP and our own RANSAC will be used\
                        for matching.", action='store_true')

    return parser


if __name__ == "__main__":
    parser = buildArgumentParser()
    args = parser.parse_args()

    poseFinder = PoseFinder(args)
    poseFinder.findPose()
