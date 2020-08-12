# @Date:   2020-08-04T10:44:39+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:49:12+02:00
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

import os

from FeatureExtractor import D2NetFeatureExtractor, CrossDomainFeatureExtractor
from Matcher import ExhaustiveMatcher, SpatialMatcher
from database import COLMAPDatabase
import torch
import argparse as ap


def buildArgumentParser():
    parser = ap.ArgumentParser(
        "Tool for deep feature extraction and matching."
    )

    parser.add_argument("input_dir", help="Input dir containing the scene for \
                        reconstruction. It must contain file \
                        image_list<suffix>.txt with each line containing one \
                        path to the image relative to the input dir.")
    parser.add_argument("suffix", help="Suffix to determine a specific image \
    list which will be used for matching. This is suitable if only a \
    particular selection of images shall be matched.")
    parser.add_argument("--only-features", action='store_true', help="\
    Only features will be detected, matching step will be ommitted.")
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument(
        "--processes", type=int, help="Set number of \
    concurrent processes used for matching. Usable if cuda is NOT used or if\
    GPU has large amount of memory (e.g., 8GB is needed per process).",
        default=1
    )
    parser.add_argument("--recompute-features", action='store_true', help="\
    Cached features will be overwritten with newly computed features.")
    parser.add_argument("--recompute-matches", action='store_true', help="\
    Cached matches will be overwritten with newly computed matches.")
    parser.add_argument("--database-path", help="Specifies the database\
    path. If not specified, input_dir is used as default.")
    parser.add_argument("--spatial-matching", type=float, help="\
    All pairs of images closer than N km will be matched with each other.")
    parser.add_argument("--d2net", action='store_true', help="If set d2net\
                        will be used to extract keypoints and descriptors.\
                        Otherwise our cross-domain network will be used.")
    parser.add_argument("--disable-photo2photo-matching", action='store_true',
                        help="Only photo-to-render and render-to-render\
                        will be matched, photo-to-photo will be skipped.")

    return parser


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = buildArgumentParser()
    args = parser.parse_args()

    reserve_gpu = torch.randn(1)
    if args.cuda:
        reserve_gpu = reserve_gpu.cuda()

    input_dir = args.input_dir
    suffix = args.suffix
    if len(suffix) > 0:
        suffix = "_" + suffix

    image_list_file = os.path.join(input_dir, "image_list" + suffix + ".txt")
    database_dir = input_dir
    if args.database_path:
        database_dir = args.database_path
    database_path = os.path.join(database_dir, "database" + suffix + ".db")

    if args.d2net:
        extractor = D2NetFeatureExtractor(
            input_dir, image_list_file,
            cuda=args.cuda, recompute=args.recompute_features
        )
    else:
        extractor = CrossDomainFeatureExtractor(
            input_dir, image_list_file,
            cuda=args.cuda, recompute=args.recompute_features
        )
    extractor.extract()

    if not args.only_features:
        if os.path.exists(database_path):
            print("WARNING: database already exists, it will be removed.")
            os.remove(database_path)

        print("Opening database: ", database_path)
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()

        if args.spatial_matching:
            matcher = SpatialMatcher(
                input_dir, image_list_file,
                db, suffix, recompute=args.recompute_matches, cuda=args.cuda,
                num_processes=args.processes,
                max_distance=args.spatial_matching,
                disable_photo2photo_matching=args.disable_photo2photo_matching
            )
        else:
            matcher = ExhaustiveMatcher(
                input_dir, image_list_file,
                db, suffix, recompute=args.recompute_matches, cuda=args.cuda,
                num_processes=args.processes,
                disable_photo2photo_matching=args.disable_photo2photo_matching
            )
        matcher.match()
