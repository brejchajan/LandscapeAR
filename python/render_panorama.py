# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-11T17:52:04+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:48:28+02:00
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
import glob
import pymap3d as pm3d
import pymap3d.vincenty
import numpy as np

def buildArgumentParser():
    parser = ap.ArgumentParser()

    parser.add_argument("--output-directory", help="Directory which will \
    be used to store rendered images and results. Defaults to current \
    directory.", default=os.getcwd())
    parser.add_argument("--earth-file", help="Earth xml file to be used for \
    image rendering using itr. \
    Default: ~/data/tms_datasets/alps_matterhorn_wallis_90km/alps_matterhorn_wallis.earth",
    default="~/data/tms_datasets/alps_matterhorn_wallis_90km/alps_matterhorn_wallis.earth")
    parser.add_argument("--fov", help="Set field-of-view of rendered views \
                        in degrees. Default=60", type=float, default=60)
    parser.add_argument("--grid-options", help="Specify grid radius and offset.",
                        nargs=2, default = [0, 1], type=int)
    parser.add_argument("--resolution", default=1200, type=int,
                        help="Resolution of single panorama image in pixels. \
                        Must be integer. Default=1200.")
    parser.add_argument("--with-egl", help="If this flag \
                        is used, the rendering will be done offscreen \
                        (EGL must be supported, only NVIDIA, itr\
                        needs to be compiled with it.). Specify GPU index \
                        as a parameter.")
    parser.add_argument("gps", nargs=2, type=float, help="\
                        If GPS <lat> <lon> is defined, it will be used instead \
                        of the GPS tag in the photo.")
    parser.add_argument("--add-noise", type=float, help="Randomly move the \
                        GPS position according to the gaussian distribution\
                        with the original GPS position as mean, and this \
                        parameter as standard deviation in meters.")

    return parser

def checkImagesAlreadyRendered(output_dir):
    outpath = os.path.join(output_dir, "*_depth.txt.gz")
    files_depth = glob.glob(outpath)
    if len(files_depth) >= 12:
        # all images are already rendered
        return True
    return False

def renderPanorama(args):
    lat = args.gps[0]
    lon = args.gps[1]
    r = str(args.grid_options[0])
    o = str(args.grid_options[1])

    if args.add_noise:
        sigma = args.add_noise
        randheading = np.random.uniform(0, 180)
        randdist = np.random.normal(0, sigma)
        if randdist < 0:
            randdist = -randdist
            randheading = 360 - randheading

        lat2, lon2, back_az = pm3d.vincenty.vreckon(lat, lon, randdist, randheading)
        print("Added noise to original gps: ", lat, lon,
              "new gps:", lat2, lon2,
              "distance", randdist, "heading", randheading)
        lat = lat2
        lon = lon2

    egl = ""
    if args.with_egl:
        egl = "--egl " + str(args.with_egl)
    else:
        print("WITHOUT EGL!")

    print(args.output_directory)
    if (checkImagesAlreadyRendered(args.output_directory)):
        print("Skipping, already rendered.")
        return

    cmd = "source ~/.bashrc; itr --render-grid " + str(lat) + " " + str(lon) + " " + r + " " + r + " " + o + " " + o + " " + str(args.resolution) + " " + args.output_directory + " --render-grid-mode perspective " + egl + " " + args.earth_file
    print(cmd)
    os.system(cmd)

    if (not checkImagesAlreadyRendered(args.output_directory)):
        raise RuntimeError("The images were not rendered. Please check \n\
        whether itr is installed and works properly.")


if __name__ == "__main__":
    parser = buildArgumentParser()
    args = parser.parse_args()
    renderPanorama(args)
