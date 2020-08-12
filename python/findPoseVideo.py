# @Date:   2020-08-06T15:44:24+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:51:19+02:00
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

from pose_estimation.VideoPoseFinder import VideoPoseFinder


def main():
    #snapshot = "2019-03-25_12:58:21"
    #model = "MultimodalPatchNet5lShared2l_epoch_2_step_2340000"
    #snapshot = "2019-05-27_10:35:09"
    snapshot = "2019-12-17_14:53:30"
    model = "MultimodalPatchNet5lShared2l_epoch_21_step_1210000"
    earth_file = "/home/ibrejcha/hdd_data/data/tms_datasets/yosemite/yosemite.earth"
    log_dir = "/mnt/matylda1/ibrejcha/devel/adobetrips/python"
    working_path = "/home/ibrejcha/hdd_data/data/matching_video/glacier_point"
    video_path = "/home/ibrejcha/hdd_data/data/matching_video/glacier_point/glacier_point.mp4"
    initial_gps = [37.730577, -119.573669, 0]
    fov = 70

    vpf = VideoPoseFinder(video_path, initial_gps, fov, working_path,
                          snapshot, model, earth_file, log_dir)
    vpf.processVideo()

if __name__ == "__main__":
    main()
