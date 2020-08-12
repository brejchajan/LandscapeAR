#!/bin/bash

# @Date:   2020-08-10T12:37:07+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:53:02+02:00
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

if [ $# -lt 4 ]; then
    echo "Usage: $0 <input_path> <suffix> <num_threads> <gpuindex> ['<MATCHER OPTIONS>']"
    echo "MATCHER OPTIONS: encapsulate all matcher flags between"
    echo "  appstrophes, so that shell interprets them as a single argument."
    echo "  For avaliable options, please see run "
    echo "  'python3 ../python/sfm/matching.py'."
    exit 0
fi
GPU=-1
INPUT_PATH=$1
SUFFIX=$2
NUM_THREADS=$3
GPU=$4

use_gpu=1
custom_flags=""
if [ "$#" -eq 5 ]; then
    custom_flags=$5
fi

if [ "$GPU" -eq -1 ]; then
    GPU=0
    use_gpu=0
else
    CUDA="--cuda"
fi

# preform feature extraction and matching outside COLMAP
#python3 ../python/sfm/matching.py "$INPUT_PATH" "$SUFFIX" --processes $NUM_THREADS $CUDA $custom_flags

if [ "$(echo $SUFFIX | wc -c)" -gt 1 ]; then
    SUFFIX="_"${SUFFIX}
fi
DATABASE="database${SUFFIX}.db"

colmap matches_importer --match_list_path $INPUT_PATH/matches_pairs${SUFFIX}.txt --database_path $INPUT_PATH/$DATABASE --SiftMatching.num_threads $NUM_THREADS --SiftMatching.gpu_index $GPU --SiftMatching.use_gpu $use_gpu
colmap exhaustive_matcher --database_path $INPUT_PATH/$DATABASE --SiftMatching.num_threads $NUM_THREADS --SiftMatching.gpu_index $GPU --SiftMatching.use_gpu $use_gpu

mkdir -p $INPUT_PATH/render_photo_model${SUFFIX}
sparse_bytes=$(cat $INPUT_PATH/sparse_model${SUFFIX}/cameras.txt sparse_model${SUFFIX}/images.txt sparse_model${SUFFIX}/points3D.txt | wc -c)
if [ "$sparse_bytes" -gt 0 ]; then
    echo "##################################"
    echo "RUNNING SfM with terrain reference"
    echo "##################################"
    # we have rendered images, so triangulate the renders only and then register photos with fixed renders
    mkdir -p $INPUT_PATH/triangulated_model${SUFFIX}
    colmap point_triangulator --database_path $INPUT_PATH/$DATABASE --image_path $INPUT_PATH --input_path $INPUT_PATH/sparse_model${SUFFIX} --output_path $INPUT_PATH/triangulated_model${SUFFIX} --Mapper.num_threads $NUM_THREADS
    colmap mapper --database_path $INPUT_PATH/$DATABASE --image_path $INPUT_PATH --input_path $INPUT_PATH/triangulated_model${SUFFIX} --output_path $INPUT_PATH/render_photo_model${SUFFIX} --Mapper.fix_existing_images 1 --Mapper.num_threads $NUM_THREADS
    mkdir -p $INPUT_PATH/render_photo_model_geo${SUFFIX}
    colmap model_aligner --input_path $INPUT_PATH/render_photo_model${SUFFIX} --output_path $INPUT_PATH/render_photo_model_geo${SUFFIX} --ref_images_path $INPUT_PATH/georegistration${SUFFIX}.txt --robust_alignment_max_error 10
    colmap model_converter --input_path $INPUT_PATH/render_photo_model_geo${SUFFIX} --output_path $INPUT_PATH/sfm_data${SUFFIX}.nvm --output_type NVM
else
    echo "running classical SfM"
    # we have only photos, run normal reconstruction
    colmap mapper --database_path $INPUT_PATH/$DATABASE --image_path $INPUT_PATH --output_path $INPUT_PATH/render_photo_model${SUFFIX} --Mapper.num_threads $NUM_THREADS
fi
