#!/bin/bash

# @Date:   2020-08-06T17:43:32+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:52:27+02:00
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

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_dir>"
    exit 0
fi

INPUT_DIR=$1
OUTPUT_DIR=$(dirname $INPUT_DIR)/rendered_datasets_merged_onterrain

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/real

# for each component
for i in $(ls $INPUT_DIR | egrep "[0-9]+"); do
    COMPONENT=$INPUT_DIR/$i
    cp $COMPONENT/scene_info.txt $OUTPUT_DIR/
    list_prop=$(ls $COMPONENT | grep properties);
    for prop in $list_prop; do
        above_ground_m=$(echo $(tail -n 1 $COMPONENT/$prop))
        if [ -z "$above_ground_m" ]; then
            above_ground_m=0
        fi
        echo "above ground: $above_ground_m"
        above_cmp_gt=$(echo "$above_ground_m > 0" | bc -l)
        above_cmp_lt=$(echo "$above_ground_m <= 100" | bc -l)
        if [ -z "$above_cmp_gt" ]; then
            above_cmp_gt=0
        fi
        if [ -z "$above_cmp_lt" ]; then
            above_cmp_lt=0
        fi
        if [ "$above_cmp_gt" -eq 1 ] && [ "$above_cmp_lt" -eq 1 ]; then
            # above ground and not too far, copy
            base=$(echo $prop | sed 's|_properties.txt||')
            allfiles=$(find $COMPONENT -name "$base*")
            for file in $allfiles; do
                filebase=$(basename $file)
                prop_or_tracks=$(echo "$filebase" | grep "_properties\|_tracks")
                if [ -z $prop_or_tracks ]; then
                    cp $file $OUTPUT_DIR/real
                else
                    cp $file $OUTPUT_DIR/
                fi
                echo "Copied $file, above ground: $above_ground_m"
            done
        fi
    done
done
