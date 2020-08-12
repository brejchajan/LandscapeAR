# @Date:   2020-08-07T17:23:54+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:52:47+02:00
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
mkdir -p $INPUT_DIR/export

for i in $(ls $INPUT_DIR | grep nvm); do
    SUFFIX=$(echo $i | egrep -o "[0-9]+\.nvm")
    if [ ! -z "$SUFFIX" ]; then
        IDX=${SUFFIX%.*}
        SUFFIX="_${IDX}"
    else
        IDX=0
        SUFFIX=""
        if [ $(ls $INPUT_DIR | grep nvm | wc -l) -gt 1 ]; then
            echo "Error unable to find suffix and there is more than one item to export. Exitting."
            exit 0
        fi
    fi
    photodirs=$(ls $INPUT_DIR | egrep "^photo_[0-9]+$")
    mkdir -p $INPUT_DIR/export/$IDX
    ln -s $INPUT_DIR/$i $INPUT_DIR/export/${IDX}/sfm_data.nvm
    if [ ! -z "$photodirs" ]; then
	ln -s $INPUT_DIR/photo_${IDX} $INPUT_DIR/export/${IDX}/photo_${IDX}
    else
	ln -s $INPUT_DIR/photo $INPUT_DIR/export/${IDX}/photo
    fi
    ln -s $INPUT_DIR/render_uniform $INPUT_DIR/export/${IDX}/render_uniform
done
