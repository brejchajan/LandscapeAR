#!/usr/bin/env python

# @Date:   2020-08-08T17:03:40+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified time: 2020-08-11T17:51:38+02:00
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


from tqdm import tqdm
import os.path
from argparse import ArgumentParser
import re

def generateSets(rendered_dataset_path, patches_dataset_path, setname):
    """ Generates the list of pairs of views based on the input list of images.
    The output list of pairs contains only images specified in the input list
    which shall be located at <rendered_dataset_path>/real/<setname>.txt
    The pairs of images are loaded from the patches dataset by listing
    <patches_dataset_path>.
    @type string
    @param rendered_dataset_path path to the dataset of rendered images
    using itr tool.
    @type string
    @param patches_dataset_path path to the patches dataset generated from the
    rendered dataset using genPatchesDataset.py.
    @type string
    @param setname name of the input set of images, usually train, val, test.
    To each setname a corresponding text file with .txt extension shall be
    located at <rendered_dataset_path>/real/<setname>.txt.
    """

    listfile = os.path.join(patches_dataset_path, "list_all.txt")
    if os.path.isfile(listfile):
        with open(listfile) as f:
            list_all = f.read().splitlines()
    else:
        print("Listing the patches dataset directory, this may take a while. \
        To speed-up this process, you may create list_all.txt file with listed\
        directory: " + patches_dataset_path + " inside that directory.")
        l_all = os.listdir(patches_dataset_path)
        list_all = []
        for i in tqdm(l_all):
            if os.path.isdir(os.path.join(patches_dataset_path, i)):
                list_all.append(i)

    setpath = os.path.join(os.path.join(rendered_dataset_path, "real"),
                           setname + ".txt")
    with open(setpath) as f:
        sel = f.read().splitlines()
        selected = []
        for s in sel:
            parts = os.path.splitext(s)
            selected.append(parts[0])

    output = os.path.join(patches_dataset_path, setname + "_python.txt")
    with open(output, 'w') as f:
        for i in tqdm(list_all):
            parts = i.split("-")
            n1 = parts[0]
            n2 = ''.join(parts[1:])
            n2 = re.sub(r'_random_[0-9]+.*', '', n2)
            print(n1, n2)
            if (n1 in selected) and (n2 in selected):
                f.write(i + "\n")


def buildArgumentParser():
    ap = ArgumentParser(description="Tool for generating list of pairs of \
    images as train, validation, and test set. Uses input list of image names \
    to create output set of pairs containing only combinations of images \
    from the input list.")
    ap.add_argument("rendered_dataset_path", help="Path to the dataset of \
    rendered images. Needs to contain \
    subdirectory real, from where input train.txt, val.txt, test.txt lists of \
    single images are read to form output pairs within each set.")
    ap.add_argument("setname", help="Name of the set corresponding to the name\
    of the file containing the list of source images (usually one of \
    train.txt, val.txt, test.txt)")
    ap.add_argument("patches_dataset_path", help="Directory containing the \
    generated patches between image pairs. This directory will be also used as\
    the output directory for the result set (called <setname>_python.txt)")
    return ap


def main():
    ap = buildArgumentParser()
    args = ap.parse_args()
    generateSets(args.rendered_dataset_path,
                 args.patches_dataset_path, args.setname)


if __name__ == "__main__":
    main()
