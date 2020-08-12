# LandscapeAR: Large Scale Outdoor Augmented Reality by Matching Photographs with Terrain Models Using Learned Descriptors

This repository contains official implementation of the ECCV 2020 LandscapeAR
paper. If you use this code in a scientific work, please, cite it:
```
Brejcha, J., Lukáč, M., Hold-Geoffroy, Y., Wang, O., Čadík, M.:
LandscapeAR: Large Scale Outdoor Augmented Reality by Matching Photographs with
Terrain Models Using Learned Descriptors,
In: 16th European Conference on Computer Vision (ECCV), 2020.
```
The code for rendering photographs aligned with a textured Digital Elevation
Model is coupled with the UIST 2018 Immersive Trip Reports paper. If
you also use the rendering part of the project, please, cite it:
```
BREJCHA Jan, LUKÁČ Michal, CHEN Zhili, DIVERDI Stephen a ČADÍK Martin.
Immersive Trip Reports. In: Proceedings of the 31st ACM User Interface Software
and Technology Symposium. Berlín: Association for Computing Machinery,
2018, s. 389-401. ISBN 978-1-4503-5948-1.
```


[Project webpage](http://cphoto.fit.vutbr.cz/LandscapeAR)

## Acknowledgement
This work was supported by project no. LTAIZ19004 Deep-Learning Approach to
Topographical Image Analysis; by the Ministry of Education, Youth and Sports
of the Czech Republic within the activity INTER-EXCELENCE (LT),
subactivity INTER-ACTION (LTA), ID: SMSM2019LTAIZ.

## Introduction
This project consists of several parts, which are aimed at:
- *Structure from Motion with Terrain Reference* for fine grained localization
of a batch of query photographs to the terrain model. This method was
mainly used to create the training datasets. It implements custom matching
stage, with the ability to use D2Net (for keypoint extraction and description),
or our approach (SIFT for keypoint extraction and our trained network for
description). It allows to reconstruct scenes with and also without the
terrain reference. Our custom matches are imported to COLMAP, which then
performs the 3D scene reconstruction. This module is easily extensible by
novel keypoint extractors and descriptors.
- *Training cross-domain descriptor* based on Convolutional Neural Network
(CNN). The code allows to generate training datasets from the reconstructed
scenes and train our descriptor for matching real photographs with rendered
terrain (Digital Elevation Models - DEM) textured with satellite imagery.
- *Cross domain camera pose estimation*. We leverage state-of-the-art
descriptors, such as HardNet, D2Net, and our trained descriptor in order to
match query photograph with the rendered DEM to obtain camera pose with respect
to the virtual globe. For this, EPnP + RANSAC or P4Pf + RANSAC are used,
both combined with Bundle Adjustment to refine the camera pose.

## Installation
This project depends on Python3 and [COLMAP](https://colmap.github.io/) for Structure-from-Motion (SfM) reconstruction. Further dependencies will be
installed automatically. It has been tested on GNU/Linux (Fedora 31), and MacOSX 10.15. For following steps, we assume you are working with a UNIX shell.

Clone the repository including all the submodules. The dependencies,
like HardNet or D2Net are included as submodules.
```
git clone --recurse-submodules https://github.com/brejchajan/LandscapeAR
```

Enter LandscapeAR project, create Python3 virtual environment, activate it,
and install the python dependencies. Optionally, you can skip creating the
virtual environment, if you find your Python3 environment compatible with our
requirements (can be found at `LandscapeAR/python/requirements.txt`).
```
cd landscapeAR
python3 -m venv lsar_venv
source lsar_venv/bin/activate
pip3 install -r python/requirements.txt
```

Download the pretrained models - HardNet, D2Net, NCNet, and our trained
cross-domain model. We packaged them together so that they can be obtained
easily. The downloaded models will be unpacked into the
`LandscapeAR/python/pretrained` directory.
```
cd scripts
./download_data.sh
```
Finally, build the p4pf python module to be able to run P4Pf minimal solver. For more details, see the [paper](http://cmp.felk.cvut.cz/~kukelova/publications/Bujnak-Kukelova-etal-CVPR-2008-final.pdf) and the original [code](http://cmp.felk.cvut.cz/~bujnam1/code/p4pf.zip). Furthermore, the script will also patch the NCNet submodule
so that it is compatible with PyTorch 1.3. The patch can be inspected at
`LandscapeAR/python/patches/ncnet_conv4d_padding_mode.patch`.
```
cd ../python
./build.sh
```
In the last step, please add the following paths on your PYTHONPATH, preferably
by adding following lines to your shell init script
(~/.bashrc for bash, ~/.zshrc for zsh, etc.):
```
export PYTHONPATH="<project path>/LandscapeAR/python/thirdparty/ncnet:$PYTHONPATH"
export PYTHONPATH="<project path>/LandscapeAR/python/thirdparty/d2net:$PYTHONPATH"
export PYTHONPATH="<project path>/LandscapeAR/python/thirdparty/affnet:$PYTHONPATH"
export PYTHONPATH="<project path>/LandscapeAR/python:$PYTHONPATH"
```
Restart your shell afterwards, or run:
```
unset PYTHONPATH
source ~/.bashrc (or your custom shell init script)
```

### Optional dependencies
If you want to use also the [Saddle keypoint detector](http://cmp.felk.cvut.cz/~mishkdmy/papers/SADDLE_ICPR2016.pdf) for keypoint detection, you may optionally install it from https://github.com/brejchajan/saddle_detector.git and add the python module on your PYTHONPATH.

### Installing the DEM renderer
In order to have complete visual localization pipeline, we need to be able to
render the DEM and visualize the registered photographs.
The DEM renderer `itr` is a separate project but it has been added
as a submodule located at `LandscapeAR/itr`.
LandscapeAR project expects the `itr` to be located at the system
PATH. To install it, please follow the instructions at
`LandscapeAR/itr/README.md`, or use a build script, which will
automatically build and install the `itr` together with all
dependencies:
```
cd LandscapeAR/itr
./scripts/build-fedora.sh build install Release 8
```
Feel free to change the number of processes (here 8) based on your PC
configuration. If the process goes well, the `itr` will be installed
into the `install` directory. Please, add this `install` directory to your PATH.

### DEM and satellite data for rendering
The easiest way to render DEM model with textures is to get a Mapbox account
with a free quota [https://www.mapbox.com/](https://www.mapbox.com/), and
update your API key to the example `earth` file: `LandscapeAR/itr-renderer/example.earth`. For your convenience, we also prepared a single GeoTiff
containing the DEM of the whole alps with 1 arcsecond (~30m) resolution, which
we used in all our experiments. You can
[download it here](cphoto.fit.vutbr.cz/LandscapeAR/data/alps_tiles_merged_tiled.tif).
This DEM originally comes from [viewfinderpanoramas.org](http://viewfinderpanoramas.org/), thanks!

## Usage
Let's take a look how to *Download Flickr imagery from area of interest*,
how to *render grid of panorama images* for reconstruction*, how to reconstruct a scene using *Structure from Motion with
Terrain Reference*, how to *generate a training dataset*, how to *train our
descriptor*, and finally, how to *estimate and visualize* camera pose of
a single query image.

### Downloading Flickr imagery from area of interest
In order to obtain training data, we downloaded imagery from several locations
across the European Alps. If you want to download more images to create
even larger dataset, you can use our tool as follows:
```
cd LandscapeAR/python/flickr
python3 flickrAlbumDownload.py <download_dir> <lat> <lon> <radius in km, at most 30>
```
The downloaded images will be stored into <download_dir>/images directory.
The downloaded metadata will be stored into the <download_dir> and can be later
queried with `getFlickrPhotosInfo.py`.

### Rendering a grid of panoramic images from DEM
In order to be able to use the *Structure from Motion with Terrain Reference*
in the next step, we need to render a grid of synthetic panoramic images from
the DEM, you can render them as follows:
```
itr --render-grid <center lat> <center lon> <grid width in meters> <grid height in meters> <offset x in meters> <offset y in meters> <image resolution in px> <output dir> --render-grid-mode perspective --render-grid-num-views 6 <path to .earth file>
```
For offscreen rendering on a headless machine, add `--egl <GPU num>` flag.
Usual parameters are grid width and height = 10000 m, offset x and y is the
distance between consecutive grid points, we use 1000 m, and image resolution
1024 px.

### Structure from Motion with Terrain Reference
We use this technique to build training datasets from the downloaded
photographs and terrain rendered in the previous step. This technique allows
precise alignment (registration, pose estimation) of multiple photographs of
of the same scene to the rendered terrain model. The input are the
rendered images and photographs, for which we know the area where they were captured (the same area as where we rendered the images), but we
don't know exact camera position and orientation of the photographs.

We start by creating a simple directory structure needed for the reconstruction:
```
<reconstruction root>
    <photo>
    <render_uniform>
    image_list.txt
    scene_info.txt
```
We place the downloaded photographs into the directory `photo`, and all rendered
images to the directory `render_uniform`. We copy the file `scene_info.txt` from
the directory with rendered images `render_uniform` to the
`<reconstruction root>`. Then we create a list of all images,
which will be used for the reconstruction:
```
cd <reconstruction root>
find . -name "*.png" >> image_list.txt
find . -name "*.jpg" >> image_list.txt
```
Each line of the list contains relative path from the <reconstruction root>
to a single image. You can also create more image lists to divide the
photgraphs into individual reconstruction batches. For this, use a suffix, such as `image_list_0.txt`, where `0` will be the suffix. The suffix needs to be a natural number. Splitting huge amount of photographs is a good idea no expedite the reconstruction process.
We usually use batches around 1000 photographs in a single batch.

Once the directory structure is created, we can run the reconstruction process:
```
cd LandscapeAR/scripts
./process_colmap_crossdomain.sh <reconstruction root> <suffix (if empty, use "")> <num threads> <gpu index> <'MATCHER OPTIONS'>
```
In order to use CUDA for reconstruction (strongly recommended), please, set the
gpu index for zero or greater. If you don't want to use CUDA, use negative number. The MATCHER OPTIONS need to be encapsulated with apostrophes, as this
is a single argument for the shell script. If you want to be sure to
recompute all features and matches (if you run this script multiple times),
you can add '--recompute-features --recompute-matches'. Also, you can add --d2net
to force using [D2Net](https://github.com/mihaidusmanu/d2-net) for keypoint
extraction and description. If you ommit --d2net, our trained network will be
used.

This script will first run the feature extraction and a custom matching stage.
After this, the matches will be imported to a [COLMAP](https://colmap.github.io/)
database and COLMAP will be used to create the reconstruction.

The reconstructed scenes can be easily exported for future rendering
using `itr`.
```
./exportCrossdomainScenes.sh <reconstruction root>
```
This will create a directory `export` inside `<reconstruction root>`
with all the reconstructed scenes. Each reconstructed scene can be
visualized with `itr` using:
```
cd <reconstruction root>/export
itr --directory <suffix> <path to an earth file>
```
or, they can be also rendered into a dataset using:
```
cd <reconstruction root>/export
itr --directory <suffix> --render-photoviews 1024 --output <output dir> <path to an earth file>
```

### Generating metadata for training
Once the dataset has been reconstructed using SfM with Terrain
Reference and rendered using `--render-photoviews`, we may now
generate the metadata for training. Basically, all image pairs
with some reasonable overlap are found and the corresponding
patches are stored. It can be done by calling following tool:
```
python3 python/genPatchesDataset.py <rendered dataset> real real-synth <output patches metadata>
```
Here `<rendered dataset>` is the `<output dir>` of the previous, rendering stage.

### Training
As the dataset generation is covered, let us now describe the
training procedure.

#### Training data organization
We store all training data in following hierarchy:
```
<all datasets root>
    <dataset 1>
        <rendered_dataset>
        <patches_dataset>
            <train.txt>
            <val.txt>
    <dataset 2>
        <rendered_dataset>
        <patches_dataset>
            <train.txt>
    ...
    <dataset n>
        <rendered_dataset>
        <patches_dataset>
            <train.txt>
    all_training_datasets.txt
    all_validation_datasets.txt
```
We store all datasets in the single directory `<all datasets root>`. As we see, we can have as many datasets (separately reconstructed scenes) as we want. Each dataset directory contains the `<rendered_dataset>` (rendered by the `itr` tool
in previous step), and the `<patches_dataset>` contains the
patches metadata (corresponding patches definition generated
by the `python/getPatchesDataset.py` in the previous step).
Finally, the dataset directories, which are used for training
are listed in `all_training_datasets.txt` (name can be
different), and all validation datasets are listed in `all_validation_datasets.txt`. Put each dataset directory on a separate line. Each training (validation) dataset needs to contain the `<train.txt>` (`<val.txt>`) file, which contains all training (validation) pairs, which are the names of the files contained in `<patches_dataset>`. The advantage of this hierarchy is that a singe scene can be divided to training and
validation parts (which should be disjoint).

#### Training procedure
The training can be run easily:
```
python3 python/trainPatchesDescriptors.py -l <log_and_model_out_dir> --name <name_of_the_run> --num_epochs 50 --num_workers 1 --batch_size 300 --positive_threshold 1 --distance_type 3D --cuda --learning_rate 0.00001 --margin 0.2 --color_jitter --maxres 3000 --hardnet --hardnet_filter_dist --hardnet_filter_dist_values 50 1000000 --hardnet_orig --normalize_output --adaptive_mining --adaptive_mining_step 0.0 --adaptive_mining_coeff 0.23 --no_symmetric --uniform_negatives --architecture MultimodalPatchNet5lShared2l <all datasets root> <all datasets root>/all_training_datasets.txt <all datasets root>/all_validation_datasets.txt <path to temporary cache (ideally SSD)>
```
With these parameters we trained the CNN presented in our paper. For additional auxiliary loss (as presented in our paper), please add `--los_aux`. The `--adaptive_mining` allows to gradually increase the training difficulty by step defined in `--adaptive_mining_step`. The hardness starts at value defined with `--adaptive_mining_coeff`. With the current setting, we simply fixed the adaptive mining value to 0.23, since it worked the best for. However, it seems that it is heavily dependent on margin (`--margin`), and when `--margin 0.1` is used, then the adaptive mining coeff could be set to 0.35 by `--adaptive_mining_coeff 0.35`. Higher values cause that the training collapses into a singular point, as described in our paper.

### Pose estimation
We have now covered almost the complete pipeline, the only
missing piece is the direct camera pose estimation of the query
image with respect to the rendered terrain model. For this,
we use:
```
python3 python/findPose.py --working-directory <working_directory> --matching-dir <matching_dir> -no --best-buddy-refine --maxres 3000 <query_image.jpg> Ours-aux --voting-cnt 3 --cuda -l <log_and_model_out_dir> [--gps latitude longitude] --earth-file <path_to_your_earth_file.earth>
```
This tool will run the complete pose estimation algorithm as described in our
paper. The `<working_directory>` will be used to store the rendered imagery,
and the `<matching_dir>` will be a subdirectory of the `<working_dir>` to store
the results of matching. If the corresponding rendered panorama does not exist
on the `<working_directory>`, the `itr` tool will be called to render it
(so it must be installed). Furthermore, if the photograph does not contain
GPS in its EXIF, user needs to specify the position by adding `--gps <lat> <lon>`
on the command line.

Since we wanted to be able also to experiment with new networks and architectures,
we developed a simpler tool, which matches a single photo with a single rendered image.
If you want to try it out, you can do as follows:
```
python3 python/analyzeMatches.py <query photograph> <rendered image.png> <network name (e.g, Ours)> matches -no -c --maxres 3000 -l python/pretrained/landscapeAR --sift-keypoints
```
This runs pose estimation by matching the photo to the render and then uses
the rendered depth map and camera pose to estimate the pose of the photograph.
We assume, that the rendered image has been rendered by the `itr` tool, as
described in previous sections. The `analyzeMatches.py` and `findPose.py`
allows to use different keypoint detectors -- you can experiment with them
by adding different flags, such as `--dense-uniform-keypoints` (to extract)
keypoints on a regular uniform grid, etc. For more, see the help of both tools.
By default (if not specified on the command line), `analyzeMatches.py` uses
dense uniform keypoint detection, whereas `findPose.py` uses SIFT keypoints.
