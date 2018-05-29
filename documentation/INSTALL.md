For developers: Installing the repository
==========================================================

```bash
# Linux (Ubuntu)
# (Python 3.5; modify for Python 2.7 on the first line)
# (Python 2.7, 3.3, 3.4, 3.5 are supported)
# ---------------------------------------------------------
PYTHON_VERSION=3.5
# Switch the above line to PYTHON_VERSION=2.7 if desired

# Script to configure a fresh ubuntu instance for OWAT
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

# Install condas, numpy, scipy, etc.
cd ~
if [[ $PYTHON_VERSION == 2.7 ]]; then
    MINICONDA_DIR=~/miniconda3
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
else
    MINICONDA_DIR=~/miniconda2
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
fi
chmod +x miniconda.sh
./miniconda.sh -b
# Add this Python path to $PATH so typing `python` does not go to `/src/bin/python` first, which
# is a symlink by default to `/usr/bin/python2.7`.
# Also, we make this change permanent by adding this line to ~/.profile
echo "PATH=~/miniconda3/bin:$PATH; export PATH" >> ~/.profile
# Put the path change into immediate effect
. ~/.profile
conda install --yes python=$PYTHON_VERSION atlas numpy scipy matplotlib nose pandas statsmodels h5py seaborn

# Install OpenCV
sudo apt-get install -y build-essential
sudo apt-get install -y make
sudo apt-get install -y cmake
DEPS_DIR=/home/ubuntu
OPENCV_BUILD_DIR=$DEPS_DIR/opencv/build
sudo git clone --depth 1 https://github.com/Itseez/opencv.git $DEPS_DIR/opencv
sudo mkdir $OPENCV_BUILD_DIR && cd $OPENCV_BUILD_DIR
sudo cmake -DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DWITH_CUDA=OFF -DENABLE_AVX=ON -DWITH_OPENGL=ON -DWITH_OPENCL=ON -DWITH_IPP=ON -DWITH_TBB=ON -DWITH_EIGEN=ON -DWITH_V4L=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") -DPYTHON_EXECUTABLE=$(which python3) -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") ..
sudo make -j4
sudo make install

# https://gist.github.com/itguy51/4239282
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf
sudo ldconfig
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig" | sudo tee -a /etc/bash.bashrc
echo "export PKG_CONFIG_PATH" | sudo tee -a /etc/bash.bashrc
export PYTHONPATH=$OPENCV_BUILD_DIR/lib/python3.3/site-packages:$PYTHONPATH 

# Get open-worm-analysis-toolbox
cd ~/github  # Please make this directory if it doesn't exist
git clone git@github.com:OpenWorm/open-worm-analysis-toolbox
cd open-worm-analysis-toolbox

sudo /bin/cp travis_config.txt user_config.py
# Please edit user_config.py to point to the correct folder below.
mkdir ~/example_data
cd ~/example_data
wget "https://googledrive.com/host/0B7to9gBdZEyGNWtWUElWVzVxc0E/example_contour_and_skeleton_info.mat" -O example_contour_and_skeleton_info.mat
wget "https://drive.google.com/uc?export=download&id=0B7to9gBdZEyGX2tFQ1JyRzdUYUE" -O example_video_feature_file.mat
wget "https://drive.google.com/uc?export=download&id=0B7to9gBdZEyGakg5U3loVUktRm8" -O example_video_norm_worm.mat
chmod 777 *.mat
```

Windows
------------------

1.  Install Python (2 or 3), matplotlib, numpy, scipy, seaborn, pandas, OpenCV.  You can install most of these using WinPython, and the rest, like scipy, pandas, and OpenCV, can be downloaded pre-compiled from http://www.lfd.uci.edu/~gohlke/pythonlibs/.  Once downloaded use `pip` to install the egg (Python 2) or wheel (Python 3) downloaded.
2.  Clone this GitHub repository to your computer.
*[If you want to skip using Google Drive for the below 3 steps, just visit the Example Data folder link below and manually download all the content and place on a folder on your computer.  It isn't updated too frequently so it shouldn't cause a problem to not be syncing all the time]*
3.  [OPTIONAL] If you don't already have an account, get a [Google
    Drive](https://www.google.com/intl/en/drive/) account.
4.  [OPTIONAL] Install [Google Drive for
    desktop](https://tools.google.com/dlpage/drive).
5.  [OPTIONAL] Using Google Drive, sync with the folder
    [example\_data/](https://drive.google.com/folderview?id=0B7to9gBdZEyGNWtWUElWVzVxc0E&usp=sharing),
    which is a subfolder of
    `OpenWorm/OpenWorm Public/Movement Analysis/`.
6.  In the `open-worm-analysis-toolbox/open-worm-analysis-toolbox` folder there should
    be a file `user_config_example.txt`. Rename this file as
    `user_config.py`. It will be ignored by GitHub since it is in the
    `.gitignore` file. So in `user_config.py`, specify your computer's
    specific Google Drive root directory and other settings.
7.  Try running one of the scripts in the `examples/` folder.
8.  Hopefully it runs successfully! If not:

Please contact the [OpenWorm-discuss mailing
list](https://groups.google.com/forum/#!forum/openworm-discuss) if you
encounter issues with the above steps.

You can also try running `test_setup.py` in the `/tools` folder to check
if your setup is correctly configured. Note that this tool is not 100% comprehensive.

Tools used
==========

**Language:** Python 2 or 3. The code requires use of scientific computing
packages (numpy, h5py), and as such getting the packages properly
installed can be tricky. As such, if working in Windows, we recommend
using [Spyder IDE](https://code.google.com/p/spyderlib/) and the
[WinPython distribution](http://winpython.sourceforge.net/) for Windows.
(Note, this isn't required)

**Plotting:** matplotlib is a plotting library for the Python
programming language and its NumPy numerical mathematics extension.
FFMPEG is used for video processing.

**File processing:** The Schafer Lab chose to structure their experiment
files using the “Heirarchical Data Format, Version 5”
[(HDF5)](http://en.wikipedia.org/wiki/Hierarchical_Data_Format#HDF5/)
format, ending with the extension .MAT. We are using the Python module
H5PY to extract the information from these files.

**Data storage:** Google Drive. To store examples of worm videos and
HDF5 (.mat) feature files so the open-worm-analysis-toolbox package can be put
through its paces.

**HDF reader:** [HDF
viewer](http://www.hdfgroup.org/hdf-java-html/hdfview/). Optional. This
tool can be used for debugging the file structure of the data files.

