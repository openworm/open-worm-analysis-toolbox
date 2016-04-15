# *CWL* - Updated: 3/30/2016
#   Tested Against:
#   1. Ubuntu 14.04 LTS via VirtualBox 5.0.16 on Mac OS X 10.11.4
#   2. Ubuntu 14.04 LTS (HVM) via Amazon AWS t2.micro instance (note: Not enough RAM on t2.micro)

# Linux (Ubuntu)
# (Python 3.5; modify for Python 2.7 on the first line)
# (Python 2.7, 3.3, 3.4, 3.5 are supported)
# ---------------------------------------------------------
PYTHON_VERSION=3.5
# *CWL* Give users a convenient way to modify comments for desired python version.
#PYTHON_VERSION=2.7
# Switch the above line to PYTHON_VERSION=2.7 if desired

# Script to configure a fresh ubuntu instance for OWAT
# *CWL* TODO Consider the use of expect and pass for automated
#   script responses.
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

# *CWL* It seems not every fresh Ubuntu system comes with git pre-installed.
sudo apt-get install -y git
# *CWL* AWS Ubuntu seems to have issues with miniconda's pyqt4 package 
#       without python-qt4 installed.
sudo apt-get install -y python-qt4

# Install condas, numpy, scipy, etc.
cd ~
if [[ $PYTHON_VERSION == 2.7 ]]; then
# *CWL* incorrect defaults
#    MINICONDA_DIR=~/miniconda3
    MINICONDA_DIR=~/miniconda2
#    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
else
#    MINICONDA_DIR=~/miniconda2
    MINICONDA_DIR=~/miniconda3
#    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
fi
chmod +x miniconda.sh
./miniconda.sh -b
# Add this Python path to $PATH so typing `python` does not go to `/src/bin/python` first, which
# is a symlink by default to `/usr/bin/python2.7`.
# Also, we make this change permanent by adding this line to ~/.profile
#echo "PATH=~/miniconda3/bin:$PATH; export PATH" >> ~/.profile
# *CWL* Use the set environtment variable.
#       Expanding absolute path in profile is fragile.
echo "PATH=$MINICONDA_DIR/bin:\$PATH; export PATH" >> ~/.profile

# *CWL* AWS Ubuntu seems to have problems getting these variables set up properly
#       This is still fragile for now, since not everyone uses en_US.UTF-8, but it
#       is still better than an attempt to run plot_example.py failing each time.
printf '%s\n%s\n' 'export LC_ALL=en_US.UTF-8' 'export LANG=en_US.UTF-8' >> ~/.profile

# Put the path change into immediate effect
. ~/.profile
conda install --yes python=$PYTHON_VERSION atlas numpy scipy matplotlib nose pandas statsmodels h5py seaborn

# Install OpenCV
sudo apt-get install -y build-essential
sudo apt-get install -y make
sudo apt-get install -y cmake
#DEPS_DIR=/home/ubuntu
# *CWL* Previous folder really is a fragile hardcode                            
DEPS_DIR=$HOME
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
# *CWL* Don't leave folder creation to chance                                   
mkdir -p ~/github
cd ~/github  # Please make this directory if it doesn't exist
#git clone git@github.com:OpenWorm/open-worm-analysis-toolbox
# *CWL* Permissions are denied using the above git command.
git clone https://github.com/openworm/open-worm-analysis-toolbox.git
#cd open-worm-analysis-toolbox
# *CWL* Incorrect folder path.
cd open-worm-analysis-toolbox/open_worm_analysis_toolbox

#sudo /bin/cp travis_config.txt user_config.py
# *CWL* creates a root owned file that cannot be edited otherwise
/bin/cp travis_config.txt user_config.py
# Please edit user_config.py to point to the correct folder below.
# *CWL* Use sed to automate the above step. Note that the -i option is slightly
#   fragile as it used to have different semantics for Mac OS X. El Capitan has
#   fixed this but uses a different default, generating a backup with a suffix
#   "-e" whilst Linux will simply default to replacing the file with no backup.
#   To prevent this fragile option from messing things up, we will ask sed to
#   always generate a ".bak" backup file. This is good practice anyways.
#
# Users may set EXAMPLE_DIR to anything they wish
EXAMPLE_DIR=$HOME/example_data
sed -i.bak -e "s|^EXAMPLE_DATA_PATH = .*$|EXAMPLE_DATA_PATH = \'${EXAMPLE_DIR}\'|g" user_config.py

#mkdir ~/example_data
#cd ~/example_data
# *CWL* Modified for variable use. Also made the operation idempotent
mkdir -p $EXAMPLE_DIR
cd $EXAMPLE_DIR
wget "https://googledrive.com/host/0B7to9gBdZEyGNWtWUElWVzVxc0E/example_contour_and_skeleton_info.mat" -O example_contour_and_skeleton_info.mat
wget "https://drive.google.com/uc?export=download&id=0B7to9gBdZEyGX2tFQ1JyRzdUYUE" -O example_video_feature_file.mat
wget "https://drive.google.com/uc?export=download&id=0B7to9gBdZEyGakg5U3loVUktRm8" -O example_video_norm_worm.mat
chmod 777 *.mat
