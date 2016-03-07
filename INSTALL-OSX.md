For Developers: Installing the repository on Mac OS X
=====================================================

```bash
# By Chee Wai Lee
# Created: 3/4/2016
# Modified: 3/4/2016
# Mac OS X 10.11.3 (El Capitan)
# Python 3.5 - there are some issues with OpenCV for now. Punting for now.
# Python 2.7
# ----------------------------
PYTHON_VERSION=2.7
# CWL Notes: Replace $PWD as needed
export INSTALL_DIR=$PWD

# CWL Notes: This setup pseudo-script assumes a setup sequence from a fresh OS X
#   virtual machine each time (e.g. on some Travis-CI server). Developers trying to
#   set up this environment on their personal machines will probably need to skip 
#   certain steps from anything other than a fresh OS image.
#
#   Developers are also encouraged to use flexible software stacks (modules environments,
#   virtualenv, conda environments) as much as 
#   possible so missteps do not end up catastrophically messing up your machines.
#
#   TODO: This is still a pseudo-script that assumes interactive user input, which
#     obviously will not work inside the Travis-CI environment.

# Install Homebrew
# --------------------------
# CWL Notes: The preferred way to flexibly manage Linux-like packages on Mac OS X is via Homebrew
# To setup Homebrew, the following one-time steps should be taken (see http://brew.sh)
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install wget via Homebrew 
brew install wget

# Install miniconda and associated science packages
#
# CWL Notes: I work with many different software stacks and environments.
#     As such my preference is to install into an encapsulated package
#     environment, and activate on command using the modules environment:
#
#       http://modules.sourceforge.net/
#   
#   This script will however assume the user does the default act of
#     installing into the default ~/miniconda3 folder
#     and will require explicit changes to one's .profile or .bash_profile
#     to prepend the miniconda path.
cd $INSTALL_DIR; mkdir -p install_temp; cd install_temp
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
export PATH=~/miniconda3/bin:$PATH
conda update conda
# CWL Notes: 
#  1) I prefer the use of sandboxed environments where possible, and conda
#     provides it, hence the use of "conda create -n owat <packages>"
#  2) atlas refuses to be recognized explicitly as a conda package for osx-64
#     hence the following installation sequence follow by the use of pip.
conda install anaconda-client
# CWL Notes:
# For python 3:
# conda create -n owat numpy scipy matplotlib nose pandas statsmodels h5py seaborn
# For python 2 (using the above installation of miniconda3):
conda create -n owat numpy scipy matplotlib nose pandas statsmodels h5py seaborn python=2
source activate owat
pip install atlas

# Install opencv via homebrew
brew install opencv

# Configuration steps for opencv
# TODO: Figure out what the original sequence is trying to do.

# Download and set up open-worm-analysis-toolbox
cd $INSTALL_DIR; mkdir -p github; cd github
git clone https://github.com/openworm/open-worm-analysis-toolbox.git
# old instructions do not work for me - git clone git@github.com:OpenWorm/open-worm-analysis-toolbox
cd open-worm-analysis-toolbox/open_worm_analysis_toolbox
cp travis_config.txt user_config.py
# Replace path with where the real examples are installed
#  backup created as user_config.py.tmp
sed -i.tmp 's@^EXAMPLE_DATA_PATH.*$@EXAMPLE_DATA_PATH = '\'"$INSTALL_DIR"/example_data\''@g' user_config.py

# Get some real example data
cd $INSTALL_DIR; mkdir -p example_data; cd example_data
wget "https://googledrive.com/host/0B7to9gBdZEyGNWtWUElWVzVxc0E/example_contour_and_skeleton_info.mat" -O example_contour_and_skeleton_info.mat
wget "https://drive.google.com/uc?export=download&id=0B7to9gBdZEyGX2tFQ1JyRzdUYUE" -O example_video_feature_file.mat
wget "https://drive.google.com/uc?export=download&id=0B7to9gBdZEyGakg5U3loVUktRm8" -O example_video_norm_worm.mat
chmod 777 *.mat

# CWL Notes: Mac-specific post-processing and workarounds

# 1. To handle the somewhat common "ValueError: unknown local: UTF-8" error,
# http://stackoverflow.com/questions/19961239/pelican-3-3-pelican-quickstart-error-valueerror-unknown-locale-utf-8
#    suggests setting two environment variables which I've done here, but on my own system
#    I've packaged this with miniconda3's modules environment for flexibility and for software
#    package orthogonality (somewhat) issues.
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. Helping our code find opencv (in this case my homebrew-installed version)
export PYTHONPATH=/usr/local/Cellar/opencv/2.4.12_2/lib/python2.7/site-packages:$PYTHONPATH

# 3. matplotlib.pyplot.show() (see plt.show() in 
#    open_worm_analysis_toolbox/prefeatures/worm_plotter.py) 
#    has some serious issues with Mac OS X.
#    
#  TODO: find a reasonable workaround for what we need for Mac OS X. Interactive window
#    display is apparently a no-go.

```
