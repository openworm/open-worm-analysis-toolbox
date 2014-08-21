[![Stories in Ready](https://badge.waffle.io/openworm/movement_validation.png?label=ready&title=Ready)](https://waffle.io/openworm/movement_validation)
Movement Validation
===================

Contributors: @JimHokanson, @MichaelCurrie, the [Movement Validation Team](https://github.com/orgs/openworm/teams/movement-validation)

![](https://github.com/openworm/movement_validation/blob/master/documentation/images/Validation%20Strategy.png?raw=true)

<sub><sup>Credit: OpenWorm</sub></sup>
     
![](https://github.com/openworm/movement_validation/blob/master/documentation/images/Test%20process.png?raw=true)

<sub><sup>Images: C. elegans by Bob Goldstein, UNC Chapel Hill http://bio.unc.edu/people/faculty/goldstein/  Freely licensed. Contour credit: MRC Schafer Lab.  Simulated worm: OpenWorm.</sub></sup>

###Objectives:###

1. This repository should house a test [pipeline](https://github.com/MichaelCurrie/movement_validation/blob/master/documentation/Processing%20Pipeline.md) for the OpenWorm project to run a behavioural phenotyping of its virtual worm, using the same statistical tests the Schafer lab used on their real worm data.  
[**Overall OpenWorm Milestone**: *#19*](https://github.com/openworm/OpenWorm/issues?milestone=19&state=open)  

2. In achieving goal #1, this repository will also be an open source version of the Schafer Lab's Worm Tracker 2 (WT2) analysis pipeline that goes from raw real worm videos to worm measurements detected by machine vision, to a selection of calculated worm behavioural features like average speed.

3. Also in achieving goal #1, we hope to have a system for tracking the statistics of hundreds of real and virtual worm strains in a database.


## Codebase starting point and real worm data source ##

The Schafer Lab [1] at Cambridge University has taken thousands of hours of C. elegans videos ([here is an example](http://www.youtube.com/watch?v=5FAiSgl55p0)), using their proprietary hardware/software package called Worm Tracker 2 (WT2).  The videos are processed by the WT2 software's image processing to capture the worm's skeleton and contour over time, in 15-minute segments.  This video metadata is stored in hundreds of experiment files available on their FTP server. [2]

We are using the WT2 analysis software, written in MATLAB, as a starting point for our own code development work.

This full codebase is available in the [SegWorm](https://github.com/openworm/SegWorm) repo.

A revised version of this code (revised by @JimHokanson), also in MatLab, is available in the [SegWormMatlabClasses](https://github.com/JimHokanson/SegwormMatlabClasses/) repo.

This repo, intended to be a full Python translation of the SegWormMatlabClasses repo, is the only one of the three being actively worked on.


## Further Documentation ##

Technical descriptions of the features calculated are [available in hyperlinked form](https://github.com/openworm/movement_validation/blob/master/documentation/Yemini%20Supplemental%20Data/Schafer%20Lab%20Feature%20Descriptions.md), in documents adapted from the original Schafer Lab Supplemental Documentation to [4].

Further documentation of worm movement data is available at @JimHokanson's [openworm_docs](https://github.com/JimHokanson/openworm_docs/tree/master/Projects/Movement) repo.

[Monthly Progress Reports](https://drive.google.com/folderview?id=0B9dU7zPD0s_LMm5RMGZGX2JEeGc&usp=sharing)

[How to use the worm plotter](https://github.com/openworm/movement_validation/blob/master/documentation/How%20to%20use%20WormPlotter.md)

[Movement Validation: White Paper](https://github.com/openworm/movement_validation/blob/master/documentation/Movement%20Validation%20White%20Paper.md)

Information on [downloading data from the Schafer Lab repository](https://github.com/openworm/OpenWorm/issues/82), from @slarson in the initial issue prompting the creation of this repo.

[Commit Often, Perfect Later, Publish Once: Git best practices](http://sethrobertson.github.io/GitBestPractices/)


## Tools used ##

**Language:** Python 3.x.  The code requires use of scientific computing packages (numpy, h5py), and as such getting the packages properly installed can be tricky. As such, if working in Windows, we recommend using [Spyder IDE](https://code.google.com/p/spyderlib/) and the [WinPython distribution](http://winpython.sourceforge.net/) for Windows.  ( (Note, this isn't required)

**HDF reader:** [HDF viewer](http://www.hdfgroup.org/hdf-java-html/hdfview/) - this can be used for debugging the file structure

**Plotting:** matplotlib is a plotting library for the Python programming language and its NumPy numerical mathematics extension.  FFMPEG is used for video processing.

**File processing:** The Schafer Lab chose to structure their experiment files using the  “Heirarchical Data Format, Version 5” (HDF5) format , ending with the extension .MAT.  We are using the Python module H5PY to extract the information from the Schafer Lab files.

The structure of the files is described at a high level starting on page 183 of Ev Yemeni’s 2011 thesis [3] or in the supplemental material of his 2013 Nature Methods paper [5].  In [2], for example, there are 23135 frames describing 899.9515 seconds of worm movement (900 seconds is 15 minutes)

**Markdown editor:** http://markdownpad.com/

**Data repository:** We store the .mat files that flow between the steps of our pipeline, in a DropBox shared folder "worm_data".

###Installing and running the movement_validation repo:###
1. Install Python 3.x and matplotlib
2. numpy is a library for Python that is a dependency in this repo.  If you have numpy version less than 1.8, you will need to either install it, or:
     a. Save [nanfunctions.py](https://github.com/numpy/numpy/blob/0cfa4ed4ee39aaa94e4059c6394a4ed75a8e3d6c/numpy/lib/nanfunctions.py) to your Python library directory, in the `numpy/lib/` directory, and
     b. Append the lines ```__all__ += nanfunctions.__all__``` and ```from .nanfunctions import *``` at the relevant places to numpy/lib/```__init__.py```.
3. Install Shapely, which is available [for Windows here](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
4. Clone the repository to your computer
5. Ask @MichaelCurrie or @JimHokanson to share the worm_data DropBox folder with you
6. In the wormpy directory there should be a file `user_config_example.txt`.  Rename this file as `user_config.py`.  It will be ignored by github since it is in the .gitignore file.  So in `user_config.py`, specify your computer's specific DropBox folder root directory and other settings.
7. Try running `wormpy_example.py`
8. Hopefully it runs and shows plots of the worm's contour!

Contact @MichaelCurrie for troubleshooting these steps.


###License###

movement_validation, and indeed all of OpenWorm, is free software under the [MIT license](http://opensource.org/licenses/MIT).

## Sources ##

[1] [Dr William Schafer's lab](http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/) at Cambridge University's MRC Laboratory of Molecular Biology

[2] E.g. ftp://anonymous@ftp.mrc-lmb.cam.ac.uk/pub/tjucikas/wormdatabase/results-12-06-08/Laura%20Grundy/unc-8/n491n1192/MT2611/on_food/XX/30m_wait/L/tracker_2/2010-03-19___09_14_57/unc-8%20(rev)%20on%20food%20R_2010_03_19__09_14_57___2___2_features.mat

[3] Yemini E 2011 High-throughput, single-worm tracking and analysis in Caenorhabditis elegans (University of Cambridge)

[4] Yemini, et al.  A database of Caenorhabditis elegans behavioral phenotypes. Nature methods (2013). doi:10.1038/nmeth.2560

[5] [Yemini, et al.  A database of Caenorhabditis elegans behavioral phenotypes. Nature methods (2013).  Supplementary Data.  nmeth.2560-S1](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)
