[![Stories in
Ready](https://badge.waffle.io/openworm/open-worm-analysis-toolbox.png?label=ready&title=Ready)](https://waffle.io/openworm/open-worm-analysis-toolbox)

Movement Validation
===================

![](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/images/Validation%20Strategy.png?raw=true)

Credit: OpenWorm

![](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/images/Test%20process.png?raw=true)

Images: C. elegans by Bob Goldstein, UNC Chapel Hill
<http://bio.unc.edu/people/faculty/goldstein/> Freely licensed. Contour
credit: MRC Schafer Lab. Simulated worm: OpenWorm.

Objectives:
-----------

1.  This repository should house a test
    [pipeline](https://github.com/MichaelCurrie/open-worm-analysis-toolbox/blob/master/documentation/Processing%20Pipeline.md)
    for the OpenWorm project to run a behavioural phenotyping of its
    virtual worm, using the same statistical tests the Schafer lab used
    on their real worm data.\
    [Overall OpenWorm Milestone:
    \#19](https://github.com/openworm/OpenWorm/issues?milestone=19&state=open)
2.  In achieving goal \#1, this repository will also be an open source
    version of the Schafer Lab's Worm Tracker 2 (WT2) analysis pipeline
    that goes from raw real worm videos to worm measurements detected by
    machine vision, to a selection of calculated worm behavioural
    features like average speed.
3.  Also in achieving goal \#1, we hope to have a system for tracking
    the statistics of hundreds of real and virtual worm strains in a
    database.

### Codebase starting point and real worm data source

The Schafer Lab [1] at Cambridge University has taken thousands of hours
of C. elegans videos ([here is an
example](http://www.youtube.com/watch?v=5FAiSgl55p0)), using their
proprietary hardware/software package called Worm Tracker 2 (WT2). The
videos are processed by the WT2 software's image processing to capture
the worm's skeleton and contour over time, in 15-minute segments. This
video metadata is stored in hundreds of experiment files available on
their FTP server. [2]

We are using the WT2 analysis software, written in MATLAB, as a starting
point for our own code development work.

This full codebase is available in the
[SegWorm](https://github.com/openworm/SegWorm) repo.

A revised version of this code (revised by @JimHokanson), also in
MatLab, is available in the
[SegWormMatlabClasses](https://github.com/JimHokanson/SegwormMatlabClasses/)
repo.

This repo, intended to be a full Python translation of the
SegWormMatlabClasses repo, is the only one of the three being actively
worked on.

### Further Documentation

The structure of the HDF5 files is described at a high level starting on
page 183 of Ev Yemeni’s 2011 thesis [3] or in the supplemental material
of his 2013 Nature Methods paper [5]. In [2], for example, there are
23135 frames describing 899.9515 seconds of worm movement (900 seconds
is 15 minutes)

Technical descriptions of the features calculated are [available in
hyperlinked
form](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/Yemini%20Supplemental%20Data/Schafer%20Lab%20Feature%20Descriptions.md),
in documents adapted from the original Schafer Lab Supplemental
Documentation to [4].

Further documentation of worm movement data is available at
@JimHokanson's
[openworm\_docs](https://github.com/JimHokanson/openworm_docs/tree/master/Projects/Movement)
repo.

[Monthly Progress
Reports](https://drive.google.com/folderview?id=0B9dU7zPD0s_LMm5RMGZGX2JEeGc&usp=sharing)

[Movement Validation: White
Paper](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/Movement%20Validation%20White%20Paper.md)

[OpenWorm Journal Club featuring Ev
Yemini](https://www.youtube.com/watch?v=YdBGbn_g_ls)

Information on [downloading data from the Schafer Lab
repository](https://github.com/openworm/OpenWorm/issues/82), from
[@slarson](https://github.com/slarson) in the initial issue prompting
the creation of this repo.

[Commit Often, Perfect Later, Publish Once: Git best
practices](http://sethrobertson.github.io/GitBestPractices/)

### Sources

[1] [Dr William Schafer's
lab](http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/) at Cambridge
University's MRC Laboratory of Molecular Biology

[2] E.g. `ftp://anonymous@ftp.mrc-lmb.cam.ac.uk/pub/`
`tjucikas/wormdatabase/results-12-06-08/Laura%20Grundy/`
`unc-8/n491n1192/MT2611/on_food/XX/30m_wait/L/`
`tracker_2/2010-03-19___09_14_57/unc-8%20(rev)%20on%20food`
`%20R_2010_03_19__09_14_57___2___2_features.mat`

[3] Yemini E 2011 High-throughput, single-worm tracking and analysis in
Caenorhabditis elegans (University of Cambridge)

[4] Yemini, et al. A database of Caenorhabditis elegans behavioral
phenotypes. Nature methods (2013). <doi:10.1038/nmeth.2560>

[5] [Yemini, et al. A database of Caenorhabditis elegans behavioral
phenotypes. Nature methods (2013). Supplementary Data.
nmeth.2560-S1](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)
