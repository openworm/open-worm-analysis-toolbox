Further Information on the original Schafer Lab Work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dr. Yemini at the [Schafer
Lab]((http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/) at Cambridge
University has taken thousands of hours of C. elegans videos (`here is
an example <http://www.youtube.com/watch?v=5FAiSgl55p0>`__), using their
proprietary hardware/software package called Worm Tracker 2 (WT2). The
videos are processed by the WT2 software's image processing to capture
the worm's skeleton and contour over time, in 15-minute segments. This
video metadata is stored in hundreds of experiment files available on
their FTP server. [2]

The structure of the HDF5 files is described at a high level starting on
page 183 of Dr. Eviatar Yemeni’s 2011 thesis [3] or in the supplemental
material of his 2013 Nature Methods paper [5]. In [2], for example,
there are 23135 frames describing 899.9515 seconds of worm movement (900
seconds is 15 minutes)

Technical descriptions of the features calculated are `available in
hyperlinked
form <https://github.com/openworm/movement_validation/blob/master/documentation/Yemini%20Supplemental%20Data/Schafer%20Lab%20Feature%20Descriptions.md>`__,
in documents adapted from the original Schafer Lab Supplemental
Documentation to [4].

Information on `downloading data from the Schafer Lab
repository <https://github.com/openworm/OpenWorm/issues/82>`__, from
`@slarson <https://github.com/slarson>`__ in the initial issue prompting
the creation of this repo.

Sources
-------

[1] `Dr William Schafer's
lab <http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/>`__ at Cambridge University's MRC Laboratory
of Molecular Biology

[2] E.g. ``ftp://anonymous@ftp.mrc-lmb.cam.ac.uk/pub/``
``tjucikas/wormdatabase/results-12-06-08/Laura%20Grundy/``
``unc-8/n491n1192/MT2611/on_food/XX/30m_wait/L/``
``tracker_2/2010-03-19___09_14_57/unc-8%20(rev)%20on%20food``
``%20R_2010_03_19__09_14_57___2___2_features.mat``

[3] Yemini E 2011 High-throughput, single-worm tracking and analysis in
Caenorhabditis elegans (University of Cambridge)

[4] Yemini, et al. A database of Caenorhabditis elegans behavioral
phenotypes. Nature methods (2013). doi:10.1038/nmeth.2560

[5] `Yemini, et al. A database of Caenorhabditis elegans behavioral
phenotypes. Nature methods (2013). Supplementary Data.
nmeth.2560-S1 <http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf>`__
