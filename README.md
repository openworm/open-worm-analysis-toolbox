movement_validation
===================

Date: 2013-09-22

Authors: Jim Hokanson, Michael Currie

GitHub Issue: https://github.com/openworm/OpenWorm/issues/82

Context: The Shafer Lab [1] at Cambridge University has taken thousands of hours of C. elegans videos, and used image processing to capture the worm's skeleton and contour over time, in 15-minute segments.  This video metadata is stored in hundreds of experiment files available on their FTP server. [2]

This repository should eventually house a test pipeline for the OpenWorm project to run a behavioural phenotyping of its virtual worm, running the same test statistics the Shafer lab used on their worm data.  This test pipeline will hopefully compare these virtual worm statistics against real worm statistics obtained at the Shafer lab.


## Further documentation of worm movement data ##

Located here: https://github.com/JimHokanson/openworm_docs/tree/master/Movement


## Tools used ##

Language: Python

Plotting: matplotlib is a plotting library for the Python programming language and its NumPy numerical mathematics extension.

File processing: The Shafer Lab chose to structure their experiment files using the  “Heirarchical Data Format, Version 5” (HDF5) format , ending with the extension .MAT.  We are using the python module H5PY to extract the information from the Shafer Lab files.

The structure of the files is described at a high level starting on page 183 of Ev Yemeni’s 2011 thesis [.  In [2], for example, there are 23135 frames describing 899.9515 seconds of worm movement (900 seconds is 15 minutes)


## Sources ##

[1] Dr William Schafer's lab at the MRC Laboratory of Molecular Biology http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/

[2] E.g. ftp://anonymous@ftp.mrc-lmb.cam.ac.uk/pub/tjucikas/wormdatabase/results-12-06-08/Laura%20Grundy/unc-8/n491n1192/MT2611/on_food/XX/30m_wait/L/tracker_2/2010-03-19___09_14_57/unc-8%20(rev)%20on%20food%20R_2010_03_19__09_14_57___2___2_features.mat

[3] Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. X. & Schafer, W. R. A database of Caenorhabditis elegans behavioral phenotypes. Nature methods (2013). doi:10.1038/nmeth.2560

In particular see:

http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf
