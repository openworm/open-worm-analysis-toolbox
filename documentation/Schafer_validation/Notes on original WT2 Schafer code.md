Further Information on the original Schafer Lab Work
====================================================

Dr. Yemini at the [Schafer
Lab](<http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/>) at Cambridge
University has taken thousands of hours of C. elegans videos ([here is
an example](http://www.youtube.com/watch?v=5FAiSgl55p0)), using their
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

Technical descriptions of the features calculated are [available in
hyperlinked
form](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/Yemini%20Supplemental%20Data/Schafer%20Lab%20Feature%20Descriptions.md),
in documents adapted from the original Schafer Lab Supplemental
Documentation to [4].

Information on [downloading data from the Schafer Lab
repository](https://github.com/openworm/OpenWorm/issues/82), from
[@slarson](https://github.com/slarson) in the initial issue prompting
the creation of this repo.

Further analysis of differences conducted 4 May 2017 by @ver228:
================================================================

After lot of testing and correcting minor bugs I can confirm that most of the features match fairly well (attached plots). Here are my comments.  PLOTS:

- https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/N2_A_24C_L_5_2015_06_16__19_54_27___feat_comparison.pdf
- https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/N2_A_24C_R_4_2015_06_16__19_22_48___feat_comparison.pdf

1) The features below do not change sign for clockwise/anticlockwise ventral orientation in Segworm but they do in Openworm:
eigenworms
locomotion.foraging_bends.*
locomotion.velocity.*.direction
path.curvature

2) path.duration.* match the distribution but no the scatter plots. This is because the vectors used in the scatter plots are not time vectors, but rather the non-zeros elements of an X-Y grid. There must be some difference in the matlab and python implementations while ordering the non-zeros elements of an array, but since the order of this vector does not have a physical meaning I would prefer not to worry about it.

3) posture.direction.* values differ due to the stage alignment. In segworm the skeletons coordinates are rotated to match the stage point of reference. In tierpsy the stage coordinates are rotated to match the video point of reference. The change is to make life easier if somebody in the future would like to implement a new computer vision algorithm and would like to compare it with the existing skeletons. I would not worry too much about this feature since their value depend on the arena coordinates that in our case has an arbitrary value.

4) tierpsy skeletons are consistently shorter than segworm. I used the openworm toolbox to calculate the features on segworm skeletons, and compare them with tierpsy. I observed a small shift in the distributions of around 3 pixels. I think this is due to the smoothing I used in tierpsy for the skeletons. However, since the difference between skeletons when plotted next to each other is very small I would not worry too much about it.

5) posture.bends.*.std_dev absolute values are consistently larger in segworm than in tierpsy specially for the head and tail. This differences disappear if I use openworm toolbox on segworm skeletons. Since the bend calculation is a very straight forward operation I assume the discrepancies are due to the way angles are calculated. I seem to remember that segworm calculates the skeletons angles before doing the skeleton normalization. Strangely, even if the bends stdev differ their means are very similar in segworm and tierpsy.

6) The events in locomotion.motion_events.paused/forward.* are a bit different between the trackers. This difference seems to be at least in part due to difference in the skeletons, both tierpsy and segworm give different results while being feed to the openworm toolbox. Strangely this does not happens with locomotion.motion_events.backward. I assume the difference is in part due to the different skeleton lengths. The threshold for the events are calculated as 5% (or 2.5%) of the worm length. I tried to change the threshold factor to compensate the change in length but the extra or missing events are still there. I assume the event features can be quite prone to detect small movements as events.

7) Remember that locomotion.velocity.*.direction differ due to the bug of dividing by fps or by time.


Finally, for feature work I think it would be good to change the definition of several features to ensure the continuity in the time vectors, like in locomotion.crawling_bends.* or posture.bends.*.std_dev. This is not a problem if we are using only the features distributions, but could be bad while dealing with time series. 
This correction should be straight forward for features that depend on an angle since the discontinuities are due to the cyclic nature of angle coordinates. The current code deals with the jumps using something like x[x<-180] += 360 but this only works for a single jump. It would be necessary to do a similar step as what is done to calculate the angles for the eigenworms. 
More challenging will be to deal with features that have their sign assigned by other features like the bends std. This is tricky since if the signed feature is very small it can oscillate around zero and produce large jumps in the secondary feature. Probably using a ratio or something similar could solve this problem. 






Sources
-------

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
