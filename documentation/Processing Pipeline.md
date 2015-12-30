Open Worm Analysis Toolbox Video Processing Pipeline
=============================================

This repository intends to code the processing pipeline from raw video
to statistics. Much of the code comes from the Schafer Lab's Worm
Tracker 2 (WT2) software.

This repo currently takes **STEP 3** as the starting point, and does not
code for steps 2 or 3, because the Schafer Lab already has an [analysis
tool
online](http://www.mrc-lmb.cam.ac.uk/wormtracker/index.php?action=analysis)
that transforms the raw video into normalized data.

It might be useful to also port the machine vision processing for **STEP
2** and **STEP 3**, but only if we wish to improve upon their
algorithms. This is tempting since their algorithm cannot process nearly
20% of the raw video frames (see p.131 of Yemini's dissertation) (e.g.
it cannot deal with coiling or omega turns) This would involve working
with the SegWorm repo and/or porting it to Python (from MATLAB).

PIPELINE: STEPS 1 to 9: Outline
-------------------------------

**STEP 1**: Conduct experiment / simulation, end up with raw video.

**STEP 2**: Machine vision processing to obtain measurements

**STEP 3**: Normalize measurements to frames of 49 data points for each
measurement type

**STEP 4**: Stitch the "blocks" into one .mat file.

**STEP 5**: Calculate feature information for each frame

**STEP 6**: Calculate extended feature information for each frame.

**STEP 7**: Calculate statistics summarizing the features across all
frames

**STEP 8**: Add statistics to a database

**STEP 9**: Run reports from the database

Schafer Lab Code Sources
------------------------

**STEP 1**: n/a

**STEPS 2-3**: [Worm Analysis Toolbox
1.3.4](http://www.mrc-lmb.cam.ac.uk/wormtracker/index.php?action=analysis).

**STEP 4**: Added by Jim as `createObjectFromFiles` in
[NormalizedWorm](https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%40normalized_worm/normalized_worm.m)

**STEPS 5-6**: Some of the code is in the SegwormMatlabClasses repo,
under
[tree/master/oldFeatures](https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/oldFeatures).

**STEPS 7-9**: n/a

PIPELINE: STEPS 1 to 9: Details
-------------------------------

### 1. Raw video

Conduct experiment: - case: REAL WORM: Capture video of worm movement
using a test protocol, in tandem with a control worm. - case: VIRTUAL
WORM: Run simulation engine, output video.

Raw video, plus tracking plate movement data + other metadata (time of
filming, vulva location, whether worm flipped during video, strain of
worm used, Lab name, etc)

![](images/STEP%200-1.bmp?raw=true)

Credit: OpenWorm / <http://dorkutopia.com/tag/xbox-one/>

### 2. Measurements

*(Machine vision processing step.)*

This gives the worm contour and skeleton, etc.

![](images/STEP%202.gif?raw=true)

Credit: Ev Yemini

### 3. Normalized measurements

*(Normalize each worm video frame to just 49 points; necessary for
frame-by-frame comparability.)*

The existing WT2 code covering **STEP 2** and **STEP 3** is
[available](http://www.mrc-lmb.cam.ac.uk/wormtracker/index.php?action=analysis)
at the Schafer lab site.

The Schafer Lab code creates a bunch of files, some of which are the
norm files: - normBlock1 - normBlock2 - normBlock3 - ... -
normBlock10 -mec-4 (u253) off food
x\_2010\_04\_21\_\_17\_19\_20\_\_1\_failedFrames.mat (in the .data
folder)

Many of the features are derived rather easily from the skeleton and
contour, such as the area of the head which can be calculated relatively
easy from the contour.

When a model worm is created, these steps will need to be reproduced.
This is described in [OpenWorm Issue
144](https://github.com/openworm/OpenWorm/issues/144).

**Relevant Code:**

-   [blob/master/Worms/Features/wormDataInfo.m](https://github.com/openworm/SegWorm/blob/master/Worms/Features/wormDataInfo.m)
    (This is the original code which starts to describe this expanded
    set of base features)
-   [feature/roots.m](https://github.com/JimHokanson/SegWorm/blob/classes/new_code/%2Bseg_worm/%2Bfeature/roots.m)

[SegWorm/Pipeline/featureProcess.m](https://github.com/JimHokanson/mrc_wormtracker_gui/blob/master/SegWorm/Pipeline/featureProcess.m) -
This file is responsible for creating the feature mat files

### 4. Combine normalized measurments into one file

Jim wrote a function called
[NormalizedWorm](https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%40normalized_worm/normalized_worm.m).createObjectFromFiles
that stitches these "blocks" together, into an easy-to-deal-with file
called norm\_obj.mat. (Previously, this merging of the blocks was
embedded within the feature processing code, which complicated the
feature processing code unnecessarily.)

### 5. Worm features

*(Feature calculation in Python based on WT2 code.)*

"Features" are properities of the worm, derived from the measurements
data. For instance, the area of the head (a feature) is calculated from
"skeleton" and width data (which are considered "measurements")

### 6. Worm features (expanded)

*(Feature calculation in Python based on WT2 code.)*

Using the expanded set of base features the Schafer lab has computed a
much larger set of features. As an example, the worm length provides 4
features, one overall, and three more when computed during forward
movement, backwards movements, and when paused.

For more on this, see: [Expanded Features](Expanded_Features.md)

**Relevant
Code** -<https://github.com/openworm/SegWorm/blob/master/Worms/Statistics/wormStatsInfo.m>

### 7. Worm statistics

*(Stats calculation in Python based on WT2 code.)*

The result of conversion from the video values to those used during
statistical testing. The Schafer lab stats calculation code excludes
data in certain situations, normalizes some values, and appears to
quantize the frame data to reduce memory requirements. This process will
need to eventually be described here in more detail.

### 8. Database of statistics on multiple worms

This database could perhaps be made available to researchers everywhere
to use, to act as a central repository for C. elegans behavioural
statistics.

In fact, the Schafer lab currently has such a database,
[wormbehavior.mrc-lmb.cam.ac.uk](http://wormbehavior.mrc-lmb.cam.ac.uk/)

### 9. Reports

Reports are run from data in the statistics database, and can take the
form of a summary pixel grid, pairwise boxplots, and other charts.

![](images/STEP%207.bmp?raw=true)

Credit: Ev Yemini
