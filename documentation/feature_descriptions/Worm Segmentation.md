Worm Segmentation
=================

From Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. X. & Schafer,
W. R. A database of Caenorhabditis elegans behavioral phenotypes. Nat.
Methods (2013). `doi:10.1038/nmeth.2560`

Video frames are extracted using the Matlab videoIO toolbox by Gerald
Dalley. There is a sharp contrast between the worm and background in our
video images. Worm pixels are segmented from the background using the
Otsu method (Otsu 1975)to find a threshold. The largest 8-connected
component in the thresholded image is assumed to be the worm. Frames in
which the worm touches the image boundaries, is too small, lacks a clear
head and tail, or has unrealistic body proportions are not analyzed
further. Frames containing stage movement are also removed to eliminate
bad segmentations wherein the worm image may be blurred (see the section
titled "Absolute Coordinates"). Given our desire for accurate and
precise measures as well as the large data volume (due to a high video
frame rate), we err on the side of caution and attempt to reject
ambiguous segmentations rather than include them.

Once the worm has been thresholded, its contour is extracted by tracing
the worm's perimeter. The head and tail are located as sharp, convex
angles on either side of the contour. The skeleton is extracted by
tracing the midline of the contour from head to tail. During this
process, widths and angles are measured at each skeleton point to be
used later for feature computation. At each skeleton point, the width is
measured as the distance between opposing contour points that determine
the skeleton midline. Similarly, each skeleton point serves as a vertex
to a bend and is assigned the supplementary angle to this bend. The
supplementary angle can also be expressed as the difference in tangent
angles at the skeleton point. This angle provides an intuitive
measurement. Straight, unbent worms have an angle of 0°. Right angles
are 90°. And the largest angle theoretically possible, a worm bending
back on itself, would measure 180°. The angle is signed to provide the
bend's dorsal-ventral orientation. When the worm has its ventral side
concave within the bend, the bending angle is signed negatively.

Pixel count is a poor measure of skeleton and contour lengths. For this
reason, we use chain-code lengths (Freeman 1961). Each
laterally-connected pixel is counted as 1. Each diagonally-connected
pixel is counted as √2. The supplementary angle is determined, per
skeleton point, using edges 1/12 the skeleton's chain-code length, in
opposing directions, along the skeleton. When insufficient skeleton
points are present, the angle remains undefined (i.e., the first and
last 1/12 of the skeleton have no bending angle defined). 1/12 of the
skeleton has been shown to effectively measure worm bending in previous
trackers and likely reflects constraints of the bodywall muscles, their
innervation, and cuticular rigidity (Cronin et al. 2005).

[Freeman 1961] Herbert Freeman. 1961. On the encoding of arbitrary
geometric configurations. *IRE Transactions on Electronic Computers*,
10:260-268.
