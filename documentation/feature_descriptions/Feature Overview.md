Feature Overview
================

From From Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. X. &
Schafer, W. R. A database of Caenorhabditis elegans behavioral
phenotypes. Nat. Methods (2013). `doi:10.1038/nmeth.2560`

All feature are computed from the worm's segmented contour and skeleton.
The skeleton and each side of the contour are scaled down to 49 points
for feature computation. Wild-type worms have four quadrants of
longitudinal, staggered bodywall muscles (Sulston & Horvitz 1977). Each
quadrant contains 24 such muscles with the exception of the ventral-left
quadrant, which has 23. With a sampling of 49 points, the skeleton and
contour sides have a well-defined midpoint. Moreover, since the worm is
confined to two dimensions, its bodywall muscles present roughly 24
degrees of freedom (although in practice it seems to be far less
(Stephens et al. 2008). With 49 points we have 2 samples per degree of
freedom and, therefore, expect to be sampling above the Nyquist rate for
worm posture.

A common notation is used to define the body parts. The head is
controlled by the first four bodywall muscles, per
quadrant - approximately 1/6 the length of the worm (White et al. 1986).
Similarly, the neck is controlled by the next four bodywall muscles, per
quadrant - approximately 1/6 the length of the worm. For this reason,
we define the head as the first 1/6 of the worm and the neck as the next
1/6 of the worm (skeleton points 1-8 and 9-16, respectively). For
symmetry, we define the tail and "hips", in a similar manner, on the
opposite end of the worm. The tail is the last 1/6 of the worm and the
hips are defined as the next 1/6 (skeleton points 42-49 and 34-41,
respectively). The midbody is defined as the remaining middle 1/3 of the
worm (skeleton points 17-33). For some features, the head and tail are
further subdivided to extract their tips, the first and last 1/12 of the
worm (skeleton points 1-4 and 46-49, respectively).

Frame-by-frame features are represented by top-level histograms and
statistics as well as subdivisions exploring their values during
forward, backward, and paused states. This is to measure behaviors that
depend on the state of notion such as foraging amplitude, which is
reduced during reversals in wild-type worms (Alkema et al. 2005). Many
features are signed to reflect dorsal-ventral orientation,
forward-backward trajectory, and other special cases (e.g., eigenworm
projection) to capture any asymmetry. Finally, event-style
features(coiling, turning, and motion states) are summarized using
global and local measures. Global measures include the event frequency,
the ratio of time spent within the event to the total experiment time,
and a similar measure for the ratio of the distance covered within the
event to the total distance traveled by the worm (when available). Local
measures include the time spent in every individual event, the distance
covered in each event (when available), and both the time and distance
covered between each pair of successive events.
