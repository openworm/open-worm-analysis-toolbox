Posture Features
================

1. Bends
--------

Worm bending is measured using the supplementary angles to the bends
formed along the skeleton, with each skeleton point serving as the
vertex to its respective bend:

![](fig%204%20b%20-%20bend%20angle.gif)

The supplementary angle can also be expressed as the difference in
tangent angles at the skeleton point. The supplementary angle provides
an intuitive measurement. Straight, unbent worms have an angle of 0°.
Right angles are 90°. And the largest angle theoretically possible, a
worm bending back on itself, would measure 180°. The supplementary angle
is determined, per skeleton point, using edges 1/12 the skeleton’s
chain-code length, in opposing directions, along the skeleton. When
insufficient skeleton points are present, the angle remains undefined
(i.e., the first and last 1/12 of the skeleton have no bending angle
defined). The mean and standard deviation are measured for each body
segment. The angle is signed to provide the bend’s dorsal-ventral
orientation. When the worm has its ventral side internal to the bend,
the bending angle is signed negatively.

Credit for this definition: [Yemini *et al.*
(2013)](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

2. Bend Count
-------------

The bend count is a rough measure of the number of bends along the worm.
The supplementary skeleton angles are measured during segmentation and
signed to reflect their dorsal-ventral orientation. These angles are
convolved with a Gaussian filter, 1/12 the length of the skeleton, with
a width defined by the Matlab “gausswin” function’s default a of 2.5 and
normalized such that the filter integrates to 1, to smooth out any
high-frequency changes. The angles are then sequentially checked from
head to tail. Every time the angle changes sign or hits 0°, the end of a
bend has been found and the count is incremented. Bends found at the
start and end of the worm must reflect a segment at least 1/12 the
skeleton length in order to be counted. This ignores small bends at the
tip of the head and tail.

Credit for this definition: [Yemini *et al.*
(2013)](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

3. Eccentricity
---------------

The eccentricity of the worm’s posture is measured using the
eccentricity of an equivalent ellipse to the worm’s filled contour. The
orientation of the major axis for the equivalent ellipse is used in
computing the amplitude, wavelength, and track length (\#4, 5, and 6,
described below).

Credit for this definition: [Yemini *et al.*
(2013)](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

4. Amplitude
------------

Worm amplitude is expressed in two forms: a) the maximum amplitude found
along the worm body and, b) the ratio of the maximum amplitudes found on
opposing sides of the worm body (wherein the smaller of these two
amplitudes is used as the numerator). The formula and code originate
from the publication ["An automated system for measuring parameters of
nematode sinusoidal movement" by Cronin *et al.*
(2005)](http://www.biomedcentral.com/1471-2156/6/5).

The worm skeleton is rotated to the horizontal axis using the
orientation of the equivalent ellipse and the skeleton’s centroid is
positioned at the origin. The maximum amplitude is defined as the
maximum y coordinate minus the minimum y coordinate. The amplitude ratio
is defined as the maximum positive y coordinate divided by the absolute
value of the minimum negative y coordinate. If the amplitude ratio is
greater than 1, we use its reciprocal.

5. Wavelength
-------------

The worm’s primary and secondary wavelength are computed by treating the
worm’s skeleton as a periodic signal. The formula and code originate
from the publication ["An automated system for measuring parameters of
nematode sinusoidal movement" by Cronin *et al.*
(2005)](http://www.biomedcentral.com/1471-2156/6/5).

The worm’s skeleton is rotated as described above for the amplitude. If
there are any overlapping skeleton points (the skeleton’s x coordinates
are not monotonically increasing or decreasing in sequence --e.g., the
worm is in an S shape) then the shape is rejected, otherwise the Fourier
transform computed. The primary wavelength is the wavelength associated
with the largest peak in the transformed data. The secondary wavelength
is computed as the wavelength associated with the second largest
amplitude (as long as it exceeds half the amplitude of the primary
wavelength). The wavelength is capped at twice the value of the worm’s
length. In other words, a worm can never achieve a wavelength more than
double its size.

6. Track Length
---------------

The worm’s track length is the range of the skeleton’s horizontal
projection (as opposed to the skeleton’s arc length) after rotating the
worm to align it with the horizontal axis. The formula and code
originate from the publication ["An automated system for measuring
parameters of nematode sinusoidal movement" by Cronin *et al.*
(2005)](http://www.biomedcentral.com/1471-2156/6/5).

7. Coils
--------

Worm coiling (touching) events are found by scanning the video frame
annotations. During segmentation, every frame that cannot be segmented
is annotated with a cause for failure. Two of these annotations reflect
coiling events. First, if we find fewer than two sharp ends on the
contour (reflecting the head and tail) then the head and/or tail are
obscured in a coiling event. Second, if the length between the head and
tail on one side of the contour is more than double that of the other
side, the worm has either assumed an omega bend or is crossed like a
wreath. Empirically, less than 1/5 of a second is a very fast touch and
not usually reflective of coiling. Therefore, when a period of
unsegmented video frames exceeds 1/5 of a second, and either of the
coiling annotations are found, we label the event coiling.

Credit for this definition: [Yemini *et al.*
(2013)](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

8. Eigen Projections
--------------------

The eigenworm amplitudes are a measure of worm posture. They are the
projections onto the first six eigenworms which together account for 97%
of the variance in posture. The eigenworms were computed from 15 N2
videos (roughly three hours of video, 1/3 of a million frames) as
previously described in [Stephens *et. al*
(2008)](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1000028).

Briefly, 48 tangent angles are calculated along the skeleton and rotated
to have a mean angle of zero. [Principal component
analysis](http://en.wikipedia.org/wiki/Principal_component_analysis) is
performed on the pooled angle data and we keep the six principal
components (or eigenworms) that capture the most variance. The first
eigenworm roughly corresponds to body curvature. The next two eigenworms
are akin to sine and cosine waves encoding the travelling wave during
crawling. The fourth eigenworm captures most of the remaining variance
at the head and tail. Projected amplitudes are calculated from the
posture in each frame. Even for the mutants, the data is always
projected onto the N2-derived eigenworms.

(Also discussed at [issue
79](https://github.com/openworm/open-worm-analysis-toolbox/issues/79))

![](eigenworms.png)

9. Orientation
--------------

The worm’s orientation is measured overall (from tail to head) as well
as for the head and tail individually. The overall orientation is
measured as the angular direction from the tail to the head centroid.
The head and tail centroids are computed as the mean of their respective
skeleton points.

The head and tail direction are computed by splitting these regions in
two, then computing the centroid of each half. The head direction is
measured as the angular direction from the its second half (the centroid
of points 5-8) to its first half (the centroid of points 1-4). The tail
direction is measured as the angular direction from the its second half
(the centroid of points 42-45) to its first half (the centroid of points
46-49).

Credit for this definition: [Yemini *et al.*
(2013)](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

10. Kinks
---------

Apparently this feature is undocumented in the original scientific
paper. [Yemini *et al.*
(2013)](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

Nevertheless, we do have code to calculate it, and it does appear as a
data item in the feature files available in the *C. elegans* behavioural
database.
