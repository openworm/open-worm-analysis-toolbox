Absolute Coordinates
====================

From Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. X. & Schafer,
W. R. A database of Caenorhabditis elegans behavioral phenotypes. Nat.
Methods (2013). `doi:10.1038/nmeth.2560`

Many features require plate (or absolute) coordinates rather than pixel
coordinates defined with respect to the camera field of view. Prior to
recording, all trackers are regularly calibrated to determine the
conversion from pixels to absolute coordinates. When recording is
complete, stage movements are matched to their video signature in order
to convert segmented worms to absolute coordinates (offset by the
stage's location).

During recording, every stage movement is logged. When recording has
completed, the video is scanned to locate motion frames. Because
re-centering the worm causes an abrupt change in both the image
background and the worm's location. these changes are simply measured as
the pixel variance in the difference between subsequent frames. The Otsu
method is used to find an appropriate threshold for delineating
stage-movement frames. The number of stage movements and the intervals
between them are matched against the log of software-issued stage
movement commands. If the match fails (an infrequent event usually
caused by worms reaching the boundary of their plate or external factors
damaging the recording), the worm and its video are discarded and not
used. In our data set, roughly 48 of the videos were discarded due to
stage-movement failures.

With the stage movements matched to their video signature, the Otsu
threshold is employed once again to compute a local threshold that
delineates a more accurate start and end for each individual stage
movement. The same algorithm is also employed for the interval at the
start of the video until the first stage movement and, similarly, from
the last stage movement until the end of the video. With this in place,
stage movement frames are discarded and each interval between stage
movements is assigned a stage location. Thereafter, each segmented worm
is converted to its absolute coordinates on the plate.
