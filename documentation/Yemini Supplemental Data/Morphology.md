Morphology Features
===================

Credit for all Locomotion feature definitions: [Yemini et al.
(2013)](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

1. Length
---------

Worm length is computed from the segmented skeleton by converting the
[chain-code](http://en.wikipedia.org/wiki/Chain_code) pixel length to
microns.

2. Widths
---------

Worm width is computed from the segmented skeleton. The head, midbody,
and tail widths are measured as the mean of the widths associated with
the skeleton points covering their respective sections. These widths are
converted to microns.

3. Area
-------

The worm area is computed from the number of pixels within the segmented
contour. The sum of the pixels is converted to microns squared (i.e.
microns2).

4. Area/Length
--------------

No description available from Yemini paper.

5. Midbody Width/Length
-----------------------

No description available from Yemini paper.
