Schafer Lab Feature Descriptions
================================

[Supplementary Figure 1: Feature computation](s1.md)

[Supplementary Figure 2: WT2 sensitivity](s2.md)

[Supplementary Figure 3: Wild type variability](s3.md)

[Supplementary Figure 4: Phenotypic summaries](s4.md)

[Supplementary Figure 5: The clustering is shown for all 305
strains](s5.md)

[Supplementary Figure 6: New locomotion phenotypes for three TRP
channels.](s6.md)

Details of feature files and algorithms for feature measurement
---------------------------------------------------------------

Four feature categories:

-   

    Morphology

    :   -   length
        -   width
        -   area
        -   area\_per\_length
        -   width\_per\_length

-   

    Locomotion

    :   -   bends
        -   bend count (not on below list?? <-@MichaelCurrie>)
        -   eccentricity
        -   amplitude
        -   wavelength
        -   track length
        -   coils
        -   eigen\_projection
        -   orientation (not on below list?? <-@MichaelCurrie>)
        -   kinks

-   

    Path

    :   -   range
        -   curvature
        -   dwelling (aka duration ? - @MichaelCurrie)

-   

    Posture

    :   -   velocity
        -   motion\_events
        -   foraging\_bends
        -   crawling\_bends
        -   omegas
        -   upsilons

Features can be one of three "types":

-   Simple (one scalar for the whole video)
-   Movement-based (a value for each frame of the video)
-   Event-based (an integer for the whole video)

22 fundamental features:

-   

    Simple:

    :   -   duration

-   

    Movement-based:

    :   -   length
        -   width
        -   area
        -   area\_per\_length
        -   width\_per\_length
        -   bends
        -   amplitude
        -   wavelength
        -   track\_length
        -   eccentricity
        -   kinks
        -   directions
        -   eigen\_projection
        -   velocity
        -   foraging\_bends
        -   crawling\_bends
        -   range
        -   curvature

-   

    Event-based:

    :   -   coils
        -   omegas
        -   upsilons
        -   motion\_events

These are calculated for specific body parts, and specific statistical
values are taken, like mean and standard deviation, etc. This expands
the list of 22 to 93. Then the 93 features are dumped into histograms,
with extra histograms calculated for movement features (forward,
backward, paused in addition to all), and extra histograms calculated
for the signed features (with some in both movement and event features):
(data that is negative, positive, and the absolute), giving a TOTAL
histogram count of 726. (Sometimes there might not be any data though,
like if a worm only moves forward, its histogram for all movement
features isolated to just backward data will contain no data.)

-   [Morphology](Morphology.md) - [MatLab
    Code](https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/%2Bseg_worm/%2Bfeatures/%40morphology) - [Python
    Code](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/open-worm-analysis-toolbox/features/WormFeatures.py)
-   [Posture](Posture.md) - [MatLab
    Code](https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/%2Bseg_worm/%2Bfeatures/%40posture) - [Python
    Code](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/open-worm-analysis-toolbox/features/posture_features.py)
-   [Locomotion](Locomotion.md) - [MatLab
    Code](https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/%2Bseg_worm/%2Bfeatures/%40locomotion) - [Python
    Code](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/open-worm-analysis-toolbox/features/locomotion_features.py)
-   [Path](Path.md) - [MatLab
    Code](https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/%2Bseg_worm/%2Bfeatures/%40path) - [Python
    Code](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/wormpy/WormFeatures.py)
-   [Event Finder Code (in
    Python)](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/open-worm-analysis-toolbox/features/events.py)

[Phenotypic Ontology](Phenotypic%20Ontology.md)
-----------------------------------------------

[Feature File Overview](Feature%20File%20Overview.md)
-----------------------------------------------------

Relevant Documents:
-------------------

[2013 - Yemini et al. in Nature Methods - A database of Caenorhabditis
elegans behavioral
phenotypes](http://www.nature.com/nmeth/journal/v10/n9/full/nmeth.2560.html)

[2013 - Yemini et al. in Nature Methods - Supplementary Data - Nature
Methods
nmeth.2560-S1](http://www.nature.com/nmeth/journal/v10/n9/extref/nmeth.2560-S1.pdf)

[2011 (November) - High-throughput, single-worm tracking and analysis in
Caenorhabditis elegans. Eviatar Yemini's PhD
dissertation.](http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/EvYemini.pdf)

Schafer Lab's *C. elegans* behavioural database back pages
----------------------------------------------------------

[Worm Segmentation](Worm%20Segmentation.md)

[Ventral Side Annotation and Head
Detection](Ventral%20Side%20Annotation%20and%20Head%20Detection.md)

[Absolute Coordinates](Absolute%20Coordinates.md)

[Feature Overview](Feature%20Overview.md)
