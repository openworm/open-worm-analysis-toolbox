Schafer Lab
===========

[The Schafer Lab](http://www2.mrc-lmb.cam.ac.uk/groups/wschafer/), at
the [MRC Laboratory of Molecular
Biology](http://www2.mrc-lmb.cam.ac.uk/), Cambridge, UK, developed the
[Worm Analysis
Toolbox](http://www.mrc-lmb.cam.ac.uk/wormtracker/index.php?action=analysis)
to work with data from the [Worm Tracker 2.0
(WT2)](http://www.mrc-lmb.cam.ac.uk/wormtracker/). The
open-worm-analysis-toolbox package is a port of the last version of the Worm
Analysis Toolbox, version 1.3.4.

[Dr. Eviatar "Ev"
Yemini](https://sites.google.com/site/openarchitecture1/3-contributors-and-syntax/ev-yemini),
for all the original Matlab code built as part of [the WormBehavior
database](http://wormbehavior.mrc-lmb.cam.ac.uk/), written as part of
his doctorate at the Schafer Lab. Without him this project would not
have been possible. His PhD thesis was partly based on his work creating
the analysis package we ported to form the basis of the package.

[Dr. Tadas Jucikas](https://www.linkedin.com/in/tjucikas), post-doctoral
scholar at the Schafer Lab, for co-authoring the Worm Tracker 2.0
analysis package with Ev.

[Dr. William "Bill"
Schafer](http://www2.mrc-lmb.cam.ac.uk/group-leaders/n-to-s/william-schafer/),
Schafer Lab eponym, for allowing us to use and adapt Ev's code.

[Dr. André
Brown](http://csc.mrc.ac.uk/research-group/behavioural-genomics/),
formerly a post-doctoral scholar at the Schafer Lab, and now principal investigator at the Brown Lab at Imperial College London, for contributions to the
original Matlab code and for his work on behavioural motifs, which we hope to add.  With Dr. Kerr he created the tracker-commons format.

[Dr. Rex Kerr](https://www.janelia.org/kerr-lab), principal investigator at the Kerr Lab at Janelia Research Campus, for his work on the tracker-commons format.

[Barry
Bentley](http://www.neuroscience.cam.ac.uk/directory/profile.php?bb421),
PhD student at the Schafer Lab, for his ongoing support of our use of
their code.

[Laura
Grundy](http://www2.mrc-lmb.cam.ac.uk/group-leaders/n-to-s/william-schafer/),
lab assistant at the Schafer Lab, for filming many hundreds of hours of
C. elegans videos.

[Schafer Lab
Acknowledgements:](https://github.com/openworm/SegWorm/blob/master/Worms/Printing/methodsTIF.m#L1514)

-   WT2 employs
    [Java](http://en.wikipedia.org/wiki/Java_(programming_language))
    (version 1.6) for the tracking software and
    [Matlab](http://www.mathworks.com/products/matlab/) (version 2010a)
    for the analysis software. Standard Matlab functions are used along
    with the Image Processing, Statistics, and Bioinformatics toolboxes.
-   Special thanks and acknowledgements are due to [Christopher J.
    Cronin](http://wormlab.caltech.edu/members/pictures/IMG_0084.jpg)
    and [Paul W.
    Sternberg](http://wormlab.caltech.edu/members/paul.html) at
    Caltech's [Sternberg Lab](http://wormlab.caltech.edu/) for supplying
    the Matlab code from their publication, ["An automated
    system for measuring parameters of nematode sinusoidal movement"
    (Cronin et al.
    2005)](http://www.ncbi.nlm.nih.gov/pubmed/15698479), for use
    in the [locomotion
    features](https://github.com/openworm/open-worm-analysis-toolbox/blob/master/documentation/Yemini%20Supplemental%20Data/Locomotion.md).
-   [Joan Lasenby](http://www-sigproc.eng.cam.ac.uk/Main/JL) and [Nick
    Kingsbury](http://www-sigproc.eng.cam.ac.uk/Main/NGK) were an
    invaluable resource for computer vision questions.
-   [Andrew
    Deonarine](http://www.immunology.cam.ac.uk/directory/adeonari@mrc-lmb.cam.ac.uk),
    [Richard Samworth](http://www.statslab.cam.ac.uk/~rjs57/),
    [Sreenivas
    Chavali](http://www.wolfson.cam.ac.uk/people/dr-sreenivas-chavali),
    [Guilhem Chalancon](http://www.mrc-lmb.cam.ac.uk/genomes/guilhem/),
    [Sarah
    Teichmann](http://www.ebi.ac.uk/about/people/sarah-teichmann), and
    [Dr. M. Madan Babu](http://mbgroup.mrc-lmb.cam.ac.uk/about-m-madan/)
    provided a wealth of help and information for the bioinformatic
    clustering and statistics.
-   Thanks to the [OpenCV](http://opencv.org/) computer vision library.
-   Thanks to [Gerald Dalley](http://people.csail.mit.edu/dalleyg/) for
    the [videoIO toolbox for
    Matlab](http://sourceforge.net/projects/videoio/).

[Matlab Central File Exchange](http://www.mathworks.com/matlabcentral/fileexchange/)
====================================================================================

Thanks and acknowledgements are due for the following freely available
code:

-   [export\_fig](https://github.com/ojwoodford/export_fig) function by
    [Oliver Woodford](https://github.com/ojwoodford)
-   [notBoxPlot](http://www.mathworks.com/matlabcentral/fileexchange/26508-notboxplot-alternative-to-box-plots)
    function by [Rob
    Campbell](http://www.mathworks.ca/matlabcentral/fileexchange/authors/49773)
-   [swtest](http://www.mathworks.com/matlabcentral/fileexchange/13964-shapiro-wilk-and-shapiro-francia-normality-tests)
    function by [Ahmed Ben
    Saïda](http://www.mathworks.com/matlabcentral/fileexchange/authors/27181)
-   [fexact](http://www.mathworks.com/matlabcentral/fileexchange/22550-fisher-s-exact-test)
    function by [Michael
    Boedigheimer](https://www.linkedin.com/profile/view?id=155041881)
-   [rdir](http://www.mathworks.com/matlabcentral/fileexchange/19550-recursive-directory-listing)
    function by [Gus
    Brown](http://www.mathworks.gr/matlabcentral/fileexchange/authors/30177)
-   [xlswrite1](http://www.mathworks.com/matlabcentral/fileexchange/10465-xlswrite1)
    function by [Matt
    Swartz](http://www.mathworks.com/matlabcentral/fileexchange/authors/22868)

OpenWorm
========

[Dr. Stephen Larson](https://github.com/slarson), OpenWorm's project
coordinator, for the original idea to adapt Ev's code and for
encouragement and support.

[The Movement Analysis Team](https://github.com/orgs/openworm/teams/movement-analysis):

-   [Dr. James "Jim" Hokanson](https://github.com/JimHokanson) for
    improving upon the original Matlab code, doing much of the
    translation of the code into this Python repository, overall
    architectural decisions, documentation, and for his [Matlab Standard
    Library](https://github.com/JimHokanson/matlab_standard_library).
-   [Chee Wai Lee](https://github.com/cheelee) for his work on this repository and tracker-commons.
-   [Michael Currie](https://github.com/MichaelCurrie) for helping with
    the translation into Python, and helping with repository
    documentation.
-   [Aidan Rocke](https://github.com/AidanRocke) for his work translation Dr. Brown's behavioural syntax code into Python.
   
Past team members:
-   [Chris Linzy]
-   [Chris Barnes]
-   [Joe Bowen]
