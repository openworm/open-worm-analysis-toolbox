Feature File Overview
=====================

From From Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. X. &
Schafer, W. R. A database of Caenorhabditis elegans behavioral
phenotypes. Nat. Methods (2013). `doi:10.1038/nmeth.2560`

The features are presented within four types of files available online
at:

<http://wormbehavior.mrc-lmb.cam.ac.uk/>

PDF files provide a visual summary of the data, per strain. CSV files
provide a spreadsheet of the data, per strain. And, three types of MAT
files are provided to access the strain data and statistics as well as
the skeleton, contour, and feature data for each individual experiment,
per frame.

The MAT files, per worm, are available for every experiment. To ensure
high-quality experimental data, strain collections of experiments and
controls were filtered and only include worm videos of at least 20fps,
14-15 minutes long, wherein at least 20% of the frames were segmented.
We only include data collected Monday through Saturday, from 8am to 6pm.
This resulted in a mean of 24 worms per strain with a minimum of 12 and
a standard deviation of 14. Controls were chosen from the filtered N2
data collection by matching the strain collections to controls performed
within the same week. This resulted in a mean of 63 controls, per strain
collection, with a minimum of 18 and a standard deviation of 29. We
examined 100 videos (roughly 2 million frames) from our filtered
collection and found that the head was correctly labeled with a mean and
standard deviation of 95.17 ± 17.5% across individual videos and 95.69%
of the frames collectively.

Outliers can compress visual details in their corresponding histograms.
For this reason, the strain collections underwent one more filtering
step prior to inclusion in the PDF files. Experiments were discarded
wherein any of the worm data exceeded reasonable bounds of 250 to 2000
microns for length, 25 to 250 microns for width, and/or -1000 to 1000
microns/seconds for the midbody speed. Outliers were seldom found.
Overall, 49 non-control worms were lost from a collection of 7,529
experiments. No strain collection lost more than 2 worms. The N2
collection of controls lost 5 worms from its total of 1,218 experiments.
The CSV files and MAT statistical-significance files are available for
both the primary quality-filtered data sets and the secondary,
outlier-filtered data sets.

Shapiro-Wilk testing (performed using the “swtest” function by Ahmed Ben
Saïda) of each feature measure (with corrections for multiple
comparisons) showed a maximum q-value of 0.0095 over our collective N2
data set, indicating that, in aggregate, none of the measures are
normally distributed. Further testing across all strain collections
(which have far lower sampling than the N2 collective) and their
controls, indicated a roughly 2:1 ratio of normal to non-normal
distributions, rejecting the null hypothesis of normality at a q-value
of 0.05. Therefore, we chose to test strain measurements against their
controls by using the non-parametric Wilcoxon rank-sum test (with the
null hypothesis that both sets of mean values were drawn from the same
distribution). In four strains, at least one measure was detected
exclusively in either the strain or its control, meaning the measurement
was always observed within one set and never in the other (e.g., some
strains never perform reversals). When this occurred, we used a Fisher’s
exact test to measure the probability that our sets were drawn from the
same distribution of observed and unobserved events. Occasionally,
features measurements had insufficient values for testing due to low
sampling (e.g., omega-turn events), these measures were ignored and
their p-value marked as undefined. In total, our 702 measurements were
obtained for each of 305 strains in addition to collections of our N2
worms by hour (9am-4pm, with 8am and 5pm discarded due to very low
sampling), weekday (Tuesday-Friday, with Monday and Saturday discarded
due to very low sampling), and month (January-December). We used
False-Discovery Rate (FDR) to correct for nearly 702 measures by 329
groups and transform the p-values to their q-value equivalents. This
method was described in ["A direct approach to false discovery rates" by
Storey (June 2001)](http://www.genomine.org/papers/directfdr.pdf).

Our unfiltered histograms, presented within individual MAT files, were
constructed by choosing standard bin resolutions (widths and centers)
that resulted in roughly 103 bins, per feature, for our N2 data. When
plotting histograms, we use a common formula to downsample the bins. We
measure the square root of the total number of data samples contributing
to the collective histogram. If this value is less than the number of
bins available, the histogram is downsampled to reduce the number of
bins to the nearest integer at or below the computed square root. When
multiple histograms are plotted together, the smallest common bin size
is used to downsample all the histograms to the same bin width and
centers.

PDF Files
---------

The PDF (portable document format) files include five sections:

1.  a table of contents and overview of the results,
2.  a short summary of the most important features,
3.  the details for every feature,
4.  traces of the worm paths, and
5.  a reference with the experimental methods.

Each page uses a color scheme to provide quick visual summaries of its
results. All pages display tabs, on the right side, that explain their
color scheme. The initial summary page of histograms (page 2) displays
an example histogram that acts as a guide to understanding histogram
plots and the statistics displayed in their titles. The page formats are
as follows:

1.  The table of contents details the layout of the PDF file. All
    feature measures are shown alongside their minimum q-value and a
    page number for details. The table of contents page also shows an
    overview with the experiment annotation and its phenotypic ontology
    (see the section titled “Phenotypic Ontology”).
2.  There are three summary pages. These pages show important feature
    histograms, with the collective experiments in color and their
    controls in gray. The background color, for the histogram plots,
    indicates the minimum q-value significance for the plotted feature.
    The title of each plot provides several statistical measures for the
    experiment and control collections. An example histogram, at the
    beginning of the first summary page, provides a reference to
    interpret the aforementioned statistical measures. Significant
    measures, with q = 0.05, are marked in bold font within the plot
    title.

    The crawling frequency, worm velocity, foraging speed, all event
    features, path range, and dwelling are shown on a pseudo log-value
    scale to improve readability within their small summary histograms.
    This pseudo log-value scale is achieved by taking the magnitude of
    the data values (to avoid complex numbers resulting from the
    logarithms of any negative numbers), translating the magnitude by 1
    (to avoid the logarithms of any values less than 1, which would
    invert the sign of the data), taking the logarithm, then re-signing
    the formerly negative data values.

3.  The detail pages present a detailed view of the histograms for every
    feature. They follow a similar format to the summary pages except
    that they never use a log scale for feature values. The title of
    each plot provides a large set of statistical measures. The control
    values are shown between square brackets. The statistical values
    include: a) the number of worms providing measurements (“WORMS”); b)
    the number of measurements sampled for the collection of worms
    (“SAMPLES”); c) the mean of the data (“ALL”) alongside the SEM and,
    when the data is signed, the means for the absolute data values
    (ABS), positive data values only (“POS”), and negative data values
    only (“NEG”) alongside their SEMs as well; d) the p-value results
    using Wilcoxon rank-sum testing and q-value results using False
    Discovery Rate correction (for multiple tests across 329 strain
    collections by 702 feature measurements), both labeled accordingly
    (respectively “p” and “q”);
    e\) event features also display their mean frequency (“FREQ”), the mean
    percentage of time spent in the event relative to the total experiment
    time (“TIME”), and, when available, the mean percentage of distance
    traveled during the event relative to the total distance covered during
    the experiment (“DIST”).

    Features that have motion-state subdivisions are shown with an
    additional view wherein all motion-state histograms, and their
    integral histogram, are shown on the same plot. This allows one to
    quickly distinguish behaviors dependent on the motion state. Event
    features have an additional view wherein event and inter-event
    measures are plotted on a log-probability scale to make outlying
    events more visible.

4.  The path trace pages display the paths for the worms’ head, midbody,
    and tail and heatmaps for the midbody speed and foraging amplitude.
    Pages with the head, midbody, and tail include a tab, on the right
    side, to interpret the color associated with each body part. Pages
    with heatmaps include a tab, on the right side, to interpret the
    color gradient. On the path trace plots, the start and end of each
    path is denoted by a gray and black worm, respectively. Moreover, on
    each plot, the locations for coiling events are marked by a “+” and
    those for omega turns are marked by an “x”. Body part plots use
    transparency to roughly indicate dwelling through color opacity.

    The first page of each path trace shows a collection of up to 24
    worms (when available) overlayed for both the experiment and control
    collections, at the same scale. These overlays provide a quick view
    of features such as relative path sizes, food leaving behaviors, and
    the relative locations for coiling events and omega turns. When more
    than 24 worms are available we sort the worms by date, then choose
    24 from the first to the last experiment at regular intervals. The
    paths are rotated to align their longest axis vertically, and then
    centered using the minimum and maximum x and y path values, per
    worm.

    The next page of the path traces shows each collection of 24 paths
    on the same plot, ordered roughly from largest to smallest, spaced
    out to avoid any overlay. The experiments and their controls use
    independent scales. This ordered plot provides a quick view to
    distinguish salient characteristics of experiment versus control
    paths (e.g., bordering at the edge of the food lawn).

    The subsequent pages for each path trace show the 24 individual worm
    paths, for the experiments and their controls, without rotation,
    sorted by date.

5.  The method pages provide a reference for the details of our
    methodology.

CSV Files
---------

The CSV (comma separated value) files are compatible with popular
spreadsheet programs (e.g., Microsoft Excel, Apple iWork Numbers,
OpenOffice, etc.). Each experimental collection is accompanied by four
CSV files presenting the data and statistics for all morphology
(.morphology.csv), posture (.posture.csv), motion (.motion.csv), and
path features (.path.csv). The CSV files present the strain, genotype,
and date for the experimental strain and control worms. The mean and
standard deviation are presented for each feature measure, per worm and
for the collection of experiments and controls. The p and q-values are
presented for the strain as a whole (the null hypothesis is that
experiment and control worms are drawn from the same distribution) and
for each feature measure individually. These p and q values are shown
for both the non-parametric Wilcoxon rank-sum test and the
normal-distribution Student’s t-test (unpaired samples with unequal
variance). The Shapiro-Wilk test for normality (with associated p and q
values) is also shown for each measure. Correction for multiple testing
(the q-values) was performed over our entire set of 329 groups of strain
collections by 702 measures. For the Shapiro-Wilk normality test,
correction for multiple comparisons included an additional 329
group-specific controls by 702 measures.

MAT Files
---------

### *3 types of MAT File:*

1.  Each experiment is represented in a MAT, HDF5-formatted file
    (Hierarchical Data Format Version 5 – an open, portable, file format
    with significant software support). HDF5 files are supported by most
    popular programming languages including Matlab, Octave (a free
    alternative to Matlab), R, Java, C/C++, Python, and many other
    environments. These experiment files contain the time-series feature
    data for an individual worm.
2.  Additionally, each strain collection of experiments and their
    collection of controls are also represented in a single HDF5, MAT
    file. These strain files contain histogram representations and
    summary statistics (but not significance) for the collective
    experiments.
3.  Finally, the statistical significance, for our entire collection of
    mutants, is presented in a single HDF5, MAT file.

The first two MAT file types, individual experiments and strain
collections, share a similar format. The individual experiment files
present the feature data as a time series. They also include the full
skeleton and the centroid of the contour, per frame, permitting novel
feature computations. The strain collections present the data in summary
and in histograms. The format for both file types is two top-level
structs, “info” (“wormInfo” for the strain collections) and “worm”,
which contain the experimental annotation and data, respectively.

#### The “info” struct

The “info” struct contains the experimental annotation. For the strain
collections, the “info” from each experiment is collected into an array
of structs called “wormInfo”. Both variables share the same format with
the following subfields:

1.  **wt2**. The Worm Tracker 2.0 version information.
2.  **video**. The video information. The video “length” is presented as
    both “frames” and “time”. The video “resolution” is in “fps”
    (frames/seconds), pixel “height” and “width”, the ratio of
    “micronsPerPixel”, and the codec’s “fourcc” identifier. The video
    frame “annotations” are presented for all “frames” with a
    “reference” specifying the annotation’s numerical “id”, the
    “function” it originated from, and a “message” describing the
    meaning of the annotation.
3.  **experiment**. The experiment information. The “worm” information
    is presented for its “genotype”, “gene”, “allele”, “strain”,
    “chromosome”, “sex”, “age”, the “habituation” time prior to
    recording, the location of its “ventralSide” in the video (clockwise
    or anti-clockwise from the head), the “agarSide” of its body (the
    body side touching the agar), and any other worm “annotations”. The
    “environment” information is presented for the experiment conditions
    including the “timestamp” when the experiment was performed, the
    “arena” used to contain the worm (always a low-peptone NGM plate for
    the data presented here), the “food” used (e.g., OP50 E. coli), the
    “temperature”, the peak wavelength of the “illumination”, any
    “chemicals” used, the “tracker” on which the experiment was
    performed (a numerical ID from 1 to 8), and any other environmental
    “annotations”.
4.  **files**. The name and location for the analyzed files. Each
    experiment is represented in a “video” file, “vignette” file (a
    correction for video vignetting), “info” file (with tracking
    information, e.g., the microns/pixels), a file with the log of
    “stage” movements, and the “computer” and “directory” where these
    files can be found.
5.  **lab**. The lab information where the experiment was performed. The
    lab is represented by its “name”, the “address” of the lab, the
    “experimenter” who performed the experiment, and any other
    lab-related “annotations”.

### The “worm” struct

The “worm” struct contains experimental data. The individual experiments
contain the full time series of data along with the worm’s skeleton and
the centroid of its contour, per frame. The strain collections contain
summary data and histograms in place of the time-series data. Both files
share a similar initial format with the following subfields:

1.  **morphology**. The morphology features. The morphology is
    represented by the worm’s “length”, its “width” at various body
    locations, the “area” within its contour, the “widthPerLength”, and
    the “areaPerLength”.
2.  **posture**. The posture features. The worm’s posture is represented
    by its bend count in “kinks”, measures of the “bends” at various
    body locations (computed as both a “mean” and standard deviation,
    “stdDev”), its “max” “amplitude” and its “ratio” on either side, its
    “primary” and “secondary” “wavelength”, its “trackLength”, its
    “eccentricity”, its “coils”, the orientation “directions” of various
    body parts, and its six “eigenProjections”. Individual experiment
    files also contain the “skeleton” “x” and “y” coordinates, per
    frame.
3.  **locomotion**. The motion features. Worm motion states are
    represented by “forward”, “backward”, and “paused” events, the
    “speed” and angular “direction” of the “velocity” for various body
    parts, the “amplitude” and “frequency” of the crawling “bends” for
    various body parts, as well as the “foraging” “bends” which are
    measured in an “amplitude” and “angleSpeed”, and the “turns”
    associated with “omega” and “upsilon” events. Individual experiment
    files also contain a “motion” state “mode” with values
    distinguishing forward (1), backward (-1), and paused (0) states,
    per frame.
4.  **path**. The path features. The path is represented by its “range”,
    “curvature”, and the dwelling “duration” for various body parts.
    Individual experiment files also contain the “x” and “y”
    “coordinates” of the contour’s centroid. Moreover, the individual
    experiment files present the “duration” as an “arena” with a
    “height”, “width”, and the “min” and “max” values for the “x” and
    “y” axes of the arena. The arena can be transformed to a matrix
    using the given height and width. The duration of the worm and body
    parts are represented as an array of “times” spent at the “indices”
    of the arena matrix.

All events are represented by their “frequency” and either their
“timeRatio” (the ratio of time in the event type to the total experiment
time) or, if the worm can travel during the event, the “ratio.time”
(equivalent to “timeRatio”) and “ratio.distance” (the ratio of the
distance covered in the event type to the total distance traveled during
the experiment). The individual experiment files represent each event as
“frames” with a “start” frame, “end” frame, the “time” spent in this
event instance, the “distance” traveled during this event instance (when
available), the “interTime” till the next event, and the “interDistance”
traveled till the next event. The strain collection files summarize
these fields, excluding the individual “frames” and their “start” and
“end”. The strain collection files present the data for each feature
within a “histogram” (as opposed to the individual experiment files
which simply use a time-series array of values). Furthermore, when a
feature can be subdivided by motion state, sub histograms are included
for the “forward”, “backward”, and “paused” states. All histograms
contain the “PDF” (probability distribution function) for each of their
“bins” (centered at the associated feature’s values). All histograms
also contain the “resolution” (width) of their bins, whether or not
there “isZeroBin” (would one of the bins be centered at 0?), and whether
or not the feature “isSigned” (can the feature values be negative?).

Finally, the strain collection files present their data in three types
of fields: a) individually as the “data” per experiment, b) summarized
over the “sets” of experiments and, c) aggregated in “allData” as if we
ran one giant experiment instead of our sets. In other words, “sets”
weights each experiment identically whereas “allData” weights every
frame, across all experiments, identically. The data is always
represented as both a “mean” and “stdDev” (standard deviation). The mean
and standard deviation are always computed for “all” the data. When the
data is signed, the mean and standard deviation are also computed for
the data’s “abs” (absolute value), “pos” (only the positive values), and
“neg” (only the negative values). The format for the three types of data
is as follows:

1.  **data**. The individual data for every experiment is presented in
    arrays (in the same order as the “wormInfo” experiment annotations).
    The array data presents each experiment’s individual “mean”,
    “stdDev”, the number of “samples” measured, and the experiment’s
    data “counts” for each one of the histogram’s “bins”.
2.  **sets**. The data for the set of experiments is presented as the
    “mean”, “stdDev”, and “samples” (the number of experiments) of the
    collected set.
3.  **allData**. The aggregate of all data measurements, as if the
    collection of videos were instead one long, giant video, is
    presented as a “mean”, “stdDev”, the total “samples” (the total
    number of frames wherein the data was measured), and the aggregate
    of “counts” for each one of the histogram’s bins.

Statistical Significance MAT File
---------------------------------

The statistical significance for all strains is collected into a single
MAT file. This file contains three top-level structs with information
for both the “worm” and “control” collections as well as the “dataInfo”
necessary to interpret the included matrices of data. The matrices are
organized as rows of strains and columns of feature measures. The “worm”
struct has the following subfields: 1. **info**. The worm information
for each strain collection presented as their “strain”, “genotype”,
“gene”, and “allele”.

2.  **stats**. The statistics for each strain collection presented, for
    every feature measure, as their “mean”, “stdDev” (standard
    deviation), “samples” (the number of worms providing a measurement
    for the feature – e.g., not all worms execute omega turns), and
    “zScore” relative to the control (a simple normalization to the
    control ­-note that the collection of N2 controls has no zScore).
    Measurements exclusively found in the experimental group have a
    zScore of infinity and those found exclusively found in the control
    are -infinity. Furthermore, we include Shapiro-Wilk tests of data
    normality, per measure, in “pNormal” and correction for multiple
    testing, using their False-Discovery rate q-value replacements, in
    “qNormal”. The q-values are computed across all measures per
    “strain” and their associated controls (roughly 1404 tests) and
    across “all” strain and control measures collectively (roughly 329
    by 1404 tests).
3.  **sig**. The statistical significance for each strain collection is
    presented, for every feature measure, as their “pTValue” (Student’s
    t-test p-value, unpaired samples with unequal variance) and
    “pWValue” (Wilcoxon rank-sum test p-value). The “qTValue” and
    “qWValue” represent the False-Discovery rate q-value replacements
    for the “pTValue” and “pWValue” respectively. The q-values are
    computed across all measures per “strain” (roughly 702 tests) and
    across “all” strains and measures collectively (roughly 329 by 702
    tests). The collection of N2s has no associated significance.

### The “control” struct

The “control” struct contains the control “stats” in an identical format
to the “worm” struct “stats”, but without the “zScores”.

### The “dataInfo” struct

The “dataInfo” struct provides information for each column of the
feature measure matrices used in the “worm” and “control” structs. Each
feature measure has a “name”, a “unit” of measurement, titles for three
possible subdivisions (“title1”, “title2”, and “title3” – the title of
the feature itself, its motion state, and its signed subdivision),
helpful indexed offsets for these titles (“title1I”, “title2I”, and
“title3I”), an associated struct “field” to locate the feature in our
other MAT files, the corresponding “index” for the struct field (e.g.,
the six eigenworm projections are represented in a field, as a 6-element
array), “isMain” (is this the main feature as opposed to a subdivision
of a main feature?), the feature “category” (morphology “m”, posture
“s”, motion “l”, path “p”), the feature “type” (simple data “s”, motion
data “m”, event summary data “d”, event data “e”, inter-event data “i”),
the feature “subtype” (none “n”, forward motion state “f”, backward
motion state “b”, paused state “p”, event-time data “t”, eventdistance
data “d”, event-frequency data “h”), and information regarding the
feature’s “sign” (the feature is signed “s”, unsigned “u”, is the
absolute value of the data “a”, contains only positive data values “p”,
contains only negative data values “n”).
