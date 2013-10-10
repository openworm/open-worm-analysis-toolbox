## Conversion Documentation ##

This code provides a high level overview of going from a video to features for worm analysis and comparison.

## 1) Video to Skeleton, Contour, Events ##

The first step in the conversion process is to take a video and extract base features such as the worm's skeleton and contour, as well as events such as coiling, egg laying, etc.. Given that the Schafer lab has provided us with data from the next step, this can be avoided. It might only be useful to implement this if we wish to improve upon their algorithms as currently there are things that their video parser is unable to analyze. This would involve working with the SegWorm repo and/or porting it to Python (from Matlab).

## 2) Expanded Base Features ##

The data provided by the Schafer lab exists as this expanded set of base features. Many of the features are derived rather easily from the skeleton and contour, such as the area of the head which  can be calculated relatively easy from the contour.

The code for this conversion is currently not publicly available from the Schafer lab. It is described in Ev Yemeni's thesis and in the supplemental material of their 2013 Nature Methods paper (Yemini et al, 2013). Unlike the base set of features which are a bit more obvious, this process will eventually need to be documented and implemented to extract these features from the model.

**Relevant Code**
- https://github.com/openworm/SegWorm/blob/master/Worms/Features/wormDataInfo.m - This is the original code which starts to describe this expanded set of base features
- https://github.com/JimHokanson/SegWorm/blob/classes/new_code/%2Bseg_worm/%2Bfeature/roots.m - This is the same information but it has been cleaned up a bit by @JimHokanson - (Note, this might move as this code gets updated)


## 3) All Features ##

Using the expanded set of base features the Schafer lab has computed a much larger set of features. As an example, the worm length provides 4 features, one overall, and three more when computed during forward movement, backwards movements, and when paused.

**Relevant Code**
- https://github.com/openworm/SegWorm/blob/master/Worms/Statistics/wormStatsInfo.m


## 4) Values for statistical testing (aka "All the other stuff" step) ##

At some point there needs to be a conversion from the video values to those used during statistical testing. The Schafer lab code excludes data in certain situations, normalizes some values, and appears to quantize the frame data to reduce memory requirements. This process will need to eventually be described.

## 5) Summary Matlab Calls ##

- worm parsing (multiple files)
- featureProcess.m (part of GUI code, not yet publicly shared ...)
- worm2histogram.m
- worm2stats.m (This can be skipped, since the stats are a subset of the histogram)
- worm2StatsInfo.m
- wormStats2Matrix.m
  - wormStatsInfo.m

## Sources ##

- Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. X. & Schafer, W. R. A database of Caenorhabditis elegans behavioral phenotypes. Nature methods (2013). doi:10.1038/nmeth.2560