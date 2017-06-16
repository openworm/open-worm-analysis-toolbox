Phenotypic Ontology
===================

(@MichaelCurrie: Doesn't appear to match the horizontal axis labels on
the [pcolor
plots](http://matplotlib.org/examples/pylab_examples/pcolor_small.html),
so what do these represent?)

The phenotypic ontology attempts to find significant features and reduce
our large set of statistical measures to several simple terms. Each
ontological term has a prefix indicating whether all significant
measurements agree that the feature is greater (+), less (-), or
different (.) than the control. A feature is said to be different than
its control whenever the magnitude has no direct meaning (e.g.,
asymmetry does not translate to a clear description of the measurement
being less nor greater than the control) or its measures do not express
a simple magnitude (e.g., the strain pauses with greater frequency but
spends less time in each paused event). Each term also has a suffix
indicating the minimum q-value (significance) found for the term’s
defining measures (\* when q = 0.05; \*\* when q = 0.01; \*\*\* when q =
0.001; and, \*\*\*\* when q = 0.0001). The q-value is a p-value
replacement that corrects for multiple testing10. The ontology terms are
as follows:

1.  Length. The worm’s length.
2.  Width. The worm’s head, midbody, and/or tail width.
3.  Area. The worm’s area if neither the “Length” nor “Width” were found
    significant.
4.  Proportion. The worm’s area/length and/or width/length if neither
    the “Length”, “Width”, nor “Area” were found significant.
5.  Head Bends. The worm’s head bend mean and/or standard deviation.
6.  Tail Bends. The worm’s tail bend mean and/or standard deviation.
7.  Posture Amplitude. The worm’s maximum amplitude and/or amplitude
    ratio.
8.  Posture Wavelength. The worm’s primary and/or secondary wavelength.
9.  Posture Wave. The worm’s track length if neither the “Posture
    Amplitude” nor the “Posture Wavelength” were found significant.
10. Body Bends. The worm’s eccentricity, its number of bends, and/or its
    neck/midbody/hips bend mean and/or standard deviation; only if
    neither the “Posture Amplitude”, “Posture Wavelength”, nor “Posture
    Wave” were found significant.
11. Pose. The worm’s eigenworm projections if neither the “Head Bends”,
    “Body Bends”, “Tail Bends”, “Posture Amplitude”, “Posture
    Wavelength”, nor “Posture Wave” were found significant.
12. Coils. The worm’s coiling event details.
13. Foraging. The worm’s foraging amplitude and/or angular speed.
14. Forward Velocity. The worm’s forward (positive) velocity vector.
15. Backward Velocity. The worm’s backward (negative) velocity vector.
16. Velocity. The worm’s velocity vector magnitude and/or asymmetry if
    neither the “Forward Velocity” nor “Backward Velocity” were found
    significant.
17. Head Motion. The worm’s head-tip and/or head velocity vectors if
    neither the “Foraging”, “Forward Velocity”, nor “Backward Velocity”
    were found significant.
18. Tail Motion. The worm’s tail-tip and/or tail velocity vectors if
    neither the “Forward Velocity” nor “Backward Velocity” were found
    significant.
19. Forward Motion. The worm’s forward motion event details.
20. Pausing. The worm’s pausing event details.
21. Backward Motion. The worm’s backward motion event details.
22. Crawling Amplitude. The worm’s crawling amplitude.
23. Crawling Frequency. The worm’s crawling frequency.
24. Turns. The worm’s omega and/or upsilon event details.
25. Path Range. The worm’s path range.
26. Path Curvature. The worm’s path curvature.
27. Dwelling. The worm’s dwelling if its “Pausing” was not found
    significant.

