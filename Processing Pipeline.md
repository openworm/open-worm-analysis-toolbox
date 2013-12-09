## Processing Pipeline ##

NOTE: processing steps are written in parentheses.  The state of the data is described at each numerical step in the pipeline.

(conduct experiment:
case: REAL WORM: Capture video of worm movement using a test protocol, in tandem with a control worm.
case: VIRTUAL WORM: Run simulation engine, output video.)

*1. Raw video + tracking plate movement data + other metadata (time of filming, vulva location, whether worm flipped during vidoe, strain of worm used, Lab name, etc)

(machine vision processing step)

*2. Measurements: Worm contour and skeleton

(normalize each from to just 49 points)
 
 3. Normalized measurements

(feature calculation in python based on WT2 code)
 
 4. Worm features

(stats calculation in python based on WT2 code)

5. Worm statistics

(enter statistics as a record into the database)

6. Database of statistics on multiple worms

(interactively perform operations on the database, to produce the final result:)

7. Summary pixel grid and other charts of worms, compared.
