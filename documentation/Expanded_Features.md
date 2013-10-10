## Expanded Features ##

JAH NOTE: This file is still a work in progress ...

A full enumeration of these is coded in:

seg_worm.w.stats.wormStatsInfo.m (old version: wormStatsInfo.m)

- movement type - might have 4 or 16 values
  - all directions (forward, backward, paused)
  - forward
  - backward
  - paused
     - *The following apply to above, if the feature crosses zero*
     - all values
     - absolute values
     - positive values
     - negative values
- event types - a variable number of features can be derived, some are listed ...
  - summary statistics
     - frequency
     - time ratio (if worm can't travel) - time in event versus total experiment time, if the event occurs multiple times this is presumably a sum of the durations of each event, over the duration of the experiment
     - ratio.time (equivalent to time ratio)(only if worm can travel during event)
     - ratio.distance (again, only if worm can travel during event, ratio
  - data statistics
     - time - duration???
     - inter-time - time until next event
     - inter-distance - distance traveled until next event (integrated or absolute???)
     - distance
