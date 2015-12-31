Expanded Features
=================

JAH NOTE: This file is still a work in progress ...

A full enumeration of these is coded in:

seg\_worm.w.stats.wormStatsInfo.m (old version: wormStatsInfo.m)

-   movement type - might have 4 or 16 values
-   all directions (forward, backward, paused)
-   forward
-   backward
-   paused
    -   *The following apply to above, if the feature crosses zero*
    -   all values
    -   absolute values
    -   positive values
    -   negative values
-   event types - a variable number of features can be derived, some are
    listed ...
-   summary statistics
    -   frequency - \# of events/(total video time)
    -   timeRatio (if worm can't travel) - time in all events/(total
        video time)
    -   ratio.time (equivalent to timeRatio)(only if worm can travel
        during event)
    -   ratio.distance (again, only if worm can travel during event,
        ratio
-   data statistics
    -   time - duration???
    -   inter-time - time until next event, for the last event this is
        NaN
    -   inter-distance - distance traveled until next event (integrated
        or absolute???)
    -   distance
-   NOTE: Code for creating these is in event2stats.m or
    seg\_worm.events.events2stats.m
-   examples
    -   worm.posture.coils - an event without distance
    -   locomotion.motion.forward - event with distance
-   simple data type

