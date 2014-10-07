##Questions/Issues while porting the [WT2 Analysis Toolbox](http://www.mrc-lmb.cam.ac.uk/wormtracker/index.php?action=analysis)##

While porting the toolkit (version 1.3.4) to Python, several questions came up where it was not clear why things were done a certain way.  In other cases we are reasonably confident that there is an error.  We document those differences here.

###Questions###

- In the [Nature Methods paper](http://www.nature.com/nmeth/journal/v10/n9/fig_tab/nmeth.2560_F1.html), crawling features while the body motion is paused are omitted.  (Specifically, amplitude and frequency for head, midbody, and tail.)  Why?

- Older version < 3 of segmentation had a bug in saving the failed frames. They have been indexed starting 0 not 1 (because of frame number being generated from time stamp rather than globalFrameCounter). To counter act it the indices for the frames that failed need to be added 1 to shift the failed frames by one and re-allign them. Here we will make a check for that and will raise a flag to add 1 in the upcoming loop: 

```Matlab
shiftFailedFrames = 0;

if ~isempty(failedFrames) && length(failedFrames(:,1)) > 2
    if sum(frameLabels(failedFrames(2:end,1))~='f') ~= 0
        shiftFailedFrames = 1;
    end
end
```

- In computing the worm velocity, the direction is divided by fps, why?

- Dorsal/ventral orientation 

Used in:
Negate if < 2
- seg_worm.feature_helpers.computeVelocity - applied to angular speed

Negate if > 1
- seg_worm.feature_helpers.path.wormPathCurvature - applied to motion
direction ...
- seg_worm.feature_helpers.locomotion.getForaging


#Errors#

- wormTouchFrames 
	- drop and stage code switched ...
	- last frame error is off by 1 (the bit at the end) because 'i' doesn't advance like it does in the loop

- bends (TODO: Clarify which function is being used)

- indexing was incorrect for posture

- findEvent
   - sum data thresholding not implemented correctly
   - event indices are 0 based, not 1 based 

- getAmpWavelength
   - power instead of magnitude is used for comparison
   - primary and secondary wavelength may be switched ...
   - primary and secondary both capped? - drop secondary in that case?

- seg_worm.feature_helpers.computeVelocity 
	- description in supplemental doesn't match reality

- seg_worm.feature_helpers.locomotion.getForaging
	- Is the speed calculated correctly? Multiplying by fps???
	- I'm pretty sure it isn't correct

- seg_worm.feature_helpers.locomotion.getOmegaAndUpsilonTurns
	- Mismatch between description and cutoffs actually used for finding possible event frames.

- seg_worm.feature_helpers.path.wormPathCurvature
	- indices used in body angle doesn't match description
	- NOTE: There is a comment about not using the ends because of noise, but they are in seg_worm.feature_helpers.locomotion.getWormVelocity

- removePartialEvents.m
	- indexing for the end event is off by 1

- worm2StatsInfo.m
	- description of z-score doesn't match reality



#Improvements Made#

**Histogram Binning**

If you run [+seg_worm/+testing/+stats/t001_oldVsNewStats.m](https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Btesting/%2Bstats/t001_oldVsNewStats.m) in the [SegwormMatlabClasses repo](https://github.com/JimHokanson/SegwormMatlabClasses), it will run the old Schafer statistics code, then run the new Jim Hokanson statistics code, then compare them.

Both the old and the new code basically do the same thing:
1. For each of 10 control/experiment pairs of videos, for each feature, calculate p- and q- value statistics for that feature.
2. Aggregate those p- and q- statistics across all features.
3. Merge the p- and q- statistics across all 10 video pairs, treating the 10 video pairs as if they were basically all videos of ONE worm type.

As such we have a p- and q- value giving us, in effect, the probabilities of drawing the numbers we did under the null hypothesis that the mutant worm's features (given by the 10 "experiment" videos) are drawn from the same distribution as the N2 wild-type worm's features were (given by the 10 "control" videos).  

What's interesting is that while the old code, takes about 7 minutes to run, Jim's code takes just a few seconds.

It turns out that much of the speed difference has to do with the way that the histogram bins are set up.

Let's take an example feature, average forward velocity, for two videos, a and b.  Let's say the average forward velocity values taken on by video a lie between 0.5 and 10.5, whereas for video b they lie between -6 and 5.

Let's say we want to use bin widths of 1.

The original code would create bins [0.5, 1.5), [1.5, 2.5), ..., [10.5, 11.5) for a, and bins [-6, -5), [-5, -4), ... [5, 6) for b.

As you can see, these bins do not line up.  Consequently, the merge step when we combine the histograms of videos a and b together is computationally expensive.  

For large numbers the precise bin should not affect the results, so it would be perfectly appropriate to start all the bins for both a and b with whole numbers (i.e. like the bins for b), so that way the bins are identical and the merge step becomes trivial. 


**Videos are split into 10 pieces**

Ends up creating many awkward corner cases when the range of frames you want to use straddles two pieces.

All of that work you'd better only do if it gets you a tremendous performance advantage, which it did not.  Instead it just needlessly complicated the code.
