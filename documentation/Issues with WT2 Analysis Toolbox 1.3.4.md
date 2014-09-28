##Questions/Issues while porting the [WT2 Analysis Toolbox](http://www.mrc-lmb.cam.ac.uk/wormtracker/index.php?action=analysis)##

While porting the toolkit (version 1.3.4) to Python, several questions came up where it was not clear why things were done a certain way.  In other cases we are reasonably confident that there is an error.  We document those differences here.

###Questions###

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