function awesome_contours_oh_yeah_v3(frame_values)
%
%
%   Status:
%   I got intimidated with the amount of work that would need to go into
%   doing this so I went back to v2 which I think I can make work really
%   well without all the hoops that were jumped through here ...
%
%   This will be an implementation of the width algorithm via the old
%   approach. This old approach is REALLY complicated/long.


% #Widths:
% #------------------------------------
% #The caller:
% #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/linearSkeleton.m
% #see helper__skeletonize - callls seg_worm.cv.skeletonize
% #
% #
% #Initial skeletonization:
% #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bcv/skeletonize.m
% #
% #Some refinement:
% #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/cleanSkeleton.m       



%OUTLINE:
%--------
%1) Get max curvature
%
%   https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40contour/contour.m
%   computations of lf and hf angle values using peaksCircDist:
%       https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Butil/peaksCircDist.m
%
%
%   https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/initialize.m
%   https://github.com/openworm/SegWorm/blob/master/Worms/Segmentation/cleanSkeleton.m


%PART 1a:
%-------
USE_MAX = true;
USE_MIN = ~USE_MAX;

%NOTE: I later decided I didn't want to do cutoffs in the
%function so we set the bounds outside the working range
MAX_CUTOFF = -360;
MIN_CUTOFF = 360;

pixels_local = obj.pixels;
avg_worm_segment_length = obj.cc_lengths(end)/obj.N_SEGS;

hf_angle_edge_length = avg_worm_segment_length;
lf_angle_edge_length = 2 * hf_angle_edge_length;

obj.hf_angles_raw = seg_worm.cv.circCurvature(pixels_local, hf_angle_edge_length, obj.cc_lengths);
obj.lf_angles     = seg_worm.cv.circCurvature(pixels_local, lf_angle_edge_length, obj.cc_lengths);

filter_width  = ceil(avg_worm_segment_length/2);
obj.hf_angles = seg_worm.util.circConv(obj.hf_angles_raw ,[],filter_width);

FH = @seg_worm.util.peaksCircDist;
[obj.lf_ap_max,obj.lf_ap_max_I] = FH(obj.lf_angles,lf_angle_edge_length,USE_MAX,MAX_CUTOFF,obj.cc_lengths);
[obj.lf_ap_min,obj.lf_ap_min_I] = FH(obj.lf_angles,lf_angle_edge_length,USE_MIN,MIN_CUTOFF,obj.cc_lengths);
[obj.hf_ap_max,obj.hf_ap_max_I] = FH(obj.hf_angles,hf_angle_edge_length,USE_MAX,MAX_CUTOFF,obj.cc_lengths);
[obj.hf_ap_min,obj.hf_ap_min_I] = FH(obj.hf_angles,hf_angle_edge_length,USE_MIN,MIN_CUTOFF,obj.cc_lengths);

%Part 1b:
%---------
lfCMaxP = contour.lf_ap_max;
lfCMaxI = contour.lf_ap_max_I;
lfCMinP = contour.lf_ap_min;
lfCMinI = contour.lf_ap_min_I;

headI = contour.head_I;
tailI = contour.tail_I;
contour_pixels = contour.pixels;
cc_lengths     = contour.cc_lengths;
wormSegLength  = contour.avg_segment_length(true);

obj.linearSkeleton(...
    headI, tailI, lfCMinP, lfCMinI, ...
    lfCMaxP, lfCMaxI, contour_pixels, wormSegLength, cc_lengths);
