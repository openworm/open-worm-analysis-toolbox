# -*- coding: utf-8 -*-
"""
Pre-features calculation methods.

These methods are exclusively called by NormalizedWorm.from_BasicWorm_factory

The methods of the classes below are all static, so their grouping into
classes is only for convenient grouping and does not have any further
functional meaning.

Original notes from Schafer Lab paper on this process:
------------------------------------------------------
"Once the worm has been thresholded, its contour is extracted by tracing the
worm's perimeter. The head and tail are located as sharp, convex angles on
either side of the contour. The skeleton is extracted by tracing the midline of
the contour from head to tail. During this process, widths and angles are
measured at each skeleton point to be used later for feature computation. At
each skeleton point, the width is measured as the distance between opposing
contour points that determine the skeleton midline.  Similarly, each skeleton
point serves as a vertex to a bend and is assigned the supplementary angle to
this bend. The supplementary angle can also be expressed as the difference in
tangent angles at the skeleton point. This angle provides an intuitive
measurement. Straight, unbent worms have an angle of 0 degrees. Right angles
are 90 degrees. And the largest angle theoretically possible, a worm bending
back on itself, would measure 180 degress. The angle is signed to provide the
bend's dorsal-ventral orientation. When the worm has its ventral side internal
to the bend (i.e., the vectors forming the angle point towards the ventral
side), the bending angle is signed negatively.

Pixel count is a poor measure of skeleton and contour lengths. For this reason,
we use chain-code lengths (Freeman 1961). Each laterally-connected pixel is
counted as 1. Each diagonally-connected pixel is counted as sqrt(2). The
supplementary angle is determined, per skeleton point, using edges 1/12 the
skeleton''s chain-code length, in opposing directions, along the skeleton. When
insufficient skeleton points are present, the angle remains undefined (i.e. the
first and last 1/12 of the skeleton have no bending angle defined). 1/12 of the
skeleton has been shown to effectively measure worm bending in previous
trackers and likely reflects constraints of the bodywall muscles, their
innervation, and cuticular rigidity (Cronin et al. 2005)."

"""
import warnings
import numpy as np

from .. import config, utils
from .skeleton_calculator1 import SkeletonCalculatorType1
from .pre_features_helpers import WormParserHelpers

#%%


class WormParsing(object):
    """
    This might eventually move somewhere else, but at least it is
    contained within the class. It was originally in the Normalized Worm
    code which was making things a bit overwhelming.

    TODO: Self does not refer to WormParsing ...

    """

    @staticmethod
    def compute_skeleton_and_widths(h_ventral_contour,
                                    h_dorsal_contour,
                                    frames_to_plot=[]):
        """
        Compute widths and a heterocardinal skeleton from a heterocardinal
        contour.

        Parameters
        -------------------------
        h_ventral_contour: list of numpy arrays.
            Each frame is an entry in the list.
        h_dorsal_contour:
        frames_to_plot: list of ints
            Optional list of frames to plot, to show exactly how the
            widths and skeleton were calculated.

        Returns
        -------------------------
        (h_widths, h_skeleton): tuple
            h_widths : the heterocardinal widths, frame by frame
            h_skeleton : the heterocardinal skeleton, frame by frame.

        Notes
        --------------------------
        This is just a wrapper method for the real method, contained in
        SkeletonCalculatorType1.  In the future we might use
        alternative algorithms so this may become the place we swap them in.

        """
        (h_widths, h_skeleton) = \
            SkeletonCalculatorType1.compute_skeleton_and_widths(
            h_ventral_contour,
            h_dorsal_contour,
            frames_to_plot=[])

        return (h_widths, h_skeleton)
    #%%

    @staticmethod
    def _h_array2list(h_vector):
        ''' we need to change the data from a (49,2,n) array to a list of (2,49),
        a bit annoying but necessary
        '''
        # we need it to be shape (2,49) instead of (49,2) so we transpose
        h_list = [h_vector[:,:, ii].T for ii in range(h_vector.shape[2])]
        # let's use None instead of a all nan vector to indicate an invalid skeleton
        h_list = [None if np.all(np.isnan(x)) else x for x in h_list]
        return h_list

    @staticmethod
    def compute_angles(h_skeleton):
        """
        Calculate the angles

        From the Schafer Lab description:

        "Each normalized skeleton point serves as a vertex to a bend and
        is assigned the supplementary angle to this bend. The supplementary
        angle can also be expressed as the difference in tangent angles at
        the skeleton point. This angle provides an intuitive measurement.
        Straight, unbent worms have an angle of 0 degrees. Right angles
        are 90 degrees. And the largest angle theoretically possible, a
        worm bending back on itself, would measure 180 degress."

        Parameters
        ----------------
        h_skeleton: list of length n, of lists of skeleton coordinate points.
            The heterocardinal skeleton

        Returns
        ----------------
        numpy array of shape (49,n)
            An angle for each normalized point in each frame.

        Notes
        ----------------
        Original code in:
        https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
                 %2Bseg_worm/%2Bworm/%40skeleton/skeleton.m
        https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/
                 %2Bseg_worm/%2Bcv/curvature.m

        Note, the above code is written for the non-normalized worm ...
        edge_length= total_length/12

        Importantly, the above approach calculates angles not between
        neighboring pairs but over a longer stretch of pairs (pairs that
        exceed the edge length). The net effect of this approach is to
        smooth the angles

        vertex index - First one where the distance from the tip to this
                       point is greater than the edge length

        s = norm_data[]

        temp_s = np.full([config.N_POINTS_NORMALIZED,n_frames],np.NaN)
        for iFrame in range(n_frames):
           temp_

        TODO: sign these angles using ventral_mode ?? - @MichaelCurrie

        """
        #%%
        temp_angle_list = []  # TODO: pre-allocate the space we need
        
        #i am changing the skeleton to a list, this function can deal with 3D numpy arrays (like in the case of normalized worm)
        if not isinstance(h_skeleton, list):
            h_skeleton = WormParsing._h_array2list(h_skeleton)
            
        for frame_index, cur_skeleton in enumerate(h_skeleton):
            if cur_skeleton is None:
                temp_angle_list.append([])
            else:
                assert cur_skeleton.shape[0] == 2
                
                sx = cur_skeleton[0, :]
                sy = cur_skeleton[1, :]
                cur_skeleton2 = np.rollaxis(cur_skeleton, 1)
                cc = WormParserHelpers.chain_code_lengths_cum_sum(
                    cur_skeleton2)

                # This is from the old code
                edge_length = cc[-1] / 12

                # We want all vertices to be defined, and if we look starting
                # at the left_I for a vertex, rather than vertex for left and
                # right then we could miss all middle points on worms being
                # vertices

                left_lengths = cc - edge_length
                right_lengths = cc + edge_length

                valid_vertices_I = utils.find((left_lengths > cc[0]) &
                                              (right_lengths < cc[-1]))

                left_lengths = left_lengths[valid_vertices_I]
                right_lengths = right_lengths[valid_vertices_I]

                left_x = np.interp(left_lengths, cc, sx)
                left_y = np.interp(left_lengths, cc, sy)

                right_x = np.interp(right_lengths, cc, sx)
                right_y = np.interp(right_lengths, cc, sy)

                d2_y = sy[valid_vertices_I] - right_y
                d2_x = sx[valid_vertices_I] - right_x
                d1_y = left_y - sy[valid_vertices_I]
                d1_x = left_x - sx[valid_vertices_I]

                frame_angles = np.arctan2(d2_y, d2_x) - np.arctan2(d1_y, d1_x)

                frame_angles[frame_angles > np.pi] -= 2 * np.pi
                frame_angles[frame_angles < -np.pi] += 2 * np.pi

                # Convert to degrees
                frame_angles *= 180 / np.pi

                all_frame_angles = np.full_like(cc, np.NaN)
                all_frame_angles[valid_vertices_I] = frame_angles

                temp_angle_list.append(all_frame_angles)
                
                
        return  WormParserHelpers.normalize_all_frames(
            temp_angle_list, h_skeleton, config.N_POINTS_NORMALIZED)
        #%%
    #%%
    @staticmethod
    def compute_signed_area(contour):
        """
        Compute the signed area of the worm, for each frame of video, from a
        normalized contour. The area should be negative for an clockwise contour
        and positive for a anticlockwise.

        Parameters
        -------------------------
        contour: a (98,2,n)-shaped numpy array
            The contour points, in order around the worm, with two redundant
            points at the head and tail.


        Returns
        -------------------------
        area: float
            A numpy array of scalars, giving the area in each
            frame of video.  so if there are 4000 frames, area would have
            shape (4000,)

        """
        # We do this to avoid a RuntimeWarning taking the nanmean of
        # frames with nothing BUT nan entries: for those frames nanmean
        # returns nan (correctly) but still raises a RuntimeWarning.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            contour_mean = np.nanmean(contour, 0, keepdims=False)

        # Centre the contour about the origin for each frame
        # this is technically not necessary but it shrinks the magnitude of
        # the coordinates we are about to multiply for a potential speedup
        contour -= contour_mean

        # We want a new 3D array, where all the points (i.e. axis 0)
        # are shifted forward by one and wrapped.
        # That is, for a given frame:
        #  x' sub 1 = x sub 2,
        #  x' sub 2 = x sub 3,
        #  ...,
        #  x` sub n = x sub 1.     (this is the "wrapped" index)
        # Same for y.
        contour_plus_one = contour.take(range(1, contour.shape[0] + 1),
                                        mode='wrap', axis=0)

        # Now we use the Shoelace formula to calculate the area of a simple
        # polygon for each frame.
        # Credit to Avelino Javer for suggesting this.
        signed_area = np.nansum(contour[:,0,:] * contour_plus_one[:,1,:] - contour[:,1,:] * contour_plus_one[:,0,:],0) / 2

        # Frames where the contour[:,:,k] is all NaNs will result in a
        # signed_area[k] = 0.  We must replace these 0s with NaNs.
        signed_area[np.flatnonzero(np.isnan(contour[0, 0, :]))] = np.NaN

        return signed_area

    #%%
    @staticmethod
    def compute_skeleton_length(skeleton):
        """
        Computes the length of the skeleton for each frame.

        Computed from the skeleton by converting the chain-code
        pixel length to microns.

        Parameters
        ----------
        skeleton: numpy array of shape (k,2,n)
            The skeleton positions for each frame.
            (n is the number of frames, and
             k is the number points in each frame)

        Returns
        -----------
        length: numpy array of shape (n)
            The (chain-code) length of the skeleton for each frame.

        """
        # For each frame, sum the chain code lengths to get the total length
        return np.sum(WormParserHelpers.chain_code_lengths(skeleton),
                      axis=0)
