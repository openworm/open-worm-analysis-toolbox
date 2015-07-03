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
        temp_angle_list = [] # TODO: pre-allocate the space we need
                      
        for frame_index, cur_skeleton in enumerate(h_skeleton):
            if cur_skeleton is None:
                temp_angle_list.append([])
            else:
                sx = cur_skeleton[0,:]
                sy = cur_skeleton[1,:]
                cur_skeleton2 = np.rollaxis(cur_skeleton, 1)
                cc = WormParsing.chain_code_lengths_cum_sum(cur_skeleton2)
    
                # This is from the old code
                edge_length = cc[-1]/12               
                
                # We want all vertices to be defined, and if we look starting
                # at the left_I for a vertex, rather than vertex for left and
                # right then we could miss all middle points on worms being 
                # vertices
                
                left_lengths = cc - edge_length
                right_lengths = cc + edge_length
    
                valid_vertices_I = utils.find((left_lengths > cc[0]) & \
                                              (right_lengths < cc[-1]))
                
                left_lengths = left_lengths[valid_vertices_I]
                right_lengths = right_lengths[valid_vertices_I]                
                
                left_x = np.interp(left_lengths,cc,sx)
                left_y = np.interp(left_lengths,cc,sy)
            
                right_x = np.interp(right_lengths,cc,sx)
                right_y = np.interp(right_lengths,cc,sy)
    
                d2_y = sy[valid_vertices_I] - right_y
                d2_x = sx[valid_vertices_I] - right_x
                d1_y = left_y - sy[valid_vertices_I]
                d1_x = left_x - sx[valid_vertices_I] 
    
                frame_angles = np.arctan2(d2_y, d2_x) - np.arctan2(d1_y ,d1_x)
                
                frame_angles[frame_angles > np.pi] -= 2*np.pi
                frame_angles[frame_angles < -np.pi] += 2*np.pi
                
                # Convert to degrees
                frame_angles *= 180/np.pi
                
                all_frame_angles = np.full_like(cc, np.NaN)
                all_frame_angles[valid_vertices_I] = frame_angles
                
                temp_angle_list.append(all_frame_angles)

                
        return WormParsing.normalize_all_frames(temp_angle_list, 
                                                h_skeleton)

    #%%    
    @staticmethod
    def compute_area(contour):
        """
        Compute the area of the worm, for each frame of video, from a 
        normalized contour.

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
        contour_plus_one = contour.take(range(1,np.shape(contour)[0]+1),
                                        mode='wrap', axis=0)
    
        # Now we use the Shoelace formula to calculate the area of a simple
        # polygon for each frame.
        # Credit to Avelino Javer for suggesting this.
        signed_area = np.nansum(contour[:,0,:]*contour_plus_one[:,1,:] -
                                contour[:,1,:]*contour_plus_one[:,0,:], 0) / 2
    
        # Frames where the contour[:,:,k] is all NaNs will result in a 
        # signed_area[k] = 0.  We must replace these 0s with NaNs.
        signed_area[np.flatnonzero(np.isnan(contour[0,0,:]))] = np.NaN

        return np.abs(signed_area)

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
        return np.sum(WormParsing.chain_code_lengths(skeleton),
                      axis=0)


    #%%
    @staticmethod
    def chain_code_lengths(skeleton):
        """
        Computes the chain-code lengths of the skeleton for each frame.
        
        Computed from the skeleton by converting the chain-code 
        pixel length to microns.
        
        These chain-code lengths are based on the Freeman 8-direction
        chain codes:
        
        3  2  1
        4  P  0
        5  6  7        
        
        Given a sequence of (x,y)-coordinates, we could obtain a sequence of
        direction vectors, coded according to the following scheme.
        
        However, since we just need the lengths, we don't need to actually
        calculate all of these codes.  Instead we just calculate the 
        Euclidean 2-norm from pixel i to pixel i+1.
        
        Note:
        Our method is actually different than if we stepped from pixel 
        to pixel in one-pixel increments, since in our method the distances 
        can be something other than multiples of the 1- or sqrt(2)- steps 
        characteristic in Freeman 8-direction chain codes.
        
        
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
        # For each frame, for x and y, take the difference between skeleton
        # points: (hence axis=0).  Resulting shape is (k-1,2,n)
        skeleton_diffs = np.diff(skeleton, axis=0)
        # Now for each frame, for each (diffx, diffy) pair along the skeleton,
        # find the magnitude of this vector.  Resulting shape is (k-1,n)
        chain_code_lengths = np.linalg.norm(skeleton_diffs, axis=1)

        return chain_code_lengths
        
    #%%
    @staticmethod
    def chain_code_lengths_cum_sum(skeleton):
        """
        Compute the Freeman 8-direction chain-code length.
        
        Calculate the distance between a set of points and then calculate
        their cumulative distance from the first point.
        
        The first value returned has a value of 0 by definition.
        
        Parameters
        ----------------
        skeleton: numpy array
            Shape should be (k,2), where k is the number of 
            points per frame

        """
        if np.size(skeleton) == 0:
            # Handle empty set - don't add 0 as first element
            return np.empty([])
        else:
            distances = WormParsing.chain_code_lengths(skeleton)
            # Prepend a zero element so that distances' numpy array length 
            # is the same as skeleton's
            distances = np.concatenate([np.array([0.0]), distances])
            
            return np.cumsum(distances)

    #%%
    @staticmethod
    def normalize_all_frames_xy(heterocardinal_property):
        """
        Normalize a "heterocardinal" skeleton or contour into a "homocardinal"
        one, where each frame has the same number of points.
        
        We always normalize to config.N_POINTS_NORMALIZED points per frame.

        Parameters
        --------------
        prop_to_normalize: list of numpy arrays
            the outermost dimension, that of the lists, has length n
            the numpy arrays are of shape 

        Returns
        --------------
        numpy array of shape (49,2,n)
        
        """
        n_frames = len(heterocardinal_property)
        normalized_data = np.full([config.N_POINTS_NORMALIZED, 2, n_frames],
                                   np.NaN)

        for iFrame, cur_frame_value in enumerate(heterocardinal_property):
            if cur_frame_value is not None:
                # We need cur_frame_value to have shape (k,2), not (2,k)
                cur_frame_value2 = np.rollaxis(cur_frame_value, 1)
                cc = WormParsing.chain_code_lengths_cum_sum(cur_frame_value2)

                # Normalize both the x and the y
                normalized_data[:,0,iFrame] = \
                    WormParsing.normalize_parameter(cur_frame_value[0,:], cc)
                normalized_data[:,1,iFrame] = \
                    WormParsing.normalize_parameter(cur_frame_value[1,:], cc)
        
        return normalized_data          
    
    #%%
    @staticmethod
    def normalize_all_frames(property_to_normalize, xy_data):
        """
        We have a skeleton
        
        Normalize a (heterocardinal) array of lists of variable length
        down to a numpy array of shape (49,n).
        
        Parameters
        --------------
        property_to_normalize: list of length n, of numpy arrays of shape (ki)
            The property that needs to be evenly sampled
        xy_data: list of length n, of numpy arrays of shape (2, ki)
            The skeleton or contour points corresponding to the location
            along the worm where the property_to_normalize was recorded
        
        Returns
        --------------
        numpy array of shape (49,n)
            prop_to_normalize, now normalized down to 49 points per frame
        
        """
        assert(len(property_to_normalize) == len(xy_data))
        
        normalized_data_shape = [config.N_POINTS_NORMALIZED, 
                                 len(property_to_normalize)]

        normalized_data = np.full(normalized_data_shape, np.NaN)

        # Normalize one frame at a time
        for frame_index, (cur_frame_value, cur_xy) in \
                                enumerate(zip(property_to_normalize, xy_data)):
            if cur_xy is not None:
                # We need cur_xy to have shape (k,2), not (2,k)
                cur_xy_reshaped = np.rollaxis(cur_xy, axis=1)
                running_lengths = \
                    WormParsing.chain_code_lengths_cum_sum(cur_xy_reshaped)

                # Normalize cur_frame_value over an evenly-spaced set of
                # 49 values spread from running_lengths[0] to 
                #                       running_lengths[-1]
                normalized_data[:,frame_index] = \
                            WormParsing.normalize_parameter(cur_frame_value, 
                                                            running_lengths)
        
        return normalized_data
    
    #%%
    @staticmethod
    def normalize_parameter(prop_to_normalize, running_lengths):
        """
        This function finds where all of the new points will be when evenly
        sampled (in terms of chain code length) from the first to the last 
        point in the old data.

        These points are then related to the old points. If a new point is at
        an old point, the old point data value is used. If it is between two
        old points, then linear interpolation is used to determine the new 
        value based on the neighboring old values.

        NOTE: This method should be called for just one frame's data at a time.

        NOTE: For better or worse, this approach does not smooth the new data,
        since it just linearly interpolates.

        See http://docs.scipy.org/doc/numpy/reference/generated/
            numpy.interp.html

        Parameters
        -----------
        prop_to_normalize: numpy array of shape (k,) or (2,k)
            The parameter to be interpolated, where k is the number of
            points.  These are the values to be normalized.
        running_lengths: numpy array of shape (k)
            The positions along the worm where the property was
            calculated.  It is these positions that are to be "normalized",
            or made to be evenly spaced.  The parameter will then be 
            calculated at the new, evenly spaced, positions.

        Returns
        -----------
        Numpy array of shape (k,) or (k,2) depending on the provided
        shape of prop_to_normalize
        
        Notes
        ------------
        Old code:
        https://github.com/openworm/SegWorm/blob/master/ComputerVision/
                chainCodeLengthInterp.m

        """
        # Create n evenly spaced points between the first and last point in 
        # old_lengths
        new_lengths = np.linspace(running_lengths[0], running_lengths[-1],
                                  config.N_POINTS_NORMALIZED)

        # Interpolate in both the 2-d and 1-d cases
        if len(np.shape(prop_to_normalize)) == 2:
            # Assume shape is (2,k,n)
            interp_x = np.interp(new_lengths, running_lengths, 
                                 prop_to_normalize[0,:])
            interp_y = np.interp(new_lengths, running_lengths, 
                                 prop_to_normalize[1,:])
            # Compbine interp_x and interp_y together in shape ()
            return np.rollaxis(np.array([interp_x, interp_y]), axis=1)
        else:
            # Assume shape is (k,n)
            return np.interp(new_lengths, running_lengths, prop_to_normalize)

