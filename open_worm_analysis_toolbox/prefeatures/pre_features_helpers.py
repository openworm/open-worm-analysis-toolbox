# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:15:41 2015

@author: mcurrie
"""
import numpy as np


class WormParserHelpers:

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
            distances = WormParserHelpers.chain_code_lengths(skeleton)
            # Prepend a zero element so that distances' numpy array length
            # is the same as skeleton's
            distances = np.concatenate([np.array([0.0]), distances])

            return np.cumsum(distances)

    #%%
    @staticmethod
    def normalize_all_frames_xy(heterocardinal_property, num_norm_points):
        """
        Normalize a "heterocardinal" skeleton or contour into a "homocardinal"
        one, where each frame has the same number of points.

        Parameters
        --------------
        heterocardinal_property: list of numpy arrays
            the outermost dimension, that of the lists, has length n
            the numpy arrays are of shape (2,ki)
        num_norm_points: int
            The number of points to normalize to.

        Returns
        --------------
        numpy array of shape (49,2,n)

        """
        n_frames = len(heterocardinal_property)
        normalized_data = np.full([num_norm_points, 2, n_frames],
                                  np.NaN)

        for iFrame, cur_frame_value in enumerate(heterocardinal_property):
            if cur_frame_value is not None:
                # We need cur_frame_value to have shape (k,2), not (2,k)
                cur_frame_value2 = np.rollaxis(cur_frame_value, 1)
                cc = WormParserHelpers.chain_code_lengths_cum_sum(
                    cur_frame_value2)

                # Normalize both the x and the y
                normalized_data[:, 0, iFrame] = WormParserHelpers.normalize_parameter(
                    cur_frame_value[0, :], cc, num_norm_points)
                normalized_data[:, 1, iFrame] = WormParserHelpers.normalize_parameter(
                    cur_frame_value[1, :], cc, num_norm_points)

        return normalized_data

    #%%
    @staticmethod
    def normalize_all_frames(property_to_normalize, xy_data, num_norm_points):
        """
        Normalize a property as it articulates along a skeleton.

        Normalize a (heterocardinal) array of lists of variable length
        down to a numpy array of shape (num_norm_points,n).

        Parameters
        --------------
        property_to_normalize: list of length n, of numpy arrays of shape (ki)
            The property that needs to be evenly sampled
        xy_data: list of length n, of numpy arrays of shape (2, ki)
            The skeleton or contour points corresponding to the location
            along the worm where the property_to_normalize was recorded
        num_norm_points: int
            The number of points to normalize to.

        Returns
        --------------
        numpy array of shape (49,n)
            prop_to_normalize, now normalized down to 49 points per frame

        """
        assert(len(property_to_normalize) == len(xy_data))

        # Create a blank array of shape (49,n)
        normalized_data_shape = [num_norm_points, len(property_to_normalize)]
        normalized_data = np.full(normalized_data_shape, np.NaN)

        # Normalize one frame at a time
        for frame_index, (cur_frame_value, cur_xy) in \
                enumerate(zip(property_to_normalize, xy_data)):
            if cur_xy is not None:
                # We need cur_xy to have shape (k,2), not (2,k)
                cur_xy_reshaped = np.rollaxis(cur_xy, axis=1)
                running_lengths = WormParserHelpers.chain_code_lengths_cum_sum(
                    cur_xy_reshaped)

                # Normalize cur_frame_value over an evenly-spaced set of
                # 49 values spread from running_lengths[0] to
                #                       running_lengths[-1]
                normalized_data[:, frame_index] = \
                    WormParserHelpers.normalize_parameter(cur_frame_value,
                                                          running_lengths,
                                                          num_norm_points)

        return normalized_data

    #%%
    @staticmethod
    def normalize_parameter(prop_to_normalize, running_lengths,
                            num_norm_points):
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
        num_norm_points: int
            The number of points to normalize to.

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
                                  num_norm_points)

        # Interpolate in both the 2-d and 1-d cases
        if len(prop_to_normalize.shape) == 2:
            # Assume shape is (2,k,n)
            interp_x = np.interp(new_lengths, running_lengths,
                                 prop_to_normalize[0, :])
            interp_y = np.interp(new_lengths, running_lengths,
                                 prop_to_normalize[1, :])
            # Compbine interp_x and interp_y together in shape ()
            return np.rollaxis(np.array([interp_x, interp_y]), axis=1)
        else:
            # Assume shape is (k,n)
            return np.interp(new_lengths, running_lengths, prop_to_normalize)
