# -*- coding: utf-8 -*-
"""
This module defines the NormalizedWorm class

"""

import numpy as np
import scipy.io

import copy
import warnings
import os
import matplotlib.pyplot as plt

from .. import config, utils
from .basic_worm import WormPartition
from .basic_worm import BasicWorm
from .pre_features import WormParsing
from .pre_features_helpers import WormParserHelpers
from .video_info import VideoInfo


class NormalizedWorm(WormPartition):
    """
    Encapsulates the notion of a worm's elementary measurements, scaled
    (i.e. "normalized") to 49 points along the length of the worm.

    The data consists of 7 Numpy arrays (where n is the number of frames):
   - Of shape (49,2,n):
        ventral_contour
        dorsal_contour
        skeleton
    - Of shape (49,n):
        angles
        widths
    - Of shape (n):
        length
        area

    It also contains video metadata, in :
        video_info : An instance of VideoInfo

    """

    def __init__(self, other=None):
        """
        Populates an empty normalized worm.
        If other is specified, this becomes a copy constructor.

        """
        WormPartition.__init__(self)

        if other is None:
            # By default, leave all variables uninitialized
            # Initialize an empty VideoInfo instance, though:
            self.video_info = VideoInfo()
        else:
            # Copy constructor
            attributes = ['skeleton', 'ventral_contour', 'dorsal_contour',
                          'angles', 'widths', 'length', 'area'
                          'video_info']
            for a in attributes:
                setattr(self, a, copy.deepcopy(getattr(other, a)))

    @classmethod
    def from_BasicWorm_factory(cls, basic_worm, frames_to_plot_widths=[]):
        """
        Factory classmethod for creating a normalized worm with a basic_worm
        as input.  This requires calculating all the "pre-features" of
        the worm.

        Parameters
        -----------
        basic_worm: Instance of BasicWorm.  Contains either:
            h_skeleton AND/OR
            h_ventral_contour and h_dorsal_contour
        frames_to_plot_widths: list of ints
            Optional list of frames to plot, to show exactly how the
            widths and skeleton were calculated.

        Returns
        -----------
        An instance of NormalizedWorm

        # TODO: Need to add on testing for normalized data as an input

        """
        nw = cls()
        bw = basic_worm
        nw.video_info = bw.video_info

        if bw.h_ventral_contour is not None:
            # 1. Derive skeleton and widths from contour
            nw.widths, h_skeleton = \
                WormParsing.compute_skeleton_and_widths(bw.h_ventral_contour,
                                                        bw.h_dorsal_contour,
                                                        frames_to_plot_widths)
            # 3. Normalize the skeleton, widths and contour to 49 points
            #    per frame
            nw.skeleton = WormParserHelpers.\
                normalize_all_frames_xy(h_skeleton,
                                        config.N_POINTS_NORMALIZED)

            nw.widths = WormParserHelpers.\
                normalize_all_frames(nw.widths, h_skeleton,
                                     config.N_POINTS_NORMALIZED)

            nw.ventral_contour = WormParserHelpers.\
                normalize_all_frames_xy(bw.h_ventral_contour,
                                        config.N_POINTS_NORMALIZED)

            nw.dorsal_contour = WormParserHelpers.\
                normalize_all_frames_xy(bw.h_dorsal_contour,
                                        config.N_POINTS_NORMALIZED)
        else:
            # With no contour, let's assume we have a skeleton.
            # Measurements that cannot be calculated (e.g. areas) are simply
            # marked None.
            nw.skeleton = WormParserHelpers.\
                normalize_all_frames_xy(bw.h_skeleton,
                                        config.N_POINTS_NORMALIZED)
            nw.ventral_contour = None
            nw.dorsal_contour = None


        # Give frame code 1 if the frame is good and the general error
        # code 100 if the frame is bad.
        nan_mask = np.all(np.isnan(nw.skeleton), axis=(0,1))
        nw.video_info.frame_code = 1 * ~nan_mask + 100 * nan_mask

        return nw

    @classmethod
    def from_normalized_array_factory(cls, skeleton, widths, ventral_contour, dorsal_contour):
        '''
        I want to be able to construct a normalized worm by giving previously normalized data as input.
        Probably some inputs could be made optional, but in that case it is better to use BasicWorm
        '''
        #check the dimensions are correct
        dat = (skeleton, ventral_contour, dorsal_contour, widths) #pack to make calculations easier
        tot_frames = skeleton.shape[-1]
        assert all(x.shape[0] == config.N_POINTS_NORMALIZED for x in dat)
        assert all(x.shape[-1] == tot_frames for x in dat)
        assert all(x.shape[1] == 2 for x in (skeleton, ventral_contour, dorsal_contour))

        nw = cls()
        nw.ventral_contour = ventral_contour
        nw.dorsal_contour = dorsal_contour
        nw.skeleton = skeleton
        nw.widths = widths

        # Give frame code 1 if the frame is good and the general error
        # code 100 if the frame is bad.
        nan_mask = np.all(np.isnan(nw.skeleton), axis=(0,1))
        nw.video_info.frame_code = 1 * ~nan_mask + 100 * nan_mask

        return nw



    
    @property
    def length(self):
        try:
            return self._length
        except:
            self._length = WormParsing.compute_skeleton_length(self.skeleton)
            return self._length


    @property
    def area(self):
        try:
            return self._area
        except:
            if self.signed_area is not None:
                self._area = np.abs(self.signed_area)
            else:
                self._area = None
            return self._area

    @property
    def signed_area(self):
        try:
            return self._signed_area
        except:
            if self.ventral_contour is not None:
                self._signed_area = WormParsing.compute_signed_area(self.contour)
            else:
                self._x = None
            return self._signed_area




    @property
    def angles(self):
        try:
            return self._angles 
        except:
            #I could use the unnormalized skeleton (more points), but it shouldn't make much a huge difference.
            self._angles = WormParsing.compute_angles(self.skeleton)

            if self.video_info.ventral_mode == 2:
                #switch in the angle sign in case of the contour orientation is anticlockwise
                self._angles = -self._angles
            
            return self._angles

            #A second option would be to use the contour orientation to find the correct side
            # if self.signed_area is not None:
            #     #I want to use the signed area to determine the contour orientation.
            #     #first assert all the contours have the same orientation. This might be a problem 
            #     #if the worm does change ventral/dorsal orientation, but for the moment let's make it a requirement.
            #     valid = self.signed_area[~np.isnan(self.signed_area)]
            #     assert np.all(valid>=0) if valid[0]>=0 else np.all(valid<=0) 

            #     #if the orientation is anticlockwise (negative signed area) change the sign of the angles
            #     if valid[0] < 0:
            #         self._angles *= -1

            




    @classmethod
    def from_schafer_file_factory(cls, data_file_path):
        """
        Load full Normalized Worm data from the Schafer File

        data_file_path: the path to the MATLAB file

        These files were created at the Schafer Lab in a format used
        prior to MATLAB's switch to HDF5, which they did in
        MATLAB version 7.3.

        """
        nw = cls()
        nw.video_info = VideoInfo()

        if(not os.path.isfile(data_file_path)):
            raise Exception("Data file not found: " + data_file_path)
        else:
            data_file = scipy.io.loadmat(data_file_path,
                                         # squeeze unit matrix dimensions:
                                         squeeze_me=True,
                                         # force return numpy object
                                         # array:
                                         struct_as_record=False)

            # All the action is in data_file['s'], which is a numpy.ndarray
            # where data_file['s'].dtype is an array showing how the data is
            # structured.  It is structured in precisely the order specified
            # in data_keys below:

            staging_data = data_file['s']

            # NOTE: These are aligned to the order in the files.
            # these will be the keys of the dictionary data_dict
            data_keys = [
                # this just contains a string for where to find the
                # eigenworm file.  we do not use this, however, since
                # the eigenworm postures are universal to all worm files,
                # so the file is just stored in the /features directory
                # of the source code, and is loaded at the features
                # calculation step
                'EIGENWORM_PATH',
                # We can't load this one since we have an @property method
                # whose name clashes with it, derived from frame_codes
                #'segmentation_status',
                # Each code corresponds to a success / failure mode of the
                # computer vision algorithm.
                'frame_codes',        # shape is (n) integer
                'vulva_contours',     # shape is (49, 2, n) integer
                'non_vulva_contours',  # shape is (49, 2, n) integer
                'skeletons',          # shape is (49, 2, n) integer
                'angles',             # shape is (49, n) integer (degrees)
                'in_out_touches',     # shape is (49, n)
                'lengths',            # shape is (n) integer
                'widths',             # shape is (49, n) integer
                'head_areas',         # shape is (n) integer
                'tail_areas',         # shape is (n) integer
                'vulva_areas',        # shape is (n) integer
                'non_vulva_areas',    # shape is (n) integer
                'x',                  # shape is (49, n) integer
                'y']                  # shape is (49, n) integer

            # Here I use powerful python syntax to reference data elements of
            # s dynamically through built-in method getattr
            # that is, getattr(s, x)  works syntactically just like s.x,
            # only x is a variable, so we can do a list comprehension with it!
            for key in data_keys:
                # Some pre-features are dynamically calculated so their actual
                # data is stored with a '_' prefix.
                if key == 'angles':
                    out_key = '_' + key
                else:
                    out_key = key
                setattr(nw, out_key, getattr(staging_data, key))

            # We don't need the eigenworm path here, as it's the same
            # for all worm files.
            del(nw.EIGENWORM_PATH)
            # x and y are redundant since that information is already
            # in "skeletons"
            del(nw.x)
            del(nw.y)
            # in_out_touches: I can find no reference to them in Ev's thesis,
            # nor does any of the feature calculation code depend on them.
            # the only reason we have them at all is because the one example
            # file we've been using has in_out_touches as an array.
            # the shape is (49, 4642) in that file, and ALL entries are NaN.
            # Thus for all the above reasons I'm going to ignore it.
            del(nw.in_out_touches)

            # Now for something pedantic: only use plural nouns for
            # those measurements taken along multiple points per frame
            # for those with just one data point per frame, it should be
            # singular.
            # i.e. plural for numpy arrays of shape (49, n)
            #     singular for numpy arrays of shape (n)
            # and singular for numpy arrays of shape (49, 2, n)
            # (where n is the number of frames)

            nw.skeleton = nw.skeletons
            nw.ventral_contour = nw.vulva_contours
            nw.dorsal_contour = nw.non_vulva_contours
            del(nw.skeletons)
            del(nw.vulva_contours)
            del(nw.non_vulva_contours)
            nw.head_area = nw.head_areas
            nw.tail_area = nw.tail_areas
            nw.vulva_area = nw.vulva_areas
            nw.non_vulva_area = nw.non_vulva_areas
            del(nw.head_areas)
            del(nw.tail_areas)
            del(nw.vulva_areas)
            del(nw.non_vulva_areas)

            # Frame codes should be stored in the VideoInfo object
            nw.video_info.frame_code = nw.frame_codes
            del(nw.frame_codes)

            # Somehow those four areas apparently add up to the total area
            # of the worm.  No feature requires knowing anything more
            # than just the total area of the worm, so for NormalizedWorm
            # we just store one variable, area, for the whole worm's area.
            
            nw._area = nw.head_area + nw.tail_area + \
                nw.vulva_area + nw.non_vulva_area
            del(nw.head_area)
            del(nw.tail_area)
            del(nw.vulva_area)
            del(nw.non_vulva_area)


            nw._length = nw.lengths
            del(nw.lengths)
            
            return nw

    def get_BasicWorm(self):
        """
        Return an instance of NormalizedSkeletonAndContour containing this
        instance of NormalizedWorm's basic data.

        There is no purpose for this within the standard pipeline - going
        back to a BasicWorm from a NormalizedWorm would only be done
        for verification of code integrity purposes.

        Note that we can't "de-normalize" the worm so if the original
        BasicWorm from which this NormalizedWorm was derived was properly
        heterocardinal, that information is lost.  All frames in our
        generated BasicWorm here will have 49 points and thus will remain
        normalized.

        """
        bw = BasicWorm.from_contour_factory(self.ventral_contour,
                                            self.dorsal_contour)

        bw.video_info = self.video_info

        return bw


    def validate(self):
        """
        Checks array lengths, etc. to ensure that this is a valid
        instance and no further problems will arise if further
        processing is attempted on this instance

        """
        # TODO
        return True

    def rotated(self, theta_d):
        """
        Returns a NormalizedWorm instance with each frame rotated by
        the amount given in the per-frame theta_d array.

        Parameters
        ---------------------------------------
        theta_d: 1-dimensional ndarray of dtype=float
          The frame-by-frame rotation angle in degrees.
          A 1-dimensional n-element array where n is the number of
          frames, giving a rotation angle for each frame.

        Returns
        ---------------------------------------
        A new NormalizedWorm instance with the same worm, rotated
        in each frame by the requested amount.

        """
        #theta_r = theta_d * (np.pi / 180)

        #%Unrotate worm
        #%-----------------------------------------------------------------
        # wwx = bsxfun(@times,sx,cos(theta_r)) + \
        #       bsxfun(@times,sy,sin(theta_r));
        # wwy = bsxfun(@times,sx,-sin(theta_r)) +
        # bsxfun(@times,sy,cos(theta_r));

        # TODO
        return self

    @property
    def centre(self):
        """
        Frame-by-frame mean of the skeleton points

        Returns
        ---------------------------------------
        A numpy array of length n, where n is the number of
        frames, giving for each frame the mean of the skeleton points.

        """
        try:
            return self._centre
        except AttributeError:
            # We do this to avoid a RuntimeWarning taking the nanmean of
            # frames with nothing BUT nan entries: for those frames nanmean
            # returns nan (correctly) but still raises a RuntimeWarning.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                self._centre = np.nanmean(self.skeleton, 0, keepdims=False)

            return self._centre

    @property
    def angle(self):
        """
        Frame-by-frame mean of the skeleton points

        Returns
        ---------------------------------------
        A numpy array of length n, giving for each frame
        the angle formed by the first and last skeleton point.

        """
        try:
            return self._angle
        except AttributeError:
            s = self.skeleton
            # obtain vector between first and last skeleton point
            v = s[-1, :, :] - s[0, :, :]
            # find the angle of this vector
            self._angle = np.arctan2(v[1, :], v[0, :]) * (180 / np.pi)

            return self._angle

    @property
    def dropped_frames_mask(self):
        """
        Which frames are "dropped", i.e. which frames have the first
        skeleton X-coordinate set to NaN.

        Returns
        ------------------
        boolean numpy array of shape (n,) where n is the number of frames
            True if frame is dropped in the skeleton or contour

        Note
        ------------------
        We are assuming that self.validate() == True, i.e. that the
        skeleton and contour are NaN along all points in frames where
        ANY of the points are NaN.

        """
        return np.isnan(self.skeleton[0, 0, :])

    @property
    def centred_skeleton(self):
        """
        Return a skeleton numpy array with each frame moved so the
        centroid of the worm is 0,0

        Returns
        ---------------------------------------
        A numpy array with the above properties.

        """
        try:
            return self._centred_skeleton
        except AttributeError:
            s = self.skeleton

            if s.size != 0:
                s_mean = np.ones(s.shape) * self.centre
                self._centred_skeleton = s - s_mean
            else:
                self._centred_skeleton = s

            return self._centred_skeleton

    @property
    def orientation_free_skeleton(self):
        """
        Perform both a rotation and a translation of the skeleton

        Returns
        ---------------------------------------
        A numpy array, which is the centred and rotated normalized
        worm skeleton.

        Notes
        ---------------------------------------
        To perform this matrix multiplication we are multiplying:
          rot_matrix * s
        This is shape 2 x 2 x n, times 2 x 49 x n.
        Basically we want the first matrix treated as two-dimensional,
        and the second matrix treated as one-dimensional,
        with the results applied elementwise in the other dimensions.

        To make this work I believe we need to pre-broadcast rot_matrix
        into the skeleton points dimension (the one with 49 points) so
        that we have
          2 x 2 x 49 x n, times 2 x 49 x n
        #s1 = np.rollaxis(self.skeleton, 1)

        #rot_matrix = np.ones(s1.shape) * rot_matrix

        #self.skeleton_rotated = rot_matrix.dot(self.skeleton)

        """
        try:
            return self._orientation_free_skeleton
        except AttributeError:
            orientation = self.angle

            # Flip and convert to radians
            a = -orientation * (np.pi / 180)

            rot_matrix = np.array([[np.cos(a), -np.sin(a)],
                                   [np.sin(a), np.cos(a)]])

            # We need the x,y listed in the first dimension
            s1 = np.rollaxis(self.centred_skeleton, 1)

            # For example, here is the first point of the first frame rotated:
            # rot_matrix[:,:,0].dot(s1[:,0,0])

            # ATTEMPTING TO CHANGE rot_matrix from 2x2x49xn to 2x49xn
            # rot_matrix2 = np.ones((2, 2, s1.shape[1],
            #                        s1.shape[2])) * rot_matrix

            s1_rotated = []

            # Rotate the worm frame-by-frame and add these skeletons to a list
            for frame_index in range(self.num_frames):
                s1_rotated.append(rot_matrix[:, :, frame_index].dot
                                  (s1[:, :, frame_index]))
            # print(np.shape(np.rollaxis(rot_matrix[:,:,0].dot(s1[:,:,0]),0)))

            # Save the list as a numpy array
            s1_rotated = np.array(s1_rotated)

            # Fix the axis settings
            self._orientation_free_skeleton = \
                np.rollaxis(np.rollaxis(s1_rotated, 0, 3), 1)

            return self._orientation_free_skeleton

    @property
    def num_frames(self):
        """
        The number of frames in the video.

        Returns
        ---------------------------------------
        int
          number of frames in the video

        """
        try:
            return self._num_frames
        except AttributeError:
            self._num_frames = self.skeleton.shape[2]

            return self._num_frames

    def position_limits(self, dimension, measurement='skeleton'):
        """
        Maximum extent of worm's travels projected onto a given axis

        Parameters
        ---------------------------------------
        dimension: specify 0 for X axis, or 1 for Y axis.

        Notes
        ---------------------------------------
        Dropped frames show up as NaN.
        nanmin returns the min ignoring such NaNs.

        """
        d = getattr(self, measurement)
        if(len(d.shape) < 3):
            raise Exception("Position Limits Is Only Implemented for 2D data")
        return (np.nanmin(d[:, dimension, :]),
                np.nanmax(d[:, dimension, :]))

    @property
    def contour(self):
        return self.get_contour(keep_redundant_points=True)

    @property
    def contour_without_redundant_points(self):
        return self.get_contour(keep_redundant_points=False)

    def get_contour(self, keep_redundant_points=True):
        """
        The contour of the worm as one 96-point or 98-point polygon.

        That is:

        Go from ventral_contour shape (49,2,n) and
            dorsal_contour shape (49,2,n) to
            contour with      shape (96,2,n) or (98,2,n)

        Why 96 instead of 49x2 = 98?
        Because the first and last points are duplicates, so if
        keep_redundant_points=False, we omit those on the second set.

        In either case we reverse the contour so that
        it encompasses an "out and back" contour.

        """
        if keep_redundant_points:
            return np.concatenate((self.ventral_contour,
                                   self.dorsal_contour[::-1, :, :]))
        else:
            return np.concatenate((self.ventral_contour,
                                   self.dorsal_contour[-2:0:-1, :, :]))

    @property
    def contour_x(self):
        # Note that this includes 2 redundant points.
        return self.contour[:, 0, :]

    @property
    def contour_y(self):
        # Note that this includes 2 redundant points.
        return self.contour[:, 1, :]

    @property
    def skeleton_x(self):
        return self.skeleton[:, 0, :]

    @property
    def skeleton_y(self):
        return self.skeleton[:, 1, :]

    @property
    def ventral_contour_x(self):
        return self.ventral_contour[:, 0, :]

    @property
    def ventral_contour_y(self):
        return self.ventral_contour[:, 1, :]

    @property
    def dorsal_contour_x(self):
        return self.dorsal_contour[:, 0, :]

    @property
    def dorsal_contour_y(self):
        return self.dorsal_contour[:, 1, :]

    def __eq__(self, other):
        """
        Compare this Normalized worm against another.

        TODO: Idea from @JimHokanson:
        Do this on a frame by frame basis, do some sort of distance
        computation rather than all together. This might hide bad frames
        i.e. besides using correlation for comparison, a normalized distance
        comparison that could catch extreme outliers would also be useful

        """
        attribute_list = ['skeleton_x', 'skeleton_y',
                          'ventral_contour_x', 'ventral_contour_y',
                          'dorsal_contour_x', 'dorsal_contour_y',
                          'angles', 'widths', 'length', 'area']

        return utils.compare_attributes(self, other, attribute_list,
                                        high_corr_value=0.94,
                                        merge_nans_list=['angles'])

    def __repr__(self):
        # TODO: This omits the properties above ...
        return utils.print_object(self)

    def plot_path(self, posture_index):
        """
        Plot the path of the contour, skeleton and widths

        Parameters
        ----------------
        posture_index: int
            The desired posture point (along skeleton and contour) to plot.

        """
        vc = self.ventral_contour[posture_index, :, :]
        nvc = self.dorsal_contour[posture_index, :, :]
        skeleton_x = self.skeleton[posture_index, 0, :]
        skeleton_y = self.skeleton[posture_index, 1, :]

        plt.scatter(vc[0, :], vc[1, :])
        plt.scatter(nvc[0, :], nvc[1, :])
        plt.scatter(skeleton_x, skeleton_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_posture(self, frame_index):
        """
        Show a scatterplot of the contour, skeleton and widths of frame #frame

        Parameters
        ----------------
        frame_index: int
            The desired frame # to plot.

        """
        vc = self.ventral_contour[:, :, frame_index]
        nvc = self.dorsal_contour[:, :, frame_index]
        skeleton = self.skeleton[:, :, frame_index]

        plt.scatter(vc[:, 0], vc[:, 1], c='red')
        plt.scatter(nvc[:, 0], nvc[:, 1], c='blue')
        plt.scatter(skeleton[:, 0], skeleton[:, 1], c='black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_contour(self, frame_index):
        NormalizedWorm.plot_contour_with_labels(
            self.contour[:, :, frame_index])

    @staticmethod
    def plot_contour_with_labels(contour, frame_index=0):
        """
        Makes a beautiful plot with all the points labeled.

        Parameters:
        One frame's worth of a contour

        """
        contour_x = contour[:, 0, frame_index]
        contour_y = contour[:, 1, frame_index]
        plt.plot(contour_x, contour_y, 'r', lw=3)
        plt.scatter(contour_x, contour_y, s=35)
        labels = list(str(l) for l in range(0, len(contour_x)))
        for label_index, (label, x, y), in enumerate(
                zip(labels, contour_x, contour_y)):
            # Orient the label for the first half of the points in one direction
            # and the other half in the other
            if label_index <= len(contour_x) // 2 - \
                    1:  # Minus one since indexing
                xytext = (20, -20)                     # is 0-based
            else:
                xytext = (-20, 20)
            plt.annotate(
                label, xy=(
                    x, y), xytext=xytext, textcoords='offset points', ha='right', va='bottom', bbox=dict(
                    boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(
                    arrowstyle='->', connectionstyle='arc3,rad=0'))  # , xytext=(0,0))
