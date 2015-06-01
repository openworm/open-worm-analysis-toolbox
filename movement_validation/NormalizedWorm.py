# -*- coding: utf-8 -*-
"""
This module defines the NormalizedWorm class

"""

import numpy as np
import scipy.io

import copy
import warnings
import os
import time
import matplotlib.pyplot as plt


from . import config
from . import utils
from .basic_worm import WormPartition
from .basic_worm import BasicWorm
from .pre_features import WormParsing

# TODO: remove this dependency by moving feature_comparisons to utils and 
#       renaming it to something more general.
from .features import feature_comparisons as fc


class NormalizedWorm(WormPartition):
    """
    Encapsulates the notion of a worm's elementary measurements, scaled
    (i.e. "normalized") to 49 points along the length of the worm.

    The data consists of 13 Numpy arrays (where n is the number of frames):
   - Of shape (49,2,n):   (Inherited from NormalizedSkeletonAndContour)
        vulva_contour
        non_vulva_contour
        skeleton
    - Of shape (49,n):    (Not inherited)
        angles        
        in_out_touches
        widths
    - Of shape (n):       (Not inherited)
        length
        head_area
        tail_area
        vulva_area
        non_vulva_area
        segmentation_status   (not used in further processing)   (inherited)
        frame_code            (not used in further processing)   (inherited)

    Also, some metadata:
        ventral_mode
        plate_wireframe_video_key                                (inherited)
        
    """

    def __init__(self, other=None):
        """
        Populates an empty normalized worm.
        If other is specified, this becomes a copy constructor.
        
        """
        WormPartition.__init__(self)

        if other is None:
            # DEBUG: (Note from @MichaelCurrie:)
            # This should be set by the normalized worm file, since each
            # worm subjected to an experiment is manually examined to find the
            # vulva so the ventral mode can be determined.  Here we just set
            # the ventral mode to a default value as a stopgap measure
            self.ventral_mode = config.DEFAULT_VENTRAL_MODE
        else:
            # Copy constructor
            attributes = ['skeleton', 'vulva_contour', 'non_vulva_contour',
                          'angles', 'in_out_touches', 'widths', 'length',
                          'head_area', 'tail_area', 'vulva_area', 
                          'non_vulva_area', 'segmentation_status', 
                          'frame_code', 'plate_wireframe_video_key', 
                          'ventral_mode']
            for a in attributes:
                setattr(self, a, copy.deepcopy(getattr(other, a)))


    @classmethod
    def from_BasicWorm_factory(cls, basic_worm):
        """
        Factory classmethod for creating a normalized worm with a basic_worm
        as input.  This requires calculating all the "pre-features" of 
        the worm.
        
        Parameters
        -----------
        basic_worm: Instance of BasicWorm.  Contains either:
            h_skeleton AND/OR
            h_vulva_contour and h_non_vulva_contour
            
        Returns
        -----------
        An instance of NormalizedWorm
        
        """
        nw = cls()
        
        #TODO: Need to add on testing for normalized data as an input
        #TODO: This could be simplified, although order may matter somewhat
        if basic_worm.h_vulva_contour is not None:
            # 1. Derive skeleton and widths from contour
            nw.widths, h_skeleton = \
                WormParsing.computeWidths(basic_worm.h_vulva_contour, 
                                          basic_worm.h_non_vulva_contour)

            # 2. Calculate angles            
            nw.angles = WormParsing.calculateAngles(h_skeleton)

            # 3. Normalize the skeleton and contour to 49 points per frame
            nw.skeleton = WormParsing.normalizeAllFramesXY(h_skeleton)
            nw.vulva_contour = WormParsing.normalizeAllFramesXY(
                                            basic_worm.h_vulva_contour)
            nw.non_vulva_contour = WormParsing.normalizeAllFramesXY(
                                            basic_worm.h_non_vulva_contour)            

            # 4. TODO: Calculate area 
            # The old method was:
            # Using known transition regions, count the # of 'on' pixels in
            # the image. Presumably we would use something more akin
            # to the eccentricity feature code
            
            # 5. TODO:
            # Still missing:
            # - segmentation_status
            # - frame codes
            # - in_out_touches
            # I think these things would best be handled by first identifying
            # the feature code that uses them, then determing what if anything
            # we really need to calculate. Perhaps we need to modify the
            # feature code instead.
        else:
            # With no contour, let's assume we have a skeleton.
            # Measurements that cannot be calculated (e.g. areas) are simply
            # marked None.
            nw.angles = WormParsing.calculateAngles(basic_worm.skeleton)
            nw.skeleton = WormParsing.normalizeAllFramesXY(basic_worm.skeleton)
            nw.vulva_contour = None
            nw.non_vulva_contour = None
            nw.head_area = None
            nw.tail_area = None
            nw.vulva_area = None
            nw.non_vulva_area = None           
        
        # 6. Calculate length
        nw.length = WormParsing.computeSkeletonLengths(nw.skeleton)
        
        return nw
            

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
        nw.plate_wireframe_video_key = 'Schafer'
        
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
                # a string of length n, showing, for each frame of the video:
                # s = segmented
                # f = segmentation failed
                # m = stage movement
                # d = dropped frame
                # n??? - there is reference in some old code to this
                # after loading this we convert it to a numpy array.
                'segmentation_status',
                # shape is (1 n), see comments in
                # seg_worm.parsing.frame_errors
                'frame_codes',
                'vulva_contours',     # shape is (49, 2, n) integer
                'non_vulva_contours',  # shape is (49, 2, n) integer
                'skeletons',          # shape is (49, 2, n) integer
                'angles',             # shape is (49, n) integer (degrees)
                'in_out_touches',     # shpe is (49, n)
                'lengths',            # shape is (n) integer
                'widths',             # shape is (49, n) integer
                'head_areas',         # shape is (n) integer
                'tail_areas',         # shape is (n) integer
                'vulva_areas',        # shape is (n) integer
                'non_vulva_areas',    # shape is (n) integer
                'x',                  # shape is (49, n) integer
                'y']                  # shape is (49, n) integer

            # Here I use powerful python syntax to reference data elements of s
            # dynamically through built-in method getattr
            # that is, getattr(s, x)  works syntactically just like s.x,
            # only x is a variable, so we can do a list comprehension with it!
            for key in data_keys:
                setattr(nw, key, getattr(staging_data, key))

            # We don't need the eigenworm path here, as it's the same
            # for all worm files.
            del(nw.EIGENWORM_PATH)
            # x and y are redundant since that information is already 
            # in "skeletons"
            del(nw.x)
            del(nw.y)
            
            # Now for something pedantic: only use plural nouns for
            # those measurements taken along multiple points per frame
            # for those with just one data point per frame, it should be 
            # singular.
            # i.e. plural for numpy arrays of shape (49, n)
            #     singular for numpy arrays of shape (n)
            # and singular for numpy arrays of shape (49, 2, n)
            # (where n is the number of frames)

            nw.skeleton = nw.skeletons
            nw.vulva_contour = nw.vulva_contours
            nw.non_vulva_contour = nw.non_vulva_contours
            del(nw.skeletons)
            del(nw.vulva_contours)
            del(nw.non_vulva_contours)
            nw.length = nw.lengths
            nw.head_area = nw.head_areas
            nw.tail_area = nw.tail_areas
            nw.vulva_area = nw.vulva_areas
            nw.non_vulva_area = nw.non_vulva_areas
            nw.frame_code = nw.frame_codes
            del(nw.lengths)
            del(nw.head_areas)
            del(nw.tail_areas)
            del(nw.vulva_areas)
            del(nw.non_vulva_areas)
            del(nw.frame_codes)

            # Let's change the string of length n to a numpy array of single
            # characters of length n, to be consistent with the other data
            # structures
            nw.segmentation_status = np.array(list(nw.segmentation_status))
            
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
        bw = BasicWorm()

        bw.skeleton = np.copy(self.skeleton)
        bw.h_vulva_contour = np.copy(self.vulva_contour)
        bw.h_non_vulva_contour = np.copy(self.non_vulva_contour)
        bw.ventral_mode = self.ventral_mode
        
        return bw

  
    def validate(self):
        """
        Checks array lengths, etc. to ensure that this is a valid instance
        and no further problems will arise if further processing is attempted
        on this instance

        """
        # TODO
        return True

    def plot_path(self, posture_index):
        """
        Plot the path of the contour, skeleton and widths
        
        Parameters
        ----------------
        posture_index: int
            The desired posture point (along skeleton and contour) to plot.
        
        """
        vc = self.vulva_contour[posture_index,:,:]
        nvc = self.non_vulva_contour[posture_index,:,:]
        skeleton_x = self.skeleton[posture_index,0,:]
        skeleton_y = self.skeleton[posture_index,1,:]
        
        plt.scatter(vc[0,:], vc[1,:])
        plt.scatter(nvc[0,:], nvc[1,:])
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
        vc = self.vulva_contour[:,:,frame_index]
        nvc = self.non_vulva_contour[:,:,frame_index]
        skeleton = self.skeleton[:,:,frame_index]
        
        plt.scatter(vc[:,0], vc[:,1], c='red')
        plt.scatter(nvc[:,0], nvc[:,1], c='blue')
        plt.scatter(skeleton[:,0], skeleton[:,1], c='black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

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
            v = s[48, :,:]-s[0,:,:]  
            # find the angle of this vector
            self._angle = np.arctan(v[1,:]/v[0,:])*(180/np.pi)

            return self._angle

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
                s_mean = np.ones(np.shape(s)) * self.centre
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

        To make this work I believe we need to pre-broadcast rot_matrix into
        the skeleton points dimension (the one with 49 points) so that we have
          2 x 2 x 49 x n, times 2 x 49 x n
        #s1 = np.rollaxis(self.skeleton, 1)

        #rot_matrix = np.ones(np.shape(s1)) * rot_matrix

        #self.skeleton_rotated = rot_matrix.dot(self.skeleton)

        """
        try:
            return self._orientation_free_skeleton
        except AttributeError:
            orientation = self.angle
    
            # Flip and convert to radians
            a = -orientation * (np.pi / 180)
    
            rot_matrix = np.array([[np.cos(a), -np.sin(a)],
                                   [np.sin(a),  np.cos(a)]])
    
            # We need the x,y listed in the first dimension
            s1 = np.rollaxis(self.centred_skeleton, 1)
    
            # For example, here is the first point of the first frame rotated:
            # rot_matrix[:,:,0].dot(s1[:,0,0])
    
            # ATTEMPTING TO CHANGE rot_matrix from 2x2x49xn to 2x49xn
            # rot_matrix2 = np.ones((2, 2, np.shape(s1)[1], np.shape(s1)[2])) * rot_matrix
    
            s1_rotated = []
    
            # Rotate the worm frame-by-frame and add these skeletons to a list
            for frame_index in range(self.num_frames):
                s1_rotated.append(rot_matrix[:, :, frame_index].dot(s1[:,:, frame_index]))
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

    @property
    def is_segmented(self):
        """
        Returns a 1-d boolean numpy array of whether 
        or not, frame-by-frame, the given frame was segmented

        """
        return self.segmentation_status == 's'

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
        if(len(np.shape(d)) < 3):
            raise Exception("Position Limits Is Only Implemented for 2D data")
        return (np.nanmin(d[dimension, 0, :]), 
                np.nanmax(d[dimension, 1, :]))

    @property
    def contour_x(self):
        """ 
          Return the approximate worm contour, derived from data
          NOTE: The first and last points are duplicates, so we omit
                those on the second set. We also reverse the contour so that
                it encompasses an "out and back" contour
        """
        vc = self.vulva_contour
        nvc = self.non_vulva_contour
        return np.concatenate((vc[:, 0, :], nvc[-2:0:-1, 0,:]))    

    @property
    def contour_y(self):
        vc = self.vulva_contour
        nvc = self.non_vulva_contour
        return np.concatenate((vc[:, 1, :], nvc[-2:0:-1, 1,:]))    

    @property
    def skeleton_x(self):
        return self.skeleton[:, 0, :]

    @property
    def skeleton_y(self):
        return self.skeleton[:, 1, :]


    def __eq__(self,other):
        x1 = self.skeleton_x.flatten()
        x2 = other.skeleton_x.flatten()
        y1 = self.skeleton_y.flatten()
        y2 = other.skeleton_y.flatten()
        
        #TODO: Do this on a frame by frame basis, do some sort of distance 
        #computation rather than all together. This might hide bad frames        
        
        fc.corr_value_high(x1,x2,'asdf')
        fc.corr_value_high(y1,y2,'asdf')

        #return \
            #fc.corr_value_high(self.length, other.length, 'morph.length')  and \
            #self.width == other.width and \
            #fc.corr_value_high(self.area, other.area, 'morph.area')      and \
            #fc.corr_value_high(self.area_per_length, other.area_per_length, 'morph.area_per_length') and \
            #fc.corr_value_high(self.width_per_length, other.width_per_length, 'morph.width_per_length')


    def __repr__(self):
        #TODO: This omits the properties above ...
        return utils.print_object(self)

