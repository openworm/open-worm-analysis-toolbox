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

from . import config
from . import utils
from .basic_worm import WormPartition, BasicWorm
from .pre_features import WormParsing

# TODO: remove this dependency by moving feature_comparisons to utils and 
#       renaming it to something more general.
from .features import feature_comparisons as fc


class NormalizedWorm(BasicWorm, WormPartition):
    """
    Encapsulates the notion of a worm's elementary measurements, scaled
    (i.e. "normalized") to 49 points along the length of the worm.

    The data consists of 13 Numpy arrays (where n is the number of frames):
    - Of shape (49,2,n):
        vulva_contour
        non_vulva_contour
        skeleton
    - Of shape (49,n):
        angles        
        in_out_touches
        widths
    - Of shape (n):
        length
        head_area
        tail_area
        vulva_area
        non_vulva_area
        segmentation_status   (not used in further processing)
        frame_code            (not used in further processing)

    Also, some metadata:
        plate_wireframe_video_key
        
    """

    def __init__(self, normalized_worm=None):
        """
        Populates an empty normalized worm.
        If copy is specified, this becomes a copy constructor.
        
        """
        print("in NormalizedWorm consructor")
        if not normalized_worm:
            BasicWorm.__init__(self)
            WormPartition.__init__(self)
            self.angles = np.array([], dtype=float)

            # DEBUG: (Note from @MichaelCurrie:)
            # This should be set by the normalized worm file, since each
            # worm subjected to an experiment is manually examined to find the
            # vulva so the ventral mode can be determined.  Here we just set
            # the ventral mode to a default value as a stopgap measure
            self.ventral_mode = config.DEFAULT_VENTRAL_MODE

        else:
            # TODO: not sure if this will work correctly..
            super(NormalizedWorm, self).__init__(normalized_worm)
            self.angles = copy.deepcopy(normalized_worm.angles)


    @classmethod
    def from_BasicWorm_factory(cls, basic_worm):
        nw = NormalizedWorm()        
        
        # Call BasicWorm's copy constructor:
        super(NormalizedWorm, nw).__init__(basic_worm)

        # TODO: We should probably validate that the worm is valid before
        #       calculating pre-features.
        
        nw.calculate_pre_features()
        
        return nw


    @classmethod
    def from_schafer_file_factory(cls, data_file_path):
        """
        Load full Normalized Worm data from the Schafer File

        data_file_path: the path to the MATLAB file
        
        These files were created at the Schafer Lab in a format used 
        prior to MATLAB's switch to HDF5, which they did in MATLAB version 7.3.


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

            # All the action is in data_file['s'], which is a numpy.ndarray where
            # data_file['s'].dtype is an array showing how the data is structured.
            # it is structured in precisely the order specified in data_keys
            # below

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
        Return an instance of BasicWorm containing this instance of 
        NormalizedWorm's basic data.

        """
        bw = BasicWorm()
        bw.head = np.copy(self.skeleton[0,:,:])
        bw.tail = np.copy(self.skeleton[-1,:,:])
        bw.skeleton = np.copy(self.skeleton)
        # We have to reverse the contour points of the non_vulva_contour
        # so that the tail end picks up where the tail end of vulva_contour
        # left off:
        bw.contour = np.copy(np.concatenate((self.vulva_contour, 
                                             self.non_vulva_contour[::-1,:,:]), 
                                            axis=0))
        bw.ventral_mode = 'CW'
        
        return bw
    
    def calculate_pre_features(self):
        """
        Calculate "pre-features" given basic information about the worm.
        
        1. If contour is specified, normalize it to 98 points evenly split 
           between head and tail
        2. If skeleton is specified, normalize it to 49 points
        3. Calculate vulva_contour and non_vulva_contour from contour 
           (preferably) or that's not available, skeleton 
           (use an approximation in this case)
        4. Calculate widths, angles, and in_out_touches for each skeleton point
           and each frame
        5. Calculate length, head_area, tail_area, vulva_area, non_vulva area
           for each frame 

        """
        print("Calculating pre-features")
        start_time = time.time()

        # 1. Normalize contour (if available)
        # TODO.  This code is wrong, it just assigns skeleton to the contour.
        self.vulva_contour = self.skeleton
        self.non_vulva_contour = self.skeleton        
        # 2. Normalize skeleton (if available)
        # TODO
        # 3. Calculate contour (if contour isn't available)
        # or Calculate skeleton (if skeleton isn't available)
        # TODO

        # 4. Calculate widths, angles, in_out_touches
        widths, skeleton = WormParsing.computeWidths(self.vulva_contour, 
                                                     self.non_vulva_contour)
        self.widths = WormParsing.normalizeAllFrames(self, widths, skeleton)
        
        self.angles = WormParsing.calculateAngles(self, skeleton)

        elapsed = time.time() - start_time
        print('Elapsed time:', elapsed)

        # 5. Calculate length, head_area, tail_area, vulva_area, non_vulva area
        #    for each frame 
        

        # TODO
        # Much of what needs to be done here is accomplished in the 
        # below jim_pre_features_algorithm so pull that into here!
        
        # 1. If contour is specified, normalize it to 98 points evenly split 
        #   between head and tail        
        # calculating the "hemiworms" requires stepping through frame-by-frame
        # since we cannot assume that the number of points between head
        # and tail and between tail and head remains constant between frames.
        pass


    def jim_pre_features_algorithm(self, skeleton, 
                                   vulva_contour, non_vulva_contour, 
                                   is_valid):
        """ 
        Create an instance from skeleton and contour data
        
        
        TODO: We need to clarify the value for a missing frame
        I think currently it is [] but perhaps it should be None        
        
        Parameters
        --------------------------------------- 
        skeleton : list
            Each element in the list may be empty (bad frame) or be of size
            2 x n, where n varies depending on the # of pixels the worm occupied
            in the given frame
            
        vulva_contour and non_vulva_contour should start and end at the same locations, from head to tail

        """     
        
        # TODO: Skeleton is optional, but at some point we'll probably need
        # to make contour optional as well        
        
        # TODO: Do I want to grab the skeleton from here????
        t = time.time()

        # The # of points in the skeleton changes on a per frame basis. If we use
        # any old parameters based on the passed in skeleton and the new widths,
        # we will have a length mismatch        
        
        widths, skeleton = WormParsing.computeWidths(self, vulva_contour, non_vulva_contour)
        self.widths = WormParsing.normalizeAllFrames(self, widths, skeleton)
        elapsed = time.time() - t
        print('Elapsed time: %g'%(elapsed))

        
        self.angles = WormParsing.calculateAngles(self, skeleton)
        
        #t = time.time()
        self.skeleton = WormParsing.normalizeAllFramesXY(self, skeleton)
        self.vulva_contour = WormParsing.normalizeAllFramesXY(self, vulva_contour)
        self.non_vulva_contour = WormParsing.normalizeAllFramesXY(self, non_vulva_contour)
        #elapsed = time.time() - t
        
        self.length = WormParsing.computeSkeletonLengths(self, skeleton)
  
          
        
    
        
        
        """
        From Ev's Thesis:
        3.3.1.6 - page 126 (or 110 as labeled in document)
        For each section, we begin at its center on both sides of the contour. We then
        walk, pixel by pixel, in either direction until we hit the end of the section on
        opposite sides, for both directions. The midpoint, between each opposing pixel
        pair, is considered the skeleton and the distance between these pixel pairs is
        considered the width for each skeleton point.
        3) Food tracks, noise, and other disturbances can form spikes on the worm
        contour. When no spikes are present, our walk attempts to minimize the width
        between opposing pairs of pixels. When a spike is present, this strategy may
        cause one side to get stuck in the spike while the opposing side walks.
        Therefore, when a spike is present, the spiked side walks while the other side
        remains still.
        """
        
        #Areas:
        #------------------------------------
            
        """
        Final needed attributes:
        ------------------------
        #1) ??? segmentation_status   
        #2) ??? frame_codes
        #3) DONE vulva_contours        49 x 2 x n_frames
        #4) DONE non_vulva_contours    49 x 2 x n_frames
        #5) DONE skeletons : numpy.array
          - (49,2,n_frames)
        #6) DONE angles : numpy.array
          - (49,n_frames)
        #7) in_out_touches ????? 49 x n_frames
        #8) DONE lengths : numpy.array
          - (nframes,)
        #9) DONE widths : numpy.array
          - (49,n_frames)
        #10) head_areas
        #11) tail_areas
        #12) vulva_areas
        #13) non_vulva_areas 
        
        
        """

    
    def validate(self):
        """
        Checks array lengths, etc. to ensure that this is a valid instance
        and no further problems will arise if further processing is attempted
        on this instance

        """
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
        # wwx = bsxfun(@times,sx,cos(theta_r)) + bsxfun(@times,sy,sin(theta_r));
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
        # We do this to avoid a RuntimeWarning taking the nanmean of frames
        # with nothing BUT nan entries: for those frames nanmean returns nan
        # (correctly) but still raises a RuntimeWarning.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return np.nanmean(self.skeleton, 0, keepdims=False)

    @property
    def angle(self):
        """
        Frame-by-frame mean of the skeleton points

        Returns
        ---------------------------------------    
        A numpy array of length n, giving for each frame
        the angle formed by the first and last skeleton point.

        """
        s = self.skeletons
        # obtain vector between first and last skeleton point
        v = s[48, :,:]-s[0,:,:]  
        # find the angle of this vector
        return np.arctan(v[1,:]/v[0,:])*(180/np.pi)

    @property
    def centred_skeleton(self):
        """ 
        Return a skeleton numpy array with each frame moved so the 
        centroid of the worm is 0,0

        Returns
        ---------------------------------------    
        A numpy array with the above properties.

        """
        s = self.skeletons
        
        if(s.size == 0):
            s_mean = np.ones(np.shape(s)) * np.nanmean(s, 0, keepdims=False)
            return s - s_mean
        else:
            return s

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
        #s1 = np.rollaxis(self.skeletons, 1)

        #rot_matrix = np.ones(np.shape(s1)) * rot_matrix

        #self.skeletons_rotated = rot_matrix.dot(self.skeletons)    

        """

        centred_skeleton = self.centred_skeleton()
        orientation = self.angle

        a = -orientation * (np.pi / 180)

        rot_matrix = np.array([[np.cos(a), -np.sin(a)],
                               [np.sin(a),  np.cos(a)]])

        # we need the x,y listed in the first dimension
        s1 = np.rollaxis(centred_skeleton, 1)

        # for example, here is the first point of the first frame rotated:
        # rot_matrix[:,:,0].dot(s1[:,0,0])

        # ATTEMPTING TO CHANGE rot_matrix from 2x2x49xn to 2x49xn
        # rot_matrix2 = np.ones((2, 2, np.shape(s1)[1], np.shape(s1)[2])) * rot_matrix

        s1_rotated = []

        # rotate the worm frame-by-frame and add these skeletons to a list
        for frame_index in range(self.num_frames):
            s1_rotated.append(rot_matrix[:, :, frame_index].dot(s1[:,:, frame_index]))
        # print(np.shape(np.rollaxis(rot_matrix[:,:,0].dot(s1[:,:,0]),0)))

        # save the list as a numpy array
        s1_rotated = np.array(s1_rotated)

        # fix the axis settings
        return np.rollaxis(np.rollaxis(s1_rotated, 0, 3), 1)


    @property
    def num_frames(self):
        """ 
        The number of frames in the video.

        Returns
        ---------------------------------------    
        int
          number of frames in the video

        """

        # ndarray.shape returns a tuple of array dimensions.
        # the frames are along the first dimension i.e. [0].
        return self.skeleton.shape[2]

    @property
    def is_segmented(self):
        """
        Returns a 1-d boolean numpy array of whether 
        or not, frame-by-frame, the given frame was segmented

        """
        return self.segmentation_status == 's'

    def position_limits(self, dimension, measurement='skeletons'):
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

