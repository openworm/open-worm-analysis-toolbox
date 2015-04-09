# -*- coding: utf-8 -*-
"""
This module defines the NormalizedWorm class

"""

import numpy as np
import scipy.io

import warnings
import os
import inspect
import h5py
import simplejson as json

from . import config
from . import utils

class NormalizedWorm(object):
    """ 
    NormalizedWorm encapsulates the normalized measures data, loaded
    from the two files, one for the eigenworm data and the other for 
    the rest.

    This will be an intermediate representation, between the parsed,
    normalized worms, and the "feature" sets. The goal is to take in the
    code from normWorms and to have a well described set of properties
    for rewriting the feature code.

    PROPERTIES / METHODS FROM JIM'S MATLAB CODE:
    * first column is original name
    * second column is renamed name, if renamed.

    Properties:
    -----------
      segmentation_status   
      frame_codes
      vulva_contours        49 x 2 x n_frames
      non_vulva_contours    49 x 2 x n_frames
      skeletons
      angles
      in_out_touches
      lengths
      widths
      head_areas
      tail_areas
      vulva_areas
      non_vulva_areas      

      n_frames
      x - how does this differ from skeleton_x???
      y
      contour_x
      contour_y
      skeleton_x

    static methods:
      getObject              load_normalized_data(self, data_path)

    """


    """

    Notes
    ----------------------
    Originally translated from seg_worm.skeleton_indices
  
  Used in: (list is not comprehensive)
  --------------------------------------------------------
  - posture bends
  - posture directions
  
  NOTE: These are hardcoded for now. I didn't find much use in trying
  to make this dynamic based on some maximum value.
  
  Typical Usage:
  --------------------------------------------------------
  SI = seg_worm.skeleton_indices;

  """
    # The normalized worm contains precisely 49 points per frame.  Here
    # we list in a dictionary various partitions of the worm.
    worm_partitions = None
    # this stores a dictionary of various ways of organizing the partitions
    worm_parititon_subsets = None

    data_dict = None  # A dictionary of all data in norm_obj.mat


    def __init__(self, data_file_path=None):
        """ 
        Initialize this instance by loading both the worm data
        
        Parameters
        ---------------------------------------
        data_file_path: string  (optional)
          if None is specified, no data is loaded      

        """
        #TODO: Michael, why are these optional????
        if data_file_path:
            self.load_normalized_data(data_file_path)


        # These are RANGE values, so the last value is not inclusive
        self.worm_partitions = {'head': (0, 8),
                                'neck': (8, 16),
                                'midbody':  (16, 33),
                                'old_midbody_velocity': (20, 29),
                                'hips':  (33, 41),
                                'tail': (41, 49),
                                # refinements of ['head']
                                'head_tip': (0, 4),
                                'head_base': (4, 8),    # ""
                                # refinements of ['tail']
                                'tail_base': (40, 45),
                                'tail_tip': (45, 49),   # ""
                                'all': (0, 49),
                                # neck, midbody, and hips
                                'body': (8, 41)}

        self.worm_partition_subsets = {'normal': ('head', 'neck', 'midbody', 'hips', 'tail'),
                                       'first_third': ('head', 'neck'),
                                       'second_third': ('midbody',),
                                       'last_third': ('hips', 'tail'),
                                       'all': ('all',)}

        # DEBUG: (Note from @MichaelCurrie:)
        # This should be set by the normalized worm file, since each
        # worm subjected to an experiment is manually examined to find the
        # vulva so the ventral mode can be determined.  Here we just set
        # the ventral mode to a default value as a stopgap measure
        self.ventral_mode = config.DEFAULT_VENTRAL_MODE


    def load_from_JSON(self, data_file_path):
        """
        Load from JSON.  The format is:
        
        contour  (mx2xn numpy array, where m is the number of contour points)
        head     (2xn numpy array, must be a point in the contour)
        tail     (2xn numpy array, must be a point in the contour)
        
        OPTIONAL:
        segmentation_status  (if None, will assume all frames are "s")
        frame_codes          (if None, will assume all frames are 1)
        skeleton             (if None, this will be calculated in the next step)

        """
        pass
    
    def save_to_JSON(self, data_file_path):
        """
        Save to JSON

        """
        data = json.dumps(self.contour_x.tolist())
        with open(data_file_path, 'w') as outfile:
            json.dump(data, outfile, ensure_ascii=False)

        print("saved to JSON")
        pass


    @classmethod
    def load_from_matlab_data(self):
        pass
        #TODO: Merge the constructor and load_normalized_data into here...


    def load_normalized_data(self, data_file_path):
        """ 
        Load the norm_obj.mat file into this class

        Notes
        ---------------------------------------    
        Translated from getObject in SegwormMatlabClasses

        """

        if(not os.path.isfile(data_file_path)):
            raise Exception("Data file not found: " + data_file_path)
        else:
            self.data_file = scipy.io.loadmat(data_file_path,
                                              # squeeze unit matrix dimensions:
                                              squeeze_me=True,
                                              # force return numpy object
                                              # array:
                                              struct_as_record=False)

            # self.data_file is a dictionary, with keys:
            # self.data_file.keys() =
            # dict_keys(['__header__', 's', '__version__', '__globals__'])

            # All the action is in data_file['s'], which is a numpy.ndarray where
            # data_file['s'].dtype is an array showing how the data is structured.
            # it is structured in precisely the order specified in data_keys
            # below

            staging_data = self.data_file['s']

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
                # n??? - there is reference tin some old code to this
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
            # this is to build up a nice dictionary containing the data in s
            
            for key in data_keys:
                setattr(self, key, getattr(staging_data, key))
            
            #self.data_dict = {x: getattr(staging_data, x) for x in data_keys}

            # Let's change the string of length n to a numpy array of single
            # characters of length n, to be consistent with the other data
            # structures
            self.segmentation_status = np.array(list(self.segmentation_status))

            self.load_frame_code_descriptions()


    def load_frame_code_descriptions(self):
        """
        Load the frame_codes descriptions, which are stored in a .csv file

        """
        # Here we assume the CSV is located in the same directory 
        # as this current module's directory.
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'frame_codes.csv')
        f = open(file_path, 'r')

        self.frame_codes_descriptions = []

        for line in f:
            # split along ';' but ignore any newlines or quotes
            a = line.replace("\n", "").replace("'", "").split(';')
            # the actual frame codes (the first entry on each line)
            # can be treated as integers
            a[0] = int(a[0])
            self.frame_codes_descriptions.append(a)

        f.close()


    def get_partition_subset(self, partition_type):
        """ 
        There are various ways of partitioning the worm's 49 points.
        this method returns a subset of the worm partition dictionary

        TODO: This method still is not obvious to me. Also, we should move
        these things to a separate class.

        Parameters
        ---------------------------------------
        partition_type: string
          e.g. 'head'

        Usage
        ---------------------------------------
        For example, to see the mean of the head and the mean of the neck, 
        use the partition subset, 'first_third', like this:

        nw = NormalizedWorm(....)

        width_dict = {k: np.mean(nw.get_partition(k), 0) for k in ('head', 'neck')}

        OR, using self.worm_partition_subsets,

        s = nw.get_paritition_subset('first_third')
        # i.e. s = {'head':(0,8), 'neck':(8,16)}

        width_dict = {k: np.mean(nw.get_partition(k), 0) for k in s.keys()}

        Notes
        ---------------------------------------    
        Translated from get.ALL_NORMAL_INDICES in SegwormMatlabClasses / 
        +seg_worm / @skeleton_indices / skeleton_indices.m

        """

        # parition_type is assumed to be a key for the dictionary
        # worm_partition_subsets
        p = self.worm_partition_subsets[partition_type]

        # return only the subset of partitions contained in the particular
        # subset of interest, p.
        return {k: self.worm_partitions[k] for k in p}


    def get_subset_partition_mask(self, name):
        """
        Returns a boolean mask - for working with arrays given a partition.
        
        """
        keys = self.worm_partition_subsets[name]
        mask = np.zeros(49, dtype=bool)
        for key in keys:
            mask = mask | self.partition_mask(key)

        return mask


    def partition_mask(self, partition_key):
        """
        Returns a boolean numpy array corresponding to the partition requested.

        """
        mask = np.zeros(49, dtype=bool)
        slice_val = self.worm_partitions[partition_key]
        mask[slice(*slice_val)] = True
        return mask


    def get_partition(self, partition_key, data_key='skeletons',
                      split_spatial_dimensions=False):
        """    
        Retrieve partition of a measurement of the worm, that is, across all
        available frames but across only a subset of the 49 points.

        Parameters
        ---------------------------------------    
        partition_key: string
          The desired partition.  e.g. 'head', 'tail', etc.

          #TODO: This should be documented better 

          INPUT: a partition key, and an optional data key.
            If split_spatial_dimensions is True, the partition is returned 
            separated into x and y
          OUTPUT: a numpy array containing the data requested, cropped to just
                  the partition requested.
                  (so the shape might be, say, 4xn if data is 'angles')

        data_key: string  (optional)
          The desired measurement (default is 'skeletons')

        split_spatial_dimensions: bool    (optional)
          If True, the partition is returned separated into x and y

        Returns
        ---------------------------------------    
        A numpy array containing the data requested, cropped to just
        the partition requested.
        (so the shape might be, say, 4xn if data is 'angles')

        Notes
        ---------------------------------------    
        Translated from get.ALL_NORMAL_INDICES in SegwormMatlabClasses / 
        +seg_worm / @skeleton_indices / skeleton_indices.m

        """
        # We use numpy.split to split a data_dict element into three, cleaved
        # first by the first entry in the duple worm_partitions[partition_key],
        # and second by the second entry in that duple.

        # Taking the second element of the resulting list of arrays, i.e. [1],
        # gives the partitioned component we were looking for.
        partition = np.split(getattr(self,data_key),
                             self.worm_partitions[partition_key])[1]

        if(split_spatial_dimensions):
            return partition[:, 0, :], partition[:, 1,:]
        else:
            return partition


    def rotate(self, theta_d):
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
        s = self.skeletons
        with warnings.catch_warnings():
            temp = np.nanmean(s, 0, keepdims=False)

        return temp


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
        return np.arctan(v[1, :]/v[0,:])*(180/np.pi)


    def translate_to_centre(self):
        """ 
        Return a NormalizedWorm instance with each frame moved so the 
        centroid of the worm is 0,0

        Returns
        ---------------------------------------    
        A NormalizedWorm instance with the above properties.

        """
        s = self.skeletons
        s_mean = np.ones(np.shape(s)) * np.nanmean(s, 0, keepdims=False)

        #nw2 = NormalizedWorm()

        # TODO
        return s - s_mean


    def rotate_and_translate(self):
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

        skeletons_centred = self.translate_to_centre()
        orientation = self.angle

        a = -orientation * (np.pi / 180)

        rot_matrix = np.array([[np.cos(a), -np.sin(a)],
                               [np.sin(a),  np.cos(a)]])

        # we need the x,y listed in the first dimension
        s1 = np.rollaxis(skeletons_centred, 1)

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
        return self.skeletons.shape[2]

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
        d = getattr(self,measurement)
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
        vc = self.vulva_contours
        nvc = self.non_vulva_contours
        return np.concatenate((vc[:, 0, :], nvc[-2:0:-1, 0,:]))    

    @property
    def contour_y(self):
        vc = self.vulva_contours
        nvc = self.non_vulva_contours
        return np.concatenate((vc[:, 1, :], nvc[-2:0:-1, 1,:]))    

    @property
    def skeleton_x(self):
        return self.skeletons[:, 0, :]

    @property
    def skeleton_y(self):
        return self.skeletons[:, 1, :]

    def __repr__(self):
        #TODO: This omits the properties above ...
        return utils.print_object(self)


class SkeletonPartitions(object):
    #TODO: This needs to be implemented
    pass