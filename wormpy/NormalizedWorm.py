# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:01:23 2013

@author: mcurrie
A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code path:
SegwormMatlabClasses / 
+seg_worm / @normalized_worm / normalized_worm.m
"""
import numpy as np
import scipy.io
import os
from wormpy.WormExperimentFile import WormExperimentFile

class NormalizedWorm(WormExperimentFile):
  """ NormalizedWorm inherits from WormExperimentFile, which 
      initializes the skeleton data.
      NormalizedWorm loads the eigen_worm data and much other data from
      the experiment file data

      This will be an interface class between the parsed worms and the
      feature sets that are saved to disk. The goal is to take in the
      code from normWorms and to have a well described set of properties
      for rewriting the feature code.

    PROPERTIES / METHODS FROM JIM'S MATLAB CODE:
    * first column is original name
    * second column is renamed name, if renamed.

    properties / dynamic methods:
      eigen_worms      
      
      IN data_dict:
      
      EIGENWORM_PATH         
      segmentation_status   
      frame_codes
      vulva_contours
      non_vulva_contours
      skeletons
      angles
      in_out_touches
      lengths
      widths
      head_areas
      tail_areas
      vulva_areas
      non_vulva_areas      
      
      n_frames               num_frames()
      x                      skeletons_x()
      y                      skeletons_y()
      contour_x       
      contour_y       
    
    static methods:
      getObject              load_normalized_data(self, data_path)
      createObjectFromFiles  load_normalized_blocks(self, blocks_path)
                             * this last one not actually implemented yet *
                             * since I believe it is deprecated in Jim's *
                             * code *
      

  """
  data_dict = None
  
  # shape = (7, 48)
  # NOTE: It is one less than 49 because
  #       the values are calculated from paired values, 
  #       and the # of pairs is one less than the # of samples
  eigen_worms = None

  def __init__(self, data_file_path, eigen_worm_file_path):
    """ initialize this instance by loading both the worm and 
    the eigen_worm data
    
    """
    super().__init__()    
    self.load_normalized_data(data_file_path)
    self.load_eigen_worms(eigen_worm_file_path)
    
  def load_normalized_data(self, data_file_path):
    """ Load the norm_obj.mat file into this class
      
        This is a translation of getObject from Jim's original code
    """
    
    if(not os.path.isfile(data_file_path)):
      raise Exception("Data file not found: " + data_file_path)
    else:
      #TODO consider applying a dictionary to this data
      #http://stackoverflow.com/questions/7008608
         
      self.data_file = scipy.io.loadmat(data_file_path, 
                                        # squeeze unit matrix dimensions:
                                        squeeze_me = True, 
                                        # force return numpy object array:
                                        struct_as_record = False)

      # self.data_file is a dictionary, with keys:
      # self.data_file.keys() = 
      # dict_keys(['__header__', 's', '__version__', '__globals__'])
      
      # All the action is in data_file['s'], which is a numpy.ndarray where
      # data_file['s'].dtype is an array showing how the data is structured.
      # it is structured in precisely the order specified in data_keys below

      s = self.data_file['s']

      # NOTE: These are aligned to the order in the files.
      # these will be the keys of the dictionary data_dict
      data_keys = [
                'EIGENWORM_PATH',
                # a numpy array of chars for each frame of the video
                # s = segmented
                # f = segmentation failed
                # m = stage movement
                # d = dropped frame
                # n??? - there is reference tin some old code to this type # DEBUG
                'segmentation_status',
                #  shape is (1 n), see comments in seg_worm.parsing.frame_errors
                'frame_codes',
                
                # Contour data is used with 
                # seg_worm.feature_helpers.posture.getEccentricity:
                'vulva_contours',     # shape is (49, 2, nu) integer
                'non_vulva_contours', # shape is (49, 2, n) integer
                'skeletons',          # shape is (49, 2, n) integer
                'angles',             # shape is (49, 2, n) integer
                'in_out_touches',     # shape is (49, n) integer (degrees)
                'lengths',# shape is (n) integer
                'widths',# shape is (49, n) integer
                'head_areas', # shape is (n) integer
                'tail_areas',# shape is (n) integer
                'vulva_areas',# shape is (n) integer
                'non_vulva_areas', # shape is (n) integer
                'x',
                'y']
      
      # Here I use powerful python syntax to reference data elements of s
      # dynamically through built-in method getattr
      # that is, getattr(s, x)  works syntactically just like s.x, 
      # only x is a variable, so we can do a list comprehension with it!
      # this is to build up a nice dictionary containing the data in s
      self.data_dict = {x: getattr(s, x) for x in data_keys}
      
      # our derived class, WormExperimentFile, expects skeletons to be 
      # in the shape (n, 49, 2), but data_dict['skeletons'] is in the 
      # shape (49, 2, n), so we must "roll the axis" twice.
      self.skeletons = np.rollaxis(self.data_dict['skeletons'], 2)
      
    
  def load_normalized_blocks(self, blocks_path):
    """ Processes all the MatLab data "blocks" created from the raw 
        video into one coherent set of data.  This is a translation 
        of createObjectFromFiles from Jim's original code.
        
        MICHAEL: This appears to be the old way of doing this.
        I'll hold off translating this "block" processor.  
        I think norm_obj.mat actually maps directly
        to the structure I need.
    """
    pass

  def load_eigen_worms(self, eigen_worm_file_path):
    """ load_eigen_worms takes a file path and loads the eigen_worms
        which are stored in a MatLab data file
        This is a translation of get.eigen_worms(obj) in Jim's original code
    """
    if(not os.path.isfile(eigen_worm_file_path)):
      raise Exception("Eigenworm file not found: " + eigen_worm_file_path)
    else:
      # scipy.io.loadmat returns a dictionary with variable names 
      # as keys, and loaded matrices as values
      eigen_worms_file = scipy.io.loadmat(eigen_worm_file_path)

      # TODO: turn this into a numpy array, probably
      # TODO: and possibly extract other things of value from 
      #       eigen_worms_file
      self.eigen_worms = eigen_worms_file.values()

  def n_frames(self):
    """ for backwards compatibility with Jim's code, let's define n_frames
        (it does exactly the same thing as num_frames)
    """
    return self.num_frames()

  def contour_x(self):
    """ 
      Return the approximate worm contour, derived from data
      NOTE: The first and last points are duplicates, so we omit
            those on the second set. We also reverse the contour so that
            it encompasses an "out and back" contour
    """
    pass    
    # TODO    
    #return squeeze([obj.vulva_contours(:,1,:); obj.non_vulva_contours(end-1:-1:2,1,:);]);

  def contour_y(self):
    pass
    # TODO    
    #return squeeze([obj.vulva_contours(:,1,:); obj.non_vulva_contours(end-1:-1:2,1,:);]);




