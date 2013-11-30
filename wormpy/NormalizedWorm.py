# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:01:23 2013

@author: mcurrie
"""
import scipy.io
from wormpy.WormExperimentFile import WormExperimentFile

class NormalizedWorm(WormExperimentFile):
  # shape = (7, 48)
  # NOTE: It is one less than 49 because
  #       the values are calculated from paired values, 
  #       and the # of pairs is one less than the # of samples
  eigen_worms = None

  # a numpy array of chars for each frame of the video
  # s = segmented
  # f = segmentation failed
  # m = stage movement
  # d = dropped frame
  # n??? - there is reference tin some old code to this type # DEBUG
  segmentation_status = None
  # TODO
  frame_codes = None
  # TODO
  vulva_contours = None
  # TODO
  non_vulva_contours = None
  # skeletons (INHERITED)
  angles = None
  in_out_touches = None
  lengths = None
  widths = None
  head_areas = None
  tail_areas = None
  vulva_areas = None
  non_vulva_areas = None
  vental_mode = None

  def __init__(self):
    super().__init__()

  def load_eigen_worms(self, eigen_worm_file_path):
    if(self.eigen_worms != None):
      raise Exception("eigen_worms already loaded")
    else:
      # scipy.io.loadmat returns a dictionary with variable names 
      # as keys, and loaded matrices as values
      eigen_worms_file = scipy.io.loadmat(eigen_worm_file_path)

      # TODO: turn this into a numpy array, probably
      # TODO: and possibly extract other things of value from 
      #       eigen_worms_file
      self.eigen_worms = eigen_worms_file.values()

  def n_frames(self):
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







