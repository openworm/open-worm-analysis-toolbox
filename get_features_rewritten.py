# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:48:17 2013

@author: Michael Currie
A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code path:
+seg_worm / @feature_calculator / get_features_rewritten.m
"""

import os
from wormpy import WormExperimentFile
from wormpy import WormFeatures

def main():
  """
  You can run various tests from this main() function.
  """
  pass
  """
  MICHAEL'S TESTING CODE
  # create an animation of our example, and save it
  w = example_worm()
  w.interpolate_dropped_frames()  
  w.animate()  
  w.save_to_mp4("worm_animation.mp4")
  """
  
  """
  JIM'S TESTING CODE
  
  # on PC:
  NORM_PATH = 'F:\worm_data\segworm_data\video\testing_with_GUI\.data\mec-4 (u253) off food x_2010_04_21__17_19_20__1_seg\normalized';
  # on MAC:
  #NORM_PATH = '/Users/jameshokanson/Dropbox/worm_data/video/testing_with_GUI/.data/mec-4 (u253) off food x_2010_04_21__17_19_20__1_seg/normalized'
  
  get_features_rewritten(NORM_PATH)
  """
  worm = get_features_rewritten()



def get_features_rewritten(norm_folder = None):
  """
    This function takes a path, and loads a worm HDF5 experiment file,
    only looking at the skeleton and other basic information.
    From this BASIC information, contained in normalized_worm, an instance
    of WormExperimentFile, it creates an instance of WormFeatures, which
    contains all our features data.
  """

  # DEBUG: change to Jim's class later
  normalized_worm = WormExperimentFile.WormExperimentFile()

  # If no folder was specified for the worm, use the 
  # current working directory
  if(norm_folder == None):
    norm_folder = os.path.abspath(os.getcwd())
  
  # DEBUG: hardcoded for now.
  worm_file_path = os.path.join(norm_folder, 
                               "unc-8 (rev) on food " +
                               "R_2010_03_19__09_14_57___2___2_features.mat")

  normalized_worm.load_worm(worm_file_path)
  
  # use this normalized_worm to calculate features information
  # this features information will be stored in a WormFeatures instance, worm
  worm = WormFeatures.WormFeatures(normalized_worm)
  
  return worm
  
  
  

if(__name__ == '__main__'):
  main()