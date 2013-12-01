# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:48:17 2013

@author: Michael Currie
A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code path:
SegwormMatlabClasses / 
+seg_worm / @feature_calculator / get_features_rewritten.m

Here main() simply illustrates the use of classes in the wormpy module.
"""

import os
import wormpy

def main():
  """
    This is an example illustrating use of the classes in the wormpy module.
    We load the skeleton and other basic data from a worm HDF5 file,
    optionally animate it using matplotlib, and also    
    re-create the features information by deriving them from the basic data.
  """
  # DEFAULT SETTING
  NORM_PATH = None  
  # JIM'S TESTING CODE
  # on PC:
  #NORM_PATH = 'F:\worm_data\segworm_data\video\testing_with_GUI\.data\mec-4 (u253) off food x_2010_04_21__17_19_20__1_seg\normalized';
  # on MAC:
  #NORM_PATH = '/Users/jameshokanson/Dropbox/worm_data/video/testing_with_GUI/.data/mec-4 (u253) off food x_2010_04_21__17_19_20__1_seg/normalized'
  
  normalized_worm = create_example_normalized_worm(NORM_PATH)
  
  # AT THIS POINT WE COULD ANIMATE THE WORM IF WE WANTED:
  #normalized_worm.interpolate_dropped_frames()  
  #normalized_worm.animate()  
  #normalized_worm.save_to_mp4("worm_animation.mp4")
  
  #  From the basic information in normalized_worm,
  #  create an instance of WormFeatures, which contains all our features data.
  worm_features = get_features(normalized_worm)
  
  # TODO: do something more useful with the output here than just print...
  print(worm_features)



def get_features(normalized_worm):
  """ 
    INPUT: normalized_worm, an instance of wormpy.NormalizedWorm
    OUTPUT: an instance of wormpy.WormFeatures
    From the basic information in normalized_worm,
    create an instance of WormFeatures, which contains all our features data.
  """
  worm_features = wormpy.WormFeatures(normalized_worm)
  
  return worm_features


def example_from_HDF5(base_path = None):
  # If no folder was specified for the worm, use the 
  # current working directory
  if(base_path == None):
    base_path = os.path.abspath(os.getcwd())
  
  # DEBUG: hardcoded for now.
  worm_file_path = os.path.join(base_path, 
                               "unc-8 (rev) on food " +
                               "R_2010_03_19__09_14_57___2___2_features.mat")

  w = wormpy.WormExperimentFile()
  w.load_HDF5_data(worm_file_path)

  return 

def create_example_normalized_worm(norm_folder = None):
  """
    This function takes a path, and loads a worm HDF5 experiment file,
    only looking at the skeleton and other basic information.
  """


  # If no folder was specified for the worm, use the 
  # current working directory
  # FIX norm_folder so it's pointing to 
  # C:\Users\Michael\Dropbox\worm_data\video\testing_with_GUI\.data\mec-4 (u253) off food x_2010_04_21__17_19_20__1_seg\normalized
  # and fix eigen_worm_file_path to be also eigen_worm_path and be
  # C:\Users\Michael\Dropbox\worm_data
  
  if(norm_folder == None):
    norm_folder = os.path.abspath(os.getcwd())
  
  eigen_worm_file_path = os.path.join(norm_folder, 
                               "masterEigenWorms_N2.mat")

  # create our example instantiation by passing the two file locations
  normalized_worm = wormpy.NormalizedWorm(worm_file_path,
                                          eigen_worm_file_path)
  
  return normalized_worm
  

if(__name__ == '__main__'):
  main()