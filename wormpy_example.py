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
import getpass

def main():
  """
    This is an example illustrating use of the classes in the wormpy module.
    We load the skeleton and other basic data from a worm HDF5 file,
    optionally animate it using matplotlib, and also    
    re-create the features information by deriving them from the basic data.
  """
  
  # create a normalized worm from a hardcoded example location
  normalized_worm = example_nw()
  
  # AT THIS POINT WE COULD ANIMATE THE WORM'S SKELETON IF WE WANTED:
  #normalized_worm.interpolate_dropped_frames()  
  #normalized_worm.animate()  
  #normalized_worm.save_to_mp4("worm_animation.mp4")
  
  #  From the basic information in normalized_worm,
  #  create an instance of WormFeatures, which contains all our features data.
  worm_features = get_features(normalized_worm)
  
  return worm_features



def get_user_data_path():
  if(getpass.getuser() == 'Michael'):
    # michael's computer at home
    NORM_PATH = "C:\\Users\\Michael\\Dropbox\\"
  elif(getpass.getuser() == 'mcurrie'):
    # michael's computer at work
    NORM_PATH = "C:\\Backup\\Dropbox\\"    
  else:
    # if it's not Michael, assume it's Jim
    if(os.name == 'nt'): 
      # Jim is using Windows
      NORM_PATH = "F:\\"
    else:
      # otherwise, Jim is probably using his Mac
      NORM_PATH = "//Users//jameshokanson//Dropbox"
  
  return NORM_PATH  
  


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



def example_nw():
  """
    This function creates a normalized worm from a hardcoded file location
    
  """

  # first let's get a base path depending on whether it's Jim or Michael
  norm_folder = get_user_data_path()
  
  # let's hardcode one example worm
  norm_folder = os.path.join(norm_folder, 
                           "worm_data\\video\\testing_with_GUI\\.data\\" +
                           "mec-4 (u253) off food " +
                           "x_2010_04_21__17_19_20__1_seg\\normalized")
  
  data_file_path = os.path.join(os.path.abspath(norm_folder),
                                "norm_obj.mat")
  
  eigen_worm_file_path = os.path.join(os.path.abspath(norm_folder),
                                      "masterEigenWorms_N2.mat")
  
  # create our example instantiation by passing the two file locations
  normalized_worm = wormpy.NormalizedWorm(data_file_path,
                                          eigen_worm_file_path)
  
  return normalized_worm
  

if(__name__ == '__main__'):
  main()
  
  
  
  
  
