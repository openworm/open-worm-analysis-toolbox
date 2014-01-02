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
import getpass
import warnings

import wormpy

#def main():
"""
  This is an example illustrating use of the classes in the wormpy module.
  We load the skeleton and other basic data from a worm HDF5 file,
  optionally animate it using matplotlib, and also    
  re-create the features information by deriving them from the basic data.
"""
  
  # create a normalized worm from a hardcoded example location
  #print("let's do this...")
  #normalized_worm = example_nw()
  #wp = wormpy.WormPlotter(normalized_worm)

  # AT THIS POINT WE COULD ANIMATE THE WORM'S SKELETON IF WE WANTED:
  #normalized_worm.interpolate_dropped_frames()  
  #normalized_worm.animate()  
  #normalized_worm.save_to_mp4("worm_animation.mp4")
  
  # NOTE: The warning that appears comes from nanfunctions.py, because 
  # we are sometimes taking the mean and std dev of all-NaN angle arrays.
  # The mean and std_dev in these cases is set to NaN, which seems like 
  # correct behaviour to me.  So we can safely ignore this warning.  
#  worm_features = None
  #with warnings.catch_warnings():
  #  warnings.simplefilter("ignore")
    # From the basic information in normalized_worm,
    # create an instance of WormFeatures, which contains all our features data.
  #  worm_features = wormpy.WormFeatures(normalized_worm)
  
  
  
  #wp.save('test_sub.mp4')
  #plt.show()
  
  
  


def get_user_data_path():
  if(getpass.getuser() == 'Michael'):
    # michael's computer at home
    user_data_path = "C:\\Users\\Michael\\Dropbox\\"
  elif(getpass.getuser() == 'mcurrie'):
    # michael's computer at work
    user_data_path = "C:\\Backup\\Dropbox\\"    
  else:
    # if it's not Michael, assume it's Jim
    if(os.name == 'nt'): 
      # Jim is using Windows
      user_data_path = "F:\\"
    else:
      # otherwise, Jim is probably using his Mac
      user_data_path = "//Users//jameshokanson//Dropbox"
  
  return user_data_path  
  


def example_from_HDF5(base_path = None):
  # If no folder was specified for the worm, use the
  # current working directory
  if(base_path == None):
    base_path = get_user_data_path()
  
  # DEBUG: hardcoded for now.
  worm_file_path = os.path.join(base_path, 
                               "worm_data\\example_feature_files\\" +
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
  

def example_real_worm_pipeline(data_file_path, eigen_worm_file_path, other_data_file_path):
  """
    This depicts an example of how the data would flow from the Schafer real
    worm data to the features calculation and plotting
    
    At two places, we verify that our figures are the same as the 
    Schafer figures
      
  """
  
  snw_blocks = wormpy.SchaferNormalizedWormBlocks(data_file_path, eigen_worm_file_path)
  snw = snw_blocks.stitch()
  type(snw)
  # *** returns <class 'SchaferNormalizedWorm'>
  
  # NormalizedWorm can load either:
  #  --> a 'VirtualWorm' file (wrapped in a class) or
  #  --> a 'Schafer' file (wrapped in a class)
  nw = wormpy.NormalizedWorm('Schafer', snw)
  
  nw.compare_with_schafer(snw)
  #*** returns True, hopefully!
  
  wf = wormpy.WormFeatures(nw)
  
  sef = wormpy.SchaferExperimentFile(other_data_file_path)
  
  wf.compare_with_schafer(sef)
  #*** returns True, hopefully!
  
  wp = wormpy.WormPlotter(wf)
  
  wp.show()  # show the plot


def example_virtual_worm_pipeline(data_file_path):
  """
    This depicts an example of how the data would flow from the virtual worm
    to the features calculation and plotting
    
    This 'virtual' pipeline is simpler because there are no blocks to stitch
    and also we don't have to verify that our figures are the same as
    the Schafer figures
    
  """

  vw = wormpy.BasicWormData(data_file_path)
  
  # NormalizedWorm can load either:
  #  --> a 'VirtualWorm' file (wrapped in a class) or
  #  --> a 'Schafer' file (wrapped in a class)
  nw = wormpy.NormalizedWorm('VirtualWorm', vw)
  
  wf = wormpy.WormFeatures(nw)
  
  wp = wormpy.WormPlotter(wf)
  
  wp.show()
  

#if(__name__ == '__main__'):
#  main()
  
  
  
  
  
# create a normalized worm from a hardcoded example location
print("let's do this...")
nw = example_nw()
wp = wormpy.WormPlotter(nw)
