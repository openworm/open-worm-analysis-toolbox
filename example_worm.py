# -*- coding: utf-8 -*-
""" example_worm.py: illustrates the use of the wormpy.WormExperimentFile
    module
    @author: mcurrie
"""

from wormpy import WormExperimentFile
import os


def main():
  pass
  # create an animation of our example, and save it
  #w = example_worm()
  #w.interpolate_dropped_frames()  
  #w.animate()  
  #w.save_to_mp4("worm_animation.mp4")
  
  #Consider only those frames of the worm that have not been dropped:
  # numpy.ma.masked_array(w.skeleton[-w.dropped_frames])
  
  #Access just the y-axis elements of the first frame:
  #np.rollaxis(w.skeleton[0], 1)[1]

def example_worm():
  """
  This can be called from the python shell, it returns
  a WormExperimentFile instance loaded with actual example
  information
  """

  # let's assume the worm file is in the same directory as this source file
  worm_file_path = os.path.join(os.path.abspath(os.getcwd()), 
                               "unc-8 (rev) on food " +
                               "R_2010_03_19__09_14_57___2___2_features.mat")

  w = WormExperimentFile.WormExperimentFile()
  w.load_worm(worm_file_path)

  return w


if(__name__ == '__main__'):
  main()
  
  