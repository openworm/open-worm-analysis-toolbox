# -*- coding: utf-8 -*-
""" wormExample.py: illustrates the use of the wormpy.experiment_file module
    @author: mcurrie
    
"""
from wormpy import experiment_file
import os


def main():
  pass
  # create an animation of our example, and save it
  #w = example()
  #w.create_animation()
  #w.save_to_mp4("worm_animation.mp4")
  
  

def example():
  """
  This can be called from the python shell, it returns
  a WormExperimentFile instance loaded with actual example
  information
  """

  # let's assume the worm file is in the same directory as this source file
  worm_file_path = os.path.join(os.path.abspath(os.getcwd()), 
                               "unc-8 (rev) on food " +
                               "R_2010_03_19__09_14_57___2___2_features.mat")

  return experiment_file.WormExperimentFile(worm_file_path)


if(__name__ == '__main__'):
  main()
  
  