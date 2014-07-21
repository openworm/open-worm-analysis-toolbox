# -*- coding: utf-8 -*-
"""
  This module shows an example of plotting a worm's features.

"""

import os
import matplotlib.pyplot as plt

import src
from src import user_config
from src import utils


def example_WormExperimentFile():
    """
      Returns an example instance of WormExperimentFile, using the file
      paths specified in user_config.py

    """

    worm_file_path = os.path.join(user_config.DROPBOX_PATH,
                                  user_config.WORM_FILE_PATH)

    w = src.WormExperimentFile()
    w.load_HDF5_data(worm_file_path)

    return w


def example_nw():
    """
      This function creates a normalized worm from a hardcoded file location

    """

    # Let's take one example worm from our user_config.py file
    norm_folder = os.path.join(user_config.DROPBOX_PATH,
                               user_config.NORMALIZED_WORM_PATH)

    data_file_path = os.path.join(os.path.abspath(norm_folder),
                                  "norm_obj.mat")

    eigen_worm_file_path = os.path.join(os.path.abspath(norm_folder),
                                        "masterEigenWorms_N2.mat")

    # Create our example instance by passing the two file locations
    normalized_worm = src.NormalizedWorm(data_file_path,
                                            eigen_worm_file_path)

    return normalized_worm


def example_real_worm_pipeline(data_file_path,
                               eigen_worm_file_path,
                               other_data_file_path):
    """
      This depicts an example of how the data would flow from the Schafer real
      worm data to the features calculation and plotting

      At two places, we verify that our figures are the same as the 
      Schafer figures

    """

    snw_blocks = src.SchaferNormalizedWormBlocks(data_file_path,
                                                    eigen_worm_file_path)
    snw = snw_blocks.stitch()
    type(snw)
    # *** returns <class 'SchaferNormalizedWorm'>

    # NormalizedWorm can load either:
    #  --> a 'VirtualWorm' file (wrapped in a class) or
    #  --> a 'Schafer' file (wrapped in a class)
    nw = src.NormalizedWorm('Schafer', snw)

    nw.compare_with_schafer(snw)
    #*** returns True, hopefully!

    wf = src.WormFeatures(nw)

    sef = src.SchaferExperimentFile(other_data_file_path)

    wf.compare_with_schafer(sef)
    #*** returns True, hopefully!

    wp = src.WormPlotter(wf)

    wp.show()  # show the plot


def example_virtual_worm_pipeline(data_file_path):
    """
      This depicts an example of how the data would flow from the virtual worm
      to the features calculation and plotting

      This 'virtual' pipeline is simpler because there are no blocks to stitch
      and also we don't have to verify that our figures are the same as
      the Schafer figures

    """

    vw = src.BasicWormData(data_file_path)

    # NormalizedWorm can load either:
    #  --> a 'VirtualWorm' file (wrapped in a class) or
    #  --> a 'Schafer' file (wrapped in a class)
    nw = src.NormalizedWorm('VirtualWorm', vw)

    wf = src.WormFeatures(nw)

    wp = src.WormPlotter(wf)

    wp.show()


"""
  We load the skeleton and other basic data from a worm HDF5 file,
  optionally animate it using matplotlib, and also    
  re-create the features information by deriving them from the basic data.

"""
def main():
    # Create a normalized worm from a hardcoded example location

    #-------------------------------------------------------------------
    nw = example_nw()  # src.NormalizedWorm

    # From the basic information in normalized_worm,
    # create an instance of WormFeatures, which contains all our features data.
    wf = src.WormFeatures(nw)

    # Plotting demonstration
    # src.plot_frame_codes(nw)
    # plt.tight_layout()

    # I just saved a plaintext file with the motioncodes extracted from
    # the features result file, by viewing the results file using HDFView
    #motion_codes = np.genfromtxt('motion_codes.txt', delimiter='\n')
    #wp = src.WormPlotter(nw, motion_codes, interactive=False)
    wp = src.WormPlotter(nw, interactive=False)
    wp.show()

    # At this point we could save the plot to a file:
    # wp.save('test_sub.mp4')

    #utils.write_to_CSV({'mode': wf.locomotion.motion_mode, 'midbody speed':wf.locomotion.velocity['midbody']['speed']}, 'michael_latest')


if __name__ == '__main__':
    main()
