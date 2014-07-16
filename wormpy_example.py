# -*- coding: utf-8 -*-
"""
  wormpy_example.py

  @authors: @JimHokanson, @MichaelCurrie

  This is an example illustrating use of the classes in the wormpy module.

"""

import os
import warnings
import wormpy
import matplotlib.pyplot as plt
import numpy as np
from wormpy import user_config
from wormpy import feature_helpers


def example_WormExperimentFile():
    """
      Returns an example instance of WormExperimentFile, using the file
      paths specified in user_config.py

    """

    worm_file_path = os.path.join(user_config.DROPBOX_PATH,
                                  user_config.WORM_FILE_PATH)

    w = wormpy.WormExperimentFile()
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
    normalized_worm = wormpy.NormalizedWorm(data_file_path,
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

    snw_blocks = wormpy.SchaferNormalizedWormBlocks(data_file_path,
                                                    eigen_worm_file_path)
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


"""
  We load the skeleton and other basic data from a worm HDF5 file,
  optionally animate it using matplotlib, and also    
  re-create the features information by deriving them from the basic data.

"""


def dontRunMeAutomagically():
    # Code for running things as we work through translating code:
    # ------------------------------------------------------------------
    # These lines can be evaluated and run by selection

    import wormpy_example as we
    nw = we.example_nw()

    # Temporary for directly accessing features
    from wormpy.WormFeatures import WormPath as wp

    temp = wp(nw)


def main():
    # Create a normalized worm from a hardcoded example location

    #-------------------------------------------------------------------
    nw = example_nw()  # wormpy.NormalizedWorm

    # From the basic information in normalized_worm,
    # create an instance of WormFeatures, which contains all our features data.
    wf = wormpy.WormFeatures(nw)

    # Plotting demonstration
    # wormpy.plot_frame_codes(nw)
    # plt.tight_layout()

    # I just saved a plaintext file with the motioncodes extracted from
    # the features result file, by viewing the results file using HDFView
    #motion_codes = np.genfromtxt('motion_codes.txt', delimiter='\n')
    #wp = wormpy.WormPlotter(nw, motion_codes, interactive=False)
    wp = wormpy.WormPlotter(nw, interactive=False)
    wp.show()

    # At this point we could save the plot to a file:
    # wp.save('test_sub.mp4')

    #feature_helpers.write_to_CSV({'mode': wf.locomotion.motion_mode, 'midbody speed':wf.locomotion.velocity['midbody']['speed']}, 'michael_latest')


if __name__ == '__main__':
    main()
