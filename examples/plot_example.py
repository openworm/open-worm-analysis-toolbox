# -*- coding: utf-8 -*-
"""
  An example of plotting a worm's features.

"""

import sys, os
import warnings

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 

import movement_validation


def main():
    """
    Load the skeleton and other basic data from a worm HDF5 file,
    optionally animate it using matplotlib, and also    
    re-create the features information by deriving them from the basic data.
    
    """

    # Force warnings to be errors
    warnings.simplefilter("error")

    # Create a normalized worm from a hardcoded example location

    #-------------------------------------------------------------------
    nw = example_nw()  # movement_validation.NormalizedWorm

    # Placeholder for video metadata
    #v = movement_validation.VideoInfo(video_name="Example name", fps=25)

    # From the basic information in normalized_worm,
    # create an instance of WormFeatures, which contains all our features data.
    #wf = movement_validation.WormFeatures(nw, v)

    # Let's show a pie chart of how many frames were segmented
    movement_validation.worm_plotter.plot_frame_codes(nw)

    # I just saved a plaintext file with the motioncodes extracted from
    # the features result file, by viewing the results file using HDFView
    #motion_codes = np.genfromtxt('motion_codes.txt', delimiter='\n')
    #wp = movement_validation.WormPlotter(nw, motion_codes, interactive=False)
    wp = movement_validation.WormPlotter(nw, interactive=False)
    wp.show()

    # At this point we could save the plot to a file:
    # wp.save('test_sub.mp4')

    #movement_validation.utils.write_to_CSV({'mode': wf.locomotion.motion_mode, 'midbody speed':wf.locomotion.velocity['midbody']['speed']}, 'michael_latest')


def example_nw():
    """
    Return a normalized worm loaded from a hardcoded file location

    """
    # Let's take one example worm from our user_config.py file
    nw_folder = os.path.join(movement_validation.user_config.EXAMPLE_DATA_PATH)
    data_file_path = os.path.join(os.path.abspath(nw_folder),
                                  "example_video_norm_worm.mat")

    NormalizedWorm = movement_validation.NormalizedWorm
    # Load the NormalizedWorm into memory from the file.
    normalized_worm = NormalizedWorm.from_schafer_file_factory(data_file_path)

    return normalized_worm


if __name__ == '__main__':
    main()
