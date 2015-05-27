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
    create the features information by deriving them from the basic data, to
    annotate the animation with.
    
    """
    # Force warnings to be errors
    warnings.simplefilter("error")

    # Create a NormalizedWorm instance from a hardcoded example location
    nw = example_nw()

    # Placeholder for video metadata
    v = movement_validation.VideoInfo(video_name="Example name", fps=25)

    # We need to create WormFeatures to get the motion codes
    # (telling us in each frame if the worm is moving forward, backward, etc,
    #  which is nice to have so we can annotate the plot with that info)
    wf = movement_validation.WormFeatures(nw, v)
    motion_codes = wf.locomotion.motion_mode

    # Plot an animation of the worm and its motion codes
    wp = movement_validation.NormalizedWormPlottable(nw, motion_codes)
    wp.show()

    # At this point we could save the plot to a file:
    # wp.save('test_sub.mp4')

    # Finally, for fun, show a pie chart of how many frames were segmented
    movement_validation.worm_plotter.plot_frame_codes(nw)


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
