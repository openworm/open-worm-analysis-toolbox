# -*- coding: utf-8 -*-
"""
An example of plotting an animation of a worm's skeleton and contour

"""

import sys
import os
import warnings

# We must add .. to the path so that we can perform the
# import of open_worm_analysis_toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv


def main():
    """
    Load the skeleton and other basic data from a worm HDF5 file,
    optionally animate it using matplotlib, and also
    create the features information by deriving them from the basic data, to
    annotate the animation with.

    """
    # Force warnings to be errors
    warnings.simplefilter("error")

    # Load from file a normalized worm, as calculated by Schafer Lab code
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    schafer_nw_file_path = os.path.join(base_path,
                                        "example_video_norm_worm.mat")
    nw = mv.NormalizedWorm.from_schafer_file_factory(schafer_nw_file_path)

    # Placeholder for video metadata
    nw.video_info.video_name = "Example name"

    # We need to create WormFeatures to get the motion codes
    # (telling us in each frame if the worm is moving forward, backward, etc,
    #  which is nice to have so we can annotate the plot with that info)
    #wf = mv.WormFeatures(nw)
    #motion_codes = wf.locomotion.motion_mode

    # Plot an animation of the worm and its motion codes
    wp = mv.NormalizedWormPlottable(nw)  # , motion_codes)
    wp.show()

    # At this point we could save the plot to a file:
    # wp.save('test_sub.mp4')

    # Finally, for fun, show a pie chart of how many frames were segmented
    # worm_plotter.plot_frame_codes(nw)


if __name__ == '__main__':
    main()
