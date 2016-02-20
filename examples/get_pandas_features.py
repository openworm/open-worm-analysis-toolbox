# -*- coding: utf-8 -*-
"""
Demonstrate capabilities of WormFeatures' pandas extension methods.

"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We must add .. to the path so that we can perform the
# import of open-worm-analysis-toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv


def main():
    # warnings.filterwarnings('error')
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Set up the necessary file paths for file loading
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    matlab_generated_file_path = os.path.join(
        base_path, 'example_video_feature_file.mat')
    data_file_path = os.path.join(base_path, "example_video_norm_worm.mat")

    # Load the normalized worm from file
    nw = mv.NormalizedWorm.from_schafer_file_factory(data_file_path)

    # Generate the OpenWorm version of the features
    openworm_features = mv.WormFeatures(nw)

    # Load the Schafer Lab Matlab-code-generated features from disk
    matlab_worm_features = \
        mv.WormFeatures.from_disk(matlab_generated_file_path)

    df = openworm_features.get_DataFrame()

    movement_df = openworm_features.get_movement_DataFrame()

    # investigating path dwelling.
    plt.plot(df.ix[0].data_array)
    plt.plot(np.sort(df.ix[0].data_array))
    plt.show()

    # TODO
    # Maybe it would be nice to extract a simpler event representation
    # as an "extra" feature not in the canonical 93: event_starts and
    # event_ends.
    # you can use this as your canonical NON-schafer feature.

    # TODO: output movement_df to excel; look for errors or trends
    # that way!

    #df[df.feature_type != 'movement']

    #import pdb
    # pdb.set_trace()

if __name__ == '__main__':
    start_time = mv.utils.timing_function()
    main()
    print("Time elapsed: %.2f seconds" %
          (mv.utils.timing_function() - start_time))
