# -*- coding: utf-8 -*-
"""
This code loads all the features for the example video and then compares
them to a pre-saved version from the old SegWorm code.

Status: Finished, although we should really compare the features in both.
i.e. do we have the same feature names in both, since the code is only
getting the features from the old version that are in the new version

"""
import sys
import os

# We must add .. to the path so that we can perform the
# import of open-worm-analysis-toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv


def main():
    """
    Compare Schafer-generated features with our new code's generated features

    """
    # Set up the necessary file paths for file loading
    #----------------------
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    matlab_generated_file_path = os.path.join(
        base_path, 'example_video_feature_file.mat')
    data_file_path = os.path.join(base_path, "example_video_norm_worm.mat")

    # OPENWORM
    #----------------------
    # Load the normalized worm from file
    nw = mv.NormalizedWorm.from_schafer_file_factory(data_file_path)

    # Generate the OpenWorm version of the features
    print('Computing example features from normalized worm')
    openworm_features = mv.WormFeatures(nw)

    # SCHAFER LAB
    #----------------------
    print('Loading example features from disk')
    matlab_worm_features = mv.WormFeatures.from_disk(
        matlab_generated_file_path)

    # COMPARISON
    #----------------------
    # TODO: I think we should add an iteration method for worm_features

    for feature in openworm_features:
        other_feature = matlab_worm_features.get_features(feature.name)
        is_same = feature == other_feature
        if not is_same:
            print('Feature mismatch: %s' % feature.name)
            #import pdb
            # pdb.set_trace()

    print("\nDone validating features")


if __name__ == '__main__':
    print('RUNNING TEST ' + os.path.split(__file__)[1] + ':')
    start_time = mv.utils.timing_function()
    main()
    print("Time elapsed: %.2f seconds" %
          (mv.utils.timing_function() - start_time))
