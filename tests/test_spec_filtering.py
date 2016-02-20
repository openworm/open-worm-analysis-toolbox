# -*- coding: utf-8 -*-
"""
This code is testing moving features from a nested structure to being
iterated over.

Status: Still in development (by Jim)

"""
import sys
import os

# We must add .. to the path so that we can perform the
# import of movement_validation while running this as
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
    data_file_path = os.path.join(base_path, "example_video_norm_worm.mat")

    # OPENWORM
    #----------------------
    # Load the normalized worm from file
    nw = mv.NormalizedWorm.from_schafer_file_factory(data_file_path)

    specs = mv.get_feature_specs()

    #    '^locomotion\.' => string that starts with 'locomotion.'
    loco_specs = specs[specs['feature_name'].str.contains('^locomotion\.')]

    #loco_names = loco_specs['feature_name']

    # Generate the OpenWorm movement validation repo version of the features
    print('Computing example features from normalized worm')
    openworm_features = mv.WormFeatures(nw, specs=loco_specs)

    f = openworm_features.features
    feature_names = [x.name for x in f]

    # We might want to expose an interface that returns a list of features
    # that optionally allows temporary features and/or even non-requested
    # features
    d = openworm_features._features
    all_feature_names = [d[x].name for x in d]
    # all_feature_names contains morphology.length which was not requested

    # Not sure what test to run ...
    # Let's marvel at the name filtering!!!!

    print('All done with test_spec_filtering.py')

if __name__ == '__main__':
    print('RUNNING TEST ' + os.path.split(__file__)[1] + ':')
    start_time = mv.utils.timing_function()
    main()
    print("Time elapsed: %.2f seconds" %
          (mv.utils.timing_function() - start_time))
