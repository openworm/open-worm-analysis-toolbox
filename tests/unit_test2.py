# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:15:50 2015

@author: RNEL
"""

import sys, os

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
from movement_validation import user_config, NormalizedWorm
from movement_validation import WormFeatures, VideoInfo, config, utils


def test_main():
    """
    Compare Schafer-generated features with our new code's generated features

    """
    
    start_time = utils.timing_function()    
    
    # Set up the necessary file paths for file loading
    #----------------------
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    matlab_generated_file_path = os.path.join(
        base_path,'example_video_feature_file.mat')

    print(matlab_generated_file_path)

    data_file_path = os.path.join(base_path,"example_video_norm_worm.mat")

    # OPENWORM
    #----------------------
    # Load the normalized worm from file
    nw = NormalizedWorm.from_schafer_file_factory(data_file_path)

    # The frame rate is somewhere in the video info. Ideally this would all
    # come from the video parser eventually
    vi = VideoInfo('Example Video File', config.FPS)

    # Generate the OpenWorm movement validation repo version of the features
    openworm_features = WormFeatures(nw, vi)

    # SCHAFER LAB
    #----------------------
    # Load the Matlab codes generated features from disk
    matlab_worm_features = WormFeatures.from_disk(matlab_generated_file_path)

    # COMPARISON
    #----------------------
    # Show the results of the comparison
    print("\nComparison of computed features to those computed with "
          "old Matlab code")

    print("Locomotion: " + 
        str(matlab_worm_features.locomotion == openworm_features.locomotion))

    print("Posture: " +
        str(matlab_worm_features.posture == openworm_features.posture))

    print("Morphology: " +
        str(matlab_worm_features.morphology == openworm_features.morphology))

    print("Path: " +
        str(matlab_worm_features.path == openworm_features.path))

    print("\nDone validating features")

    print("Time elapsed: %.2f seconds" % 
          (utils.timing_function() - start_time))

if __name__ == '__main__':
    #start_time = utils.timing_function()
    test_main()
    #print("Time elapsed: %.2f seconds" % 
          (utils.timing_function() - start_time))