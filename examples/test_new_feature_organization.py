# -*- coding: utf-8 -*-
"""
This code is testing moving features from a nested structure to being
iterated over.

Status: Still in development (by Jim)

"""
import sys, os

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import movement_validation as mv
from movement_validation import utils

def main():
    """
    Compare Schafer-generated features with our new code's generated features

    """
    # Set up the necessary file paths for file loading
    #----------------------
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    matlab_generated_file_path = os.path.join(
        base_path,'example_video_feature_file.mat')
    data_file_path = os.path.join(base_path,"example_video_norm_worm.mat")

    # OPENWORM
    #----------------------
    # Load the normalized worm from file
    nw = mv.NormalizedWorm.from_schafer_file_factory(data_file_path)

    print('Loading example features from disk')
    matlab_worm_features = mv.WormFeaturesDos.from_schafer_file(matlab_generated_file_path)

    # Generate the OpenWorm movement validation repo version of the features
    print('Computing example features from normalized worm')
    openworm_features = mv.WormFeaturesDos(nw)
    
    all_features = matlab_worm_features.features
    for key in all_features:
        cur_feature = all_features[key]
        #Currently we are including temporary features which don't exist
        #when loading from disk
        if cur_feature is not None:
            other_feature = openworm_features.get_feature(cur_feature.name)
            print(cur_feature.name)
            is_same = cur_feature == other_feature
            if not is_same:
                import pdb
                pdb.set_trace()
     
#    # SCHAFER LAB
#    #----------------------
#    # Load the Matlab codes generated features from disk
#    matlab_worm_features = \
#        mv.WormFeatures.from_disk(matlab_generated_file_path)
#
#    # COMPARISON
#    #----------------------
#    # Show the results of the comparison
#    print("\nComparison of computed features to those computed with "
#          "old Matlab code")
#
#    print("Locomotion: " + 
#        str(matlab_worm_features.locomotion == openworm_features.locomotion))
#
#    print("Posture: " +
#        str(matlab_worm_features.posture == openworm_features.posture))
#
#    print("Morphology: " +
#        str(matlab_worm_features.morphology == openworm_features.morphology))
#
#    print("Path: " +
#        str(matlab_worm_features.path == openworm_features.path))

    print("\nDone validating features")


if __name__ == '__main__':
    start_time = mv.utils.timing_function()
    main()
    print("Time elapsed: %.2f seconds" % 
          (mv.utils.timing_function() - start_time))
    
