# -*- coding: utf-8 -*-
"""
Insert description here
"""

import sys, os


sys.path.append('..') 
import movement_validation

user_config = movement_validation.user_config
NormalizedWorm = movement_validation.NormalizedWorm
VideoInfo = movement_validation.VideoInfo
WormFeatures = movement_validation.WormFeatures


def main():
    fps = 25.8398
    fpo = movement_validation.FeatureProcessingOptions(fps)
    
    # Set up the necessary file paths for file loading
    #----------------------
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    data_file_path = os.path.join(base_path,"example_video_norm_worm.mat")

    # OPENWORM
    #----------------------
    # Load the normalized worm from file
    nw = NormalizedWorm(data_file_path)

    #The frame rate is somewhere in the video info. Ideally this would all come
    #from the video parser eventually
    vi = VideoInfo('Example Video File',25.8398)

    # Generate the OpenWorm movement validation repo version of the features
    fpo.disable_feature_sections(['morphology']) 
    openworm_features = WormFeatures(nw,vi,fpo)    
    
    openworm_features.timer.summarize()
    
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    main()
