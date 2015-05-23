# -*- coding: utf-8 -*-
"""
Insert description here
"""

import sys, os
import h5py

sys.path.append('..') 
import movement_validation

from movement_validation import utils
#from movement_validation import pre_features

user_config = movement_validation.user_config
NormalizedWorm = movement_validation.NormalizedWorm
VideoInfo = movement_validation.VideoInfo
WormFeatures = movement_validation.WormFeatures
BasicWorm = movement_validation.BasicWorm

def main():
    fps = 25.8398
    fpo = movement_validation.FeatureProcessingOptions(fps)
    
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    nw_file_path = os.path.join(base_path,"example_video_norm_worm.mat")

    nw = NormalizedWorm.from_schafer_file_factory(nw_file_path)

    
    
    bw_file_path = os.path.join(base_path,"example_contour_and_skeleton_info.mat")  

    bw = BasicWorm.from_schafer_file_factory(bw_file_path)
        
    nw2 = NormalizedWorm.from_BasicWorm_factory(bw)
    
    import pdb
    pdb.set_trace()
    nw == nw2
    
    
    import pdb
    pdb.set_trace()
    
    #Now the goal is to go from the example_input_data to the normalized
    #worm data.
    

    min_worm = pre_features.MinimalWormSpecification()



    #segmentationMain.m
    #correctVulvaSide
    #normWormProcess


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
