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
    

    example_input_data = ExampleInput(base_path)
    eid = example_input_data
    
    
    nw = NormalizedWorm.load_matlab_data(data_file_path)


#    vc  = nw.vulva_contours[:,:,0]
#    nvc = nw.non_vulva_contours[:,:,0]
#
#    import pdb
#    pdb.set_trace()

    nw2 = NormalizedWorm(eid.all_skeletons,eid.all_vulva_contours,
                         eid.all_non_vulva_contours,eid.is_valid)
    
    
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

class ExampleInput(object):
    
    def __init__(self,base_path):
        data_file_path = os.path.join(base_path,"example_contour_and_skeleton_info.mat")  
    
        h = h5py.File(data_file_path, 'r')

        #These are all references
        all_vulva_contours_refs = h['all_vulva_contours'].value
        all_non_vulva_contours_refs = h['all_non_vulva_contours'].value
        all_skeletons_refs = h['all_skeletons'].value
                
        is_stage_movement = utils._extract_time_from_disk(h,'is_stage_movement')
        is_valid = utils._extract_time_from_disk(h,'is_valid')

        all_skeletons = []
        all_vulva_contours = []
        all_non_vulva_contours = []

        #import pdb
        for valid_frame,iFrame in zip(is_valid,range(is_valid.size)):
            if valid_frame:
                all_skeletons.append(h[all_skeletons_refs[iFrame][0]].value) 
                all_vulva_contours.append(h[all_vulva_contours_refs[iFrame][0]].value)
                all_non_vulva_contours.append(h[all_non_vulva_contours_refs[iFrame][0]].value)
            else:
                all_skeletons.append([]) 
                all_vulva_contours.append([]) 
                all_non_vulva_contours.append([])
                
            
            #pdb.set_trace()           
                
        self.is_stage_movement = is_stage_movement
        self.is_valid = is_valid
        self.all_skeletons = all_skeletons
        self.all_vulva_contours = all_vulva_contours
        self.all_non_vulva_contours = all_non_vulva_contours 
    
        #'all_vulva_contours','all_non_vulva_contours','all_skeletons','is_stage_movement','is_valid');
    
           
    
        h.close()   
        
        
    
    def __repr__(self):
        return utils.print_object(self)


if __name__ == '__main__':
    main()
