# -*- coding: utf-8 -*-
"""



"""

from .. import utils

import numpy as np


def _expand_movement_features(m_feature,m_masks,num_frames):
    """
    Movement features are expanded as follows:
        - if not signed, then we have 4x based on how the worm is moving\
            - all
            - forward
            - paused
            - backward
        - if signed, then we have 16x based on the features values and 
        based on how the worm is moving
        
    *All NaN values are removed
    
    """

    #feature names
    

    motion_types = ['all', 'forward', 'paused', 'backward']
    data_types = ['all', 'absolute', 'positive', 'negative']
    
    cur_spec = m_feature.spec
    cur_data = m_feature.value

    good_data_mask = ~utils.get_non_numeric_mask(cur_data).flatten()
    
    d_masks = {}
    d_masks['all'] = good_data_mask   
    d_masks["absolute"] = good_data_mask
    d_masks["positive"] = cur_data >= 0 #bad data will be false
    d_masks["negative"] = cur_data <= 0 #bad data will be false       
    
    # Now let's create 16 histograms, for each element of
    # (motion_types x data_types)
    
    new_features = []
    for cur_motion_type in motion_types:
        
        new_features.append(_create_new_movement_feature(m_feature,m_masks,d_masks,cur_motion_type,'all'))

        #JAH: At this point ...
 

        movement_histograms.append(all_hist)

        if cur_spec.is_signed:
            
            # Histogram for the data made absolute
            # TODO: This could be improved by merging results 
            #       from positive and negative - @JimHokanson
            abs_hist = Histogram.create_histogram(abs(temp_data),
                                             cur_spec,
                                             'motion',
                                             cur_motion_type,
                                             data_types[1])
                                

            
            # Histogram for just the positive data
            pos_hist  = Histogram.create_histogram(
                            temp_data[temp_data >= 0],
                            cur_spec,
                            'motion',
                            cur_motion_type,
                            data_types[2])
            
            # Histogram for just the negative data
            neg_hist  = Histogram.create_histogram(
                            temp_data[temp_data <= 0], 
                            cur_spec, 
                            'motion', 
                            cur_motion_type, 
                            data_types[3])
            
            # Append our list with these histograms
            movement_histograms.append(abs_hist)
            movement_histograms.append(pos_hist)
            movement_histograms.append(neg_hist)

def _create_new_movement_feature(feature,m_masks,d_masks,m_type,d_type):
    """
    
    """
    FEATURE_NAME_FORMAT_STR = '%s.%s_data_with_%s_movement'
    
    cur_mask = m_masks[m_type] & d_masks[d_type]
    temp_feature = feature.copy()
    temp_spec = feature.spec.copy()
    temp_spec.name = FEATURE_NAME_FORMAT_STR % (temp_spec.name,d_type,m_type)
    temp_feature.value = feature.value[cur_mask]
    temp_feature.spec = temp_spec  
    
    return temp_feature


def expand_mrc_features(old_features):
    """
    Feature Expansion:
    ------------------
    simple - no expansion
    movement 
        - if not signed, then we have 4x based on how the worm is moving\
            - all
            - forward
            - paused
            - backward
        - if signed, then we have 16x based on the features values and 
        based on how the worm is moving
    event 
        - at some point we need to filter events :/
        - When not signed, only a single value
        - If signed then 4x, then we compute all, absolute, positive, negative
        
    Outline
    -------
    Return a new set of features in which the specs have been appropriately
    modified (need to implement a deep copy)
    """

    #Motion of the the worm's body
    motion_types = ['all', 'forward', 'paused', 'backward']
    #Value that the current feature is taking on
    data_types = ['all', 'absolute', 'positive', 'negative']

    motion_modes = old_features.get_feature('locomotion.motion_mode').value    
    
    num_frames = len(motion_modes)
    
    move_mask = {}
    move_mask["all"]      = np.ones(num_frames, dtype=bool)
    move_mask["forward"]  = motion_modes == 1
    move_mask["backward"] = motion_modes == -1
    move_mask["paused"]   = motion_modes == 0


    all_features = []
    for cur_feature in old_features:
        
        cur_spec = cur_feature.spec
        
        if cur_spec.type == 'movement':
            all_features.extend(_expand_movement_features(cur_feature,move_mask,num_frames))
        elif cur_spec.type == 'simple':
            pass
        elif cur_spec.type == 'event':
            pass
        else:
            all_features.extend(cur_feature)
            
            
    #TODO: Return new features container with all_features attached
