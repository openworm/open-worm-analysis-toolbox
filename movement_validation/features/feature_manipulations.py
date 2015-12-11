# -*- coding: utf-8 -*-
"""



"""

import numpy as np

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

    import pdb
    pdb.set_trace()    
    
    motion_modes = old_features.get_feature('locomotion.motion_mode').value    
    
    num_frames = len(motion_modes)
    
    indices_use_mask = {}
    indices_use_mask["all"]      = np.ones(num_frames, dtype=bool)
    indices_use_mask["forward"]  = motion_modes == 1
    indices_use_mask["backward"] = motion_modes == -1
    indices_use_mask["paused"]   = motion_modes == 0

    # NOTE: motion types refers to the motion of the worm's midbody
    motion_types = ['all', 'forward', 'paused', 'backward']
    data_types = ['all', 'absolute', 'positive', 'negative']
    


    for cur_spec in specs:
        cur_data = cur_spec.get_data(worm_features)

        good_data_mask = ~utils.get_non_numeric_mask(cur_data).flatten()
        
        # Now let's create 16 histograms, for each element of
        # (motion_types x data_types)
        
        for cur_motion_type in motion_types:
            if (good_data_mask.size != 
                                indices_use_mask[cur_motion_type].size):
                # DEBUG
                #import pdb
                #pdb.set_trace()
            
                assert(good_data_mask.size ==
                       indices_use_mask[cur_motion_type].size)

            cur_mask = indices_use_mask[cur_motion_type] & good_data_mask
            assert(isinstance(cur_data, np.ndarray))
            assert(isinstance(cur_mask, np.ndarray))
            assert(cur_data.size == cur_mask.size)

            temp_data = cur_data[cur_mask]

            # Create the histogram for the case where we consider all
            # numeric data                
            all_hist = Histogram.create_histogram(temp_data,
                                             cur_spec, 
                                             'motion',
                                             cur_motion_type, 
                                             data_types[0])

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
                                    
                # TODO: To get a speed-up, we could avoid reliance on 
                # create_histogram.  Instead, we could take the 
                # positive and negative aspects of the object 
                # that included all data. - @JimHokanson
                # (see the SegWormMatlabClasses version of this to see
                #  how this could be done)
                
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
        
        return movement_histograms



    
    pass

