# -*- coding: utf-8 -*-
"""


"""
import sys, warnings

class MinimalWormSpecification(object):
    """
    A time-series of contour and segmentation_status.  Also the vulva side.
    
    """
    def __init__(self, worm_frame_iterator):
        """
        Prepare a minimal worm specification from an iterator of segmented 
        frames
        
        Parameters
        -----------------
        worm_frame_iterator: Iterator of instances of Worm
        
        """
        attribute_list = ['contour', 'head', 'tail', 'segmentation_status', 
                          'frame_codes', 'vulva_side']

        for attribute in attribute_list:
            setattr(self, attribute) = \
                np.array([getattr(frame, attribute) for 
                          frame in worm_frame_list])


    def pre_features(self):
        """
        Calculate a time-series of contour and other additional, relatively
        simple measurements made that are used when calculating features.        
        
        Parameters
        ------------------
        minimal_worm_spec: instance of MinimalWormSpec
            The contour data, etc, from which we calculate additional basic
            information about the worm, frame-by-frame
        
        """
        warnings.warn("WormPreFeatures has not yet been implemented")        
        
        self.segmentation_status = minimal_worm_spec.segmentation_status
        self.frame_codes = minimal_worm_spec.frame_codes
        self.vulva_contours = None  # TODO
        self.non_vulva_contours = None # TODO
        self.skeleton = None # TODO
        self.angles = None # TODO 
        self.in_out_touches = None # TODO 
        self.lengths = None # TODO 
        self.widths = None # TODO 
        self.head_areas = None # TODO 
        self.tail_areas = None # TODO 
        self.vulva_areas = None # TODO 
        self.non_vulva_areas = None # TODO 
        self.x = None # TODO
        self.y = None # TODO
    
    

class WormPreFeatures(object):
    def __init__(self, minimal_worm_spec):
        pass



