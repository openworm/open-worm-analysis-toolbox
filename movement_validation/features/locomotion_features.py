# -*- coding: utf-8 -*-
"""
Locomotion features

Contains Processing code for:
-----------------------------
locomotion.velocity
locomotion.motion_events

"""

import numpy as np

from .. import utils

from . import events
from . import feature_comparisons as fc
# To avoid conflicting with variables named 'velocity', we 
# import this as 'velocity_module':
from . import velocity as velocity_module 


class LocomotionVelocityElement(object):
    """
    
    This class is a simple container class for a velocity element.    
    
    Attributes
    ----------
    name : string
    speed : numpy array
    direction : numpy array
    
    See Also
    --------
    LocomotionVelocity
    
    """
    def __init__(self,name,speed,direction):
        self.name = name
        self.speed = speed
        self.direction = direction
        
    def __eq__(self, other):
        return fc.corr_value_high(self.speed,other.speed,
                                  'locomotion.velocity.' + self.name + '.speed') and \
               fc.corr_value_high(self.direction,other.direction,
                                  'locomotion.velocity.' + self.name + '.direction')
    def __repr__(self):
        return utils.print_object(self)         

    @classmethod
    def from_disk(cls, parent_ref, name):
        
        self = cls.__new__(cls)        

        self.name = name
        self.speed = utils._extract_time_from_disk(parent_ref, 'speed')
        self.direction = utils._extract_time_from_disk(parent_ref, 'direction')
        
        return self

class LocomotionVelocity(object):
    
    """
    This is for the 'velocity' locomotion feature. The helper function,
    'compute_velocity' is used elsewhere.  

    Computes the worm velocity at various parts of the worm body.
        
    Attributes
    ----------
    head_tip : LocomotionVelocityElement
    head : LocomotionVelocityElement
    midbody : LocomotionVelocityElement
    tail : LocomotionVelocityElement
    tail_tip : LocomotionVelocityElement
    
    fps : float

    """
    
    # NOTE: head_tip and tail_tip overlap head and tail, respectively, and
    #       this set of partitions does not cover the neck and hips
    attribute_keys = ['head_tip', 'head', 'midbody', 'tail', 'tail_tip']
    
    def __init__(self,features_ref,ventral_mode=0):
        """
            THIS IS OUT OF DATE
            nw : a NormalizedWorm instance

        ventral_mode: int
        The ventral side mode:
        0 = unknown
        1 = clockwise
        2 = anticlockwise
        """
        #TODO: Ventral mode needs to be handled differently
        
        nw = features_ref.nw

        timer = features_ref.timer
        timer.tic()

        all_options = features_ref.options
    
        locomotion_options = all_options.locomotion
        
        fps = features_ref.video_info.fps
    
        #We need this for the midbody_distance calculation
        self.fps = fps    
    
        data_keys = list(self.attribute_keys) #Make a copy. Data keys will 
        #correspond to sections of the skeleton. Attribute keys correspond to 
        #attributes of this class.

        if(all_options.mimic_old_behaviour):
            data_keys[2] = 'old_midbody_velocity'

        #Step 1: Compute the average body angle
        #---------------------------------------------------------
        avg_body_angle = velocity_module.get_partition_angles(nw, 
                                          partition_key='body',
                                          data_key='skeletons',
                                          head_to_tail=False)

        sample_time_values = {
            'head_tip': locomotion_options.velocity_tip_diff,
            'head':     locomotion_options.velocity_body_diff,
            'midbody':  locomotion_options.velocity_body_diff,
            'tail':     locomotion_options.velocity_body_diff,
            'tail_tip': locomotion_options.velocity_tip_diff
        }

        #Step 2: Run the compute_velocity function on the different parts of
        #the body, 
        for attribute_key, data_key in zip(self.attribute_keys, data_keys):
            x, y = nw.get_partition(data_key, 'skeletons', True)

            speed, direction = velocity_module.compute_velocity(fps, x, y,
                                            avg_body_angle,
                                            sample_time_values[attribute_key],
                                            ventral_mode)[0:2]
                                            
            setattr(self,attribute_key,LocomotionVelocityElement(attribute_key,speed,direction))                                

        timer.toc('locomotion.velocity')

    def get_midbody_distance(self):

        """
        The midbody distance is used by a lot of other functions.
        
        Returns
        -------
        numpy.array
        
        """        
        
        return abs(self.midbody.speed / self.fps)

    def __eq__(self, other):
        is_same = True
        for attribute_key in self.attribute_keys:
            is_same = getattr(self,attribute_key) == getattr(other,attribute_key)
            if not is_same:
                break

        return is_same

    def __repr__(self):
        return utils.print_object(self)

    @classmethod
    def from_disk(cls, parent_ref):
        
        self = cls.__new__(cls)

        velocity_ref = parent_ref['velocity']

        for key in self.attribute_keys:
            
            #NOTE: We'll eventually need to check for old or new data ...
            #These changes are only valid for MRC features
            if key is 'head_tip':
                old_key = 'headTip'
            elif key is 'tail_tip':
                old_key = 'tailTip'
            else:
                old_key = key
            local_ref = velocity_ref[old_key]
            setattr(self,key,
                    LocomotionVelocityElement.from_disk(local_ref,key))      
        return self

class MotionEvents(object):
    
    """
    
    Attributes
    ----------
    forward : movement_validation.features.events.EventListWithFeatures
    paused : movement_validation.features.events.EventListWithFeatures
    backward : movement_validation.features.events.EventListWithFeatures
    mode : numpy.array
        - shape num_frames
        - Values are:
            -1, backward locomotion
            0, no locomotion (the worm is paused)
            1, forward locomotion
    
    """
    
    attribute_keys = ['forward','backward','paused']
    
    def __init__(self,features_ref,midbody_speed,skeleton_lengths):
        """ 
        Calculate motion codes of the locomotion events
        
        A locomotion feature.
    
        See feature description at 
          /documentation/Yemini%20Supplemental%20Data/Locomotion.md
    
        Parameters
        ----------
        midbody_speed: numpy.array
            [n_frames]
        skeleton_lengths: numpy.array
            [n_frames]
    


    
        Notes
        -----
        Formerly +seg_worm/+features/@locomotion/getWormMotionCodes.m
    
        """

        timer = features_ref.timer
        timer.tic()

        fps = features_ref.video_info.fps        
        
        locomotion_options = features_ref.options.locomotion        
        
        # Initialize the worm speed and video frames.
        num_frames = len(midbody_speed)
    
        # Compute the midbody's "instantaneous" distance travelled at each frame,
        # distance per second / (frames per second) = distance per frame
        distance_per_frame = abs(midbody_speed / fps)
    
        #  Interpolate the missing lengths.
        skeleton_lengths = utils.interpolate_with_threshold(
                               skeleton_lengths,
                               locomotion_options.motion_codes_longest_nan_run_to_interpolate)
    
        #===================================
        # SPACE CONSTRAINTS
        # Make the speed and distance thresholds a fixed proportion of the
        # worm's length at the given frame:
        worm_speed_threshold = skeleton_lengths * locomotion_options.motion_codes_speed_threshold_pct
        worm_distance_threshold = skeleton_lengths * locomotion_options.motion_codes_distance_threshold_pct
        worm_pause_threshold = skeleton_lengths * locomotion_options.motion_codes_pause_threshold_pct
    
        # Minimum speed and distance required for movement to
        # be considered "forward"
        min_forward_speed = worm_speed_threshold
        min_forward_distance = worm_distance_threshold
    
        # Minimum speed and distance required for movement to
        # be considered "backward"
        max_backward_speed = -worm_speed_threshold
        min_backward_distance = worm_distance_threshold
    
        # Boundaries between which the movement would be considered "paused"
        min_paused_speed = -worm_pause_threshold
        max_paused_speed = worm_pause_threshold
    
        # Note that there is no maximum forward speed nor minimum backward speed.
        #TODO: This might be better as a class attribute
        frame_values = {'forward': 1, 'backward': -1, 'paused': 0}
        
        min_speeds = {'forward': min_forward_speed,
                      'backward': None,
                      'paused': min_paused_speed}
        max_speeds = {'forward': None,
                      'backward': max_backward_speed,
                      'paused': max_paused_speed}
        min_distance = {'forward': min_forward_distance,
                        'backward': min_backward_distance,
                        'paused': None}
    
        #===================================
        # TIME CONSTRAINTS
        # The minimum number of frames an event had to be taking place for
        # to be considered a legitimate event
        min_frames_threshold = \
            fps * locomotion_options.motion_codes_min_frames_threshold
        # Maximum number of contiguous contradicting frames within the event
        # before the event is considered to be over.
        max_interframes_threshold = \
            fps * locomotion_options.motion_codes_max_interframes_threshold
        
        # Start with a blank numpy array, full of NaNs:
        self._mode = np.empty(num_frames, dtype='float') * np.NaN
    
        for motion_type in frame_values:
            # We will use EventFinder to determine when the
            # event type "motion_type" occurred
            ef = events.EventFinder()
    
            # "Space and time" constraints
            ef.min_distance_threshold = min_distance[motion_type]
            ef.max_distance_threshold = None  # we are not constraining max dist
            ef.min_speed_threshold = min_speeds[motion_type]
            ef.max_speed_threshold = max_speeds[motion_type]
    
            # "Time" constraints
            ef.min_frames_threshold = min_frames_threshold
            ef.max_inter_frames_threshold = max_interframes_threshold
    
            event_list = ef.get_events(midbody_speed, distance_per_frame)

            # Take the start and stop indices and convert them to the structure
            # used in the feature files
            m_event = events.EventListWithFeatures(fps,
                                                   event_list,
                                                   distance_per_frame,
                                                   compute_distance_during_event=True)
    
            setattr(self,motion_type,m_event)      
    
            #Assign motion modes
            #------------------------------------------------------------------
            event_mask = event_list.get_event_mask(num_frames)
            self._mode[event_mask] = frame_values[motion_type]
              

        timer.toc('locomotion.motion_events')


    def get_motion_mode(self):
        return self._mode

    @classmethod
    def from_disk(cls, parent_ref):

        self = cls.__new__(cls)
        
        motion_ref = parent_ref['motion']

        for key in self.attribute_keys:
            setattr(self,key,events.EventListWithFeatures.from_disk(motion_ref[key], 'MRC'))
            
        self._mode = utils._extract_time_from_disk(motion_ref, 'mode')

        return self

    def __eq__(self, other):
        motion_events_same = [self.__dict__[x].test_equality(
            other.__dict__[x], 'locomotion.motion_events.' + x)
            for x in self.attribute_keys]
        return all(motion_events_same)
    
    def __repr__(self):
        return utils.print_object(self)     
     
    @property   
    def is_paused(self):
        return self._mode == 0