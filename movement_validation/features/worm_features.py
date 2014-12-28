# -*- coding: utf-8 -*-
"""
WormFeatures module

Contains the classes needed for users to calculate the features 
of a worm from a NormalizedWorm instance.

Classes
---------------------------------------    
WormMorphology
WormLocomotion
WormPosture
WormPath

WormFeatures


A translation of Matlab code written by Jim Hokanson, in the 
SegwormMatlabClasses GitHub repo.  

Original code path:
SegwormMatlabClasses/+seg_worm/@feature_calculator/features.m

"""

import h5py  # For loading from disk
import numpy as np
import collections  # For namedtuple
import time #For FeatureTimer

from .. import utils

from . import feature_processing_options as fpo
from . import feature_comparisons as fc
from . import events
from . import path_features
from . import posture_features
from . import locomotion_features
from . import locomotion_bends
from . import locomotion_turns
from . import morphology_features

#TODO: This is being used when importing movement_validation
#This should probably be changed
#TODO: Can we do something differently ...????
try:
    from .. import user_config
except ImportError:
     raise Exception("user_config.py not found, copy the user_config_example.txt in the 'movement_validation' package to user_config.py in the same directory and edit the values")


"""
===============================================================================
===============================================================================
"""


class WormMorphology(object):

    """
    The worm's morphology features class.

    Nature Methods Description
    ---------------------------------------    
    1. Length. Worm length is computed from the segmented skeleton by
    converting the chain-code pixel length to microns.

    2. Widths. Worm width is computed from the segmented skeleton. The
    head, midbody, and tail widths are measured as the mean of the widths
    associated with the skeleton points covering their respective sections.
    These widths are converted to microns.

    3. Area. The worm area is computed from the number of pixels within the
    segmented contour. The sum of the pixels is converted to microns2.

    4. Area/Length.

    5. Midbody Width/Length.


    Notes
    ---------------------------------------    
    Formerly SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getMorphologyFeatures.m

    Old files that served as a reference:
      morphology_process.m
      schaferFeatures_process.m

    """

    def __init__(self, features_ref):
        """
        
        Parameters:
        -----------
        features_ref : WormFeatures
        
        """
        print('Calculating Morphology Features')

        nw = features_ref.nw

        self.length = nw.lengths
        
        self.width = morphology_features.Widths(features_ref)

        #TODO: This should eventually be calculated from the contour and skeleton
        self.area = nw.tail_areas + \
            nw.head_areas + \
            nw.vulva_areas + \
            nw.non_vulva_areas

        self.area_per_length = self.area / self.length
        self.width_per_length = self.width.midbody / self.length

    @classmethod
    def from_disk(cls, m_var):
        
        self = cls.__new__(cls)

        self.length = utils._extract_time_from_disk(m_var, 'length')
        self.width = morphology_features.Widths.from_disk(m_var['width'])
        self.area = utils._extract_time_from_disk(m_var, 'area')
        self.area_per_length = utils._extract_time_from_disk(m_var, 'areaPerLength')
        self.width_per_length = utils._extract_time_from_disk(m_var, 'widthPerLength')

        return self

    def __eq__(self, other):

        return \
            fc.corr_value_high(self.length, other.length, 'morph.length')  and \
            self.width == other.width and \
            fc.corr_value_high(self.area, other.area, 'morph.area')      and \
            fc.corr_value_high(self.area_per_length, other.area_per_length, 'morph.area_per_length') and \
            fc.corr_value_high(self.width_per_length, other.width_per_length, 'morph.width_per_length')

    def __repr__(self):
        return utils.print_object(self)

    def save_for_gepetto(self):
        # See
        # https://github.com/openworm/org.geppetto.recording/blob/master/org/geppetto/recording/CreateTestGeppettoRecording.py
        pass


"""
===============================================================================
===============================================================================
"""


class WormLocomotion(object):

    """
    The worm's locomotion features class.

    Attributes
    ----------    
    velocity :
    motion_events :
    motion_mode : 
    crawling_bends :
    foraging_bends :
    turns :

    """

    def __init__(self, features_ref):
        """
        Initialization method for WormLocomotion

        Parameters
        ----------
        features_ref : WormFeatures

        """
        print('Calculating Locomotion Features')    

        nw  = features_ref.nw

        self.velocity = locomotion_features.LocomotionVelocity(features_ref)

        self.motion_events = \
            locomotion_features.MotionEvents(features_ref,
                                             self.velocity.midbody.speed,
                                             nw.lengths)

        self.motion_mode = self.motion_events.get_motion_mode()

        self.crawling_bends = locomotion_bends.LocomotionCrawlingBends(
            features_ref,
            nw.angles,
            self.motion_events.is_paused,
            nw.is_segmented)

        self.foraging_bends = locomotion_bends.LocomotionForagingBends(
            features_ref,nw.is_segmented,nw.ventral_mode)

        is_stage_movement = nw.segmentation_status == 'm'

        self.turns = locomotion_turns.LocomotionTurns(features_ref, 
                                                      nw.angles,
                                                      is_stage_movement,
                                                      self.velocity.get_midbody_distance(),
                                                      nw.x,
                                                      nw.y)

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):

        # TODO: Allow for a global config that provides more info ...
        # in case anything fails ...
        #
        #   JAH: I'm not sure how this will work. We might need to move
        #   away from the equality operator to a function that returns
        #   an equality result

        #NOTE: We'll test everything instead of short-circuiting
        same_locomotion = True

        if not (self.velocity == other.velocity):
            same_locomotion = False

        if not (self.motion_events == other.motion_events):
            same_locomotion = False

        # Test motion codes
        if not fc.corr_value_high(self.motion_mode, other.motion_mode, 'locomotion.motion_mode'):
            same_locomotion = False

        #TODO: Define ne for all functions (instead of needing not(eq))
        if not (self.crawling_bends == other.crawling_bends):
            print('Mismatch in locomotion.crawling_bends events')
            same_locomotion = False
            
        if not (self.foraging_bends == other.foraging_bends):
            print('Mismatch in locomotion.foraging events')
            same_locomotion = False

        #TODO: Make eq in events be an error - use test_equality instead        
        if not (self.turns == other.turns):
            print('Mismatch in locomotion.turns events')
            same_locomotion = False

        return same_locomotion

    @classmethod
    def from_disk(cls, m_var):
        """
        Parameters
        ----------
        m_var : type???? h5py.Group???
            ?? Why is this this called m_var????
        """


        self = cls.__new__(cls)

        self.velocity = locomotion_features.LocomotionVelocity.from_disk(m_var)

        self.motion_events = locomotion_features.MotionEvents.from_disk(m_var)

        self.motion_mode = self.motion_events.get_motion_mode()

        bend_ref = m_var['bends']
        self.crawling_bends = locomotion_bends.LocomotionCrawlingBends.from_disk(bend_ref)
        
        self.foraging_bends = locomotion_bends.LocomotionForagingBends.from_disk(bend_ref['foraging'])
        
        self.turns    = locomotion_turns.LocomotionTurns.from_disk(m_var['turns'])

        return self


"""
===============================================================================
===============================================================================
"""


class WormPosture(object):

    """
    Worm posture feature class.

    Notes
    -----
    Formerly:
    SegwormMatlabClasses/+seg_worm/@feature_calculator/getPostureFeatures.m

    Former usage: 

    Prior to this, it was originally "schaferFeatures_process"

    Formerly,
    - Indices were inconsistently defined for bends relative to other code
    - stdDev for bends is signed as well, based on means ...

    Unfinished Status
    ---------------------------------------    
    (@JimHokanson, is this still true?)
    - seg_worm.feature_helpers.posture.wormKinks - not yet examined
    - distance - missing input to function, need to process locomotion
      first

    """

    def __init__(self, features_ref, midbody_distance):
        """
        Initialization method for WormPosture

        Parameters
        ----------  
        normalized_worm: a NormalizedWorm instance

        """
        print('Calculating Posture Features')            
        
        # Let's use a shorthand
        nw  = features_ref.nw
        ##options = features_ref.options
        
        self.bends = posture_features.Bends.create(features_ref)

        self.eccentricity, orientation = \
            posture_features.get_eccentricity_and_orientation(features_ref)

        amp_wave_track = posture_features.AmplitudeAndWavelength(
            orientation, features_ref)

        self.amplitude_max = amp_wave_track.amplitude_max
        self.amplitude_ratio = amp_wave_track.amplitude_ratio
        self.primary_wavelength = amp_wave_track.primary_wavelength
        self.secondary_wavelength = amp_wave_track.secondary_wavelength
        self.track_length = amp_wave_track.track_length

        self.kinks = posture_features.get_worm_kinks(features_ref)

        self.coils = posture_features.get_worm_coils(features_ref, midbody_distance)

        self.directions = posture_features.Directions(features_ref)

        #TODO: I'd rather this be a formal class
        nt = collections.namedtuple('skeleton', ['x', 'y'])
        self.skeleton = nt(nw.skeleton_x, nw.skeleton_y)

        self.eigen_projection = posture_features.get_eigenworms(features_ref)

    @classmethod
    def from_disk(cls, p_var):

        self = cls.__new__(cls)
        
        self.bends = posture_features.Bends.from_disk(p_var['bends'])

        temp_amp = p_var['amplitude']

        self.amplitude_max = utils._extract_time_from_disk(temp_amp, 'max')
        self.amplitude_ratio = utils._extract_time_from_disk(temp_amp, 'ratio')

        temp_wave = p_var['wavelength']
        self.primary_wavelength = utils._extract_time_from_disk(temp_wave, 'primary')
        self.secondary_wavelength = utils._extract_time_from_disk(temp_wave, 'secondary')

        self.track_length = utils._extract_time_from_disk(p_var, 'tracklength')
        self.eccentricity = utils._extract_time_from_disk(p_var, 'eccentricity')
        self.kinks = utils._extract_time_from_disk(p_var, 'kinks')

        self.coils = events.EventListWithFeatures.from_disk(p_var['coils'], 'MRC')

        self.directions = posture_features.Directions.from_disk(p_var['directions'])

        # TODO: Add contours
        skeleton = p_var['skeleton']
        nt = collections.namedtuple('skeleton', ['x', 'y'])
        x_temp = utils._extract_time_from_disk(skeleton,'x',is_matrix=True)
        y_temp = utils._extract_time_from_disk(skeleton,'y',is_matrix=True)
        self.skeleton = nt(x_temp.transpose(), y_temp.transpose())
        
        temp_eigen_projection = utils._extract_time_from_disk(p_var,'eigenProjection',is_matrix=True)        
        
        self.eigen_projection = temp_eigen_projection.transpose()

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):

        # TODO: 
        # Missing:
        # directions
        # bend

        #TODO: It would be nice to see all failures before returning false
        #We might want to make a comparison class that handles these details 
        #and then prints the results

        #Doing all of these comparisons and then computing the results
        #allows any failures to be printed, which at this point is useful for 
        #getting the code to align

        eq_eccentricity = fc.corr_value_high(self.eccentricity, other.eccentricity, 'posture.eccentricity',high_corr_value=0.99)
        eq_amplitude_ratio = fc.corr_value_high(self.amplitude_ratio, other.amplitude_ratio, 'posture.amplitude_ratio',high_corr_value=0.985) 
        eq_track_length = fc.corr_value_high(self.track_length, other.track_length, 'posture.track_length')
        eq_kinks = fc.corr_value_high(self.kinks, other.kinks, 'posture.kinks')
        eq_primary_wavelength = fc.corr_value_high(self.primary_wavelength,other.primary_wavelength,'posture.primary_wavelength')
        eq_secondary_wavelength = fc.corr_value_high(self.secondary_wavelength, other.secondary_wavelength, 'posture.secondary_wavelength')
        eq_amplitude_max = fc.corr_value_high(self.amplitude_max, other.amplitude_max, 'posture.amplitude_max')        
        eq_skeleton_x = fc.corr_value_high(np.ravel(self.skeleton.x), np.ravel(other.skeleton.x), 'posture.skeleton.x')
        eq_skeleton_y = fc.corr_value_high(np.ravel(self.skeleton.y), np.ravel(other.skeleton.y), 'posture.skeleton.y')
        eq_coils = self.coils.test_equality(other.coils,'posture.coils')        
        eq_eigen_projection = fc.corr_value_high(np.ravel(self.eigen_projection), np.ravel(other.eigen_projection), 'posture.eigen_projection')
        
        return \
            eq_eccentricity and \
            eq_amplitude_ratio and \
            eq_track_length and \
            eq_kinks and \
            eq_primary_wavelength and \
            eq_secondary_wavelength and \
            eq_amplitude_max and \
            eq_skeleton_x and \
            eq_skeleton_y and \
            eq_eigen_projection and \
            eq_coils




"""
===============================================================================
===============================================================================
"""


class WormPath(object):

    """
    Worm posture feature class.

    Properties
    ------------------------
    range :
    duration :
    coordinates :
    curvature :

    Notes
    ---------------------------------------    
    Formerly SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getPathFeatures.m

    """

    def __init__(self, features_ref):
        """
        Initialization method for WormPosture

        Parameters:
        -----------
        features_ref: a WormFeatures instance
        """
        print('Calculating Path Features')        

        nw = features_ref.nw

        self.range = path_features.Range(nw.contour_x, nw.contour_y)

        # Duration (aka Dwelling)
        self.duration = path_features.Duration(features_ref)

        self.coordinates = path_features.Coordinates(features_ref)

        #Curvature
        self.curvature = path_features.worm_path_curvature(features_ref)

    # TODO: Move to class in path_features
    @classmethod
    def _create_coordinates(cls, x, y):
        Coordinates = collections.namedtuple('Coordinates', ['x', 'y'])
        return Coordinates(x, y)

    @classmethod
    def from_disk(cls, path_var):

        self = cls.__new__(cls)

        self.range = path_features.Range.from_disk(path_var)
        self.duration = path_features.Duration.from_disk(path_var['duration'])

        self.coordinates = path_features.Coordinates.from_disk(path_var['coordinates'])

        #Make a call to utils loader
        self.curvature = path_var['curvature'].value[:, 0]

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):

        return \
            self.range == other.range and \
            self.duration == other.duration and \
            self.coordinates == other.coordinates and \
            fc.corr_value_high(self.curvature, other.curvature,
                               'path.curvature',
                               high_corr_value=0.95,
                               merge_nans=True)

        # NOTE: Unfortunately the curvature is slightly different. It looks the same
        # but I'm guessing there are a few off by 1 errors in it.


"""
===============================================================================
===============================================================================
"""


class WormFeatures(object):

    """ 
    WormFeatures: Takes a NormalizedWorm instance and
    during initialization calculates all the features of the worm.

    There are two ways to initialize a WormFeatures instance: 
    1. by passing a NormalizedWorm instance and generating the features, or
    2. by loading the already-calculated features from an HDF5 file.
         (via the from_disk method)
         
    Attributes
    ----------      
    video_info : movement_validation.video_info
    options : movement_validation.features.feature_processing_options
    nw : movement_validation.NormalizedWorm
    morphology : WormMorphology
    locomotion : WormLocomotion
    posture : WormPosture
    path : WormPath

    """

    def __init__(self, nw, video_info, processing_options=None):
        """
        
        Parameters
        ----------
        nw : movement_validation.NormalizedWorm
        video_info : movement_validation.video_info
        processing_options : movement_validation.features.feature_processing_options

        """
        
        #TODO: Create the normalized worm in here ... 

        if processing_options is None:
            processing_options = fpo.FeatureProcessingOptions(video_info.fps)

        #These are saved locally for reference by others when processing
        self.video_info = video_info
        self.options = processing_options
        self.nw = nw
        self.timer = FeatureTimer()
        
        self.morphology = WormMorphology(self)
        self.locomotion = WormLocomotion(self)
        self.posture = WormPosture(self, 
                                   self.locomotion.velocity.get_midbody_distance())
        self.path = WormPath(self)

    @classmethod
    def from_disk(cls, file_path):

        """
        This from disk method is currently focused on loading the features
        files as computed by the Schafer lab. Alternative loading methods
        should be possible 
        """
        h = h5py.File(file_path, 'r')
        worm = h['worm']

        self = cls.__new__(cls)

        self.morphology = WormMorphology.from_disk(worm['morphology'])
        self.locomotion = WormLocomotion.from_disk(worm['locomotion'])
        self.posture = WormPosture.from_disk(worm['posture'])
        self.path = WormPath.from_disk(worm['path'])

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        """
        Compare two WormFeatures instances by value

        """
        same_morphology = self.morphology == other.morphology
        same_locomotion = self.locomotion == other.locomotion
        same_posture = self.posture == other.posture
        same_path = self.path == other.path
        return same_morphology and \
             same_locomotion and \
             same_posture and \
             same_path
                      
            
        
class FeatureTimer(object):

    """
    This class is meant to be called in the following way by code that is 
    processing a feature.
    
    timer = features_ref.timer
    timer.tic()
    #Run the feature processing code
    timer.toc('name of feature being processed')    
        
    """

    def __init__(self):    
        self.names = []
        self.times = []
        
    def tic(self):
        self.start_time = time.time()
    
    def toc(self,name):
        self.times.append(time.time() - self.start_time)
        self.names.append(name)
        
    def __repr__(self):
        return utils.print_object(self)
        
    def summarize(self):
        """
        This can be called to display each logged function and how long it
        took to run
        """
        for (name,time) in zip(self.names,self.times):
            print('%s: %0.3fs' %(name,time))

            
