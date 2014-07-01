# -*- coding: utf-8 -*-
"""
WormFeatures.py:

@authors: @JimHokanson, @MichaelCurrie

Classes
---------------------------------------    
WormMorphology
WormLocomotion
WormPosture
WormPath

WormFeatures

Functions
---------------------------------------    
_extract_time_from_disk


A translation of Matlab code written by Jim Hokanson,
in the SegwormMatlabClasses GitHub repo.  Original code path:
SegwormMatlabClasses / 
+seg_worm / @feature_calculator / get_features_rewritten.m

"""

import csv
from . import feature_comparisons as fc
from . import EventFinder
import h5py #For loading from disk 
import numpy as np
import collections #For namedtuple
from wormpy import config
from wormpy import feature_helpers
from . import path_features
from . import posture_features
from . import utils
from . import locomotion_bends
from . import locomotion_turns


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
  
  def __init__(self, nw):
    
    self.length = nw.data_dict['lengths']
    # each item in this sub-dictionary is the per-frame mean across some
    # part of the worm the head, midbody and tail.
    #
    # shape of resulting arrays are (2, n)
    
    width_dict = {k: np.mean(nw.get_partition(k, 'widths'), 0) \
                  for k in ('head', 'midbody', 'tail')}
            
    #Make named tuple instead of dict
    nt = collections.namedtuple('Widths', width_dict.keys())
    self.width = nt(**width_dict)
          
    #TODO: The access from nw should be cleaned up, e.g. nw.head_areas        
    self.area = nw.data_dict['tail_areas']      + \
                nw.data_dict['head_areas']      + \
                nw.data_dict['vulva_areas']     + \
                nw.data_dict['non_vulva_areas']
                
    self.area_per_length  = self.area/self.length
    self.width_per_length = self.width.midbody/self.length

  @classmethod 
  def from_disk(cls, m_var):
    
    """
    
    Status: Done
    """
    self = cls.__new__(cls)   
    
    #TODO: More gracefully handle removal of the 2nd dimension ...
    self.length = m_var['length'].value[:,0]
    temp1 = m_var['width']

    temp2 = {k: temp1[k].value[:,0] for k in ('head','midbody','tail')}
    
    nt = collections.namedtuple('Widths',['head','midbody','tail'])
    self.width  = nt(**temp2) 

    self.area             = m_var['area'].value[:,0]
    self.area_per_length  = m_var['areaPerLength'].value[:,0]
    self.width_per_length = m_var['widthPerLength'].value[:,0]

    return self

  def __eq__(self,other):
    
    #TODO: Allow for a global config that provides more info ...    
    #in case anything fails ...

    #NOTE: Since all features are just attributes in this class we do
    #the evaluation here rather than calling __eq__ on the classes

    return \
      fc.corr_value_high(self.length,other.length,'morph.length')  and \
      fc.corr_value_high(self.area,other.area,'morph.area')      and \
      fc.corr_value_high(self.area_per_length,other.area_per_length,'morph.area_per_length') and \
      fc.corr_value_high(self.width_per_length,other.width_per_length,'morph.width_per_length') and \
      fc.corr_value_high(self.width.head,other.width.head,'morph.width.head') and \
      fc.corr_value_high(self.width.midbody,other.width.midbody,'morph.width.midbody') and \
      fc.corr_value_high(self.width.tail,other.width.tail,'morph.width.tail')


  def __repr__(self):
    return utils.print_object(self)  
    
  def save_for_gepetto(self):
    #See https://github.com/openworm/org.geppetto.recording/blob/master/org/geppetto/recording/CreateTestGeppettoRecording.py
    pass



"""
===============================================================================
===============================================================================
"""


class WormLocomotion(object):
  """
  The worm's locomotion features class.

  Properties
  ---------------------------------------    
  velocity:
    (head_tip, head, midbody, tail, tail_tip) x (speed, direction)
  motion
  motion_mode
  is_paused
  bends
  foraging
  omegas
  upsilons

  Properties (full listing)
  ---------------------------------------    

  .velocity
    .headTip
      .speed - time series   
      .direction - time series   
    .head - all have same format
    .midbody
    .tail
    .tailTip
  .motion
    .forward
    .backward
    .paused
    .mode - time series   
  .bends
    .foraging
      .amplitude - time series 
      .angleSpeed - time series 
    .head
      .amplitude - time series 
      .frequency - time series 
    .midbody - same as head
    .tail  - same as head
  .turns
    .omegas
    .upsilons


  Notes
  ---------------------------------------    
  Formerly SegwormMatlabClasses / 
    +seg_worm / +features / @locomotion / locomotion.m

  """


  def __init__(self, normalized_worm):
    """
    Initialization method for WormLocomotion

    Parameters
    ---------------------------------------    
    normalized_worm: a NormalizedWorm instance

    """
    # DEBUG
    #feature_helpers.write_to_CSV(
    #      {
    #        'Midbody Speed': self.velocity['midbody']['speed'],
    #        'config.FPS': np.array([config.FPS],dtype='float'),
    #        'lengths': nw.data_dict['lengths']
    #      },
    #      'motion_codes_input'
    #      )

    # let's use a shorthand
    nw = normalized_worm

    self.velocity = feature_helpers.get_worm_velocity(nw)

    self.motion_events = \
      feature_helpers.get_motion_codes(self.velocity['midbody']['speed'], 
                                       nw.data_dict['lengths'])
                                       
    #TODO: I'm not a big fan of how this is done ...
    self.motion_mode = self.motion_events['mode']

    del self.motion_events['mode']
    
    self.is_paused = self.motion_mode == 0

    self.bends = locomotion_bends.LocomotionCrawlingBends(
                                                  nw.data_dict['angles'],
                                                  self.is_paused,
                                                  nw.is_segmented)
                                                  
    """
    self.foraging = locomotion_bends.LocomotionForagingBends(
                                                  nw,
                                                  nw.is_segmented,
                                                  nw.ventral_mode)
    """
    
    midbody_distance = abs(self.velocity['midbody']['speed'] / config.FPS)
    is_stage_movement = nw.data_dict['segmentation_status'] == 'm'
    
    """
    self.turns = locomotion_turns.LocomotionTurns(nw,nw.data_dict['angles'],
                                                  is_stage_movement,
                                                  midbody_distance,
                                                  nw.skeleton_x,
                                                  nw.skeleton_y)
    """
    
  def __repr__(self):
    return utils.print_object(self)  
    
  def __eq__(self,other):
    
    #TODO: Allow for a global config that provides more info ...    
    #in case anything fails ...

    #NOTE: Since all features are just attributes in this class we do
    #the evaluation here rather than calling __eq__ on the classes

    #self_velocity  = self.velocity
    #other_velocity = other.velocity


  
    #velocities_same = [self_velocity[x] == other_velocity[x] for x in self_velocity]
    for key in self.velocity:
      self_speed      = self.velocity[key]['speed']
      self_direction  = self.velocity[key]['direction']
      other_speed     = other.velocity[key]['speed']
      other_direction = other.velocity[key]['direction']

      same_speed = fc.corr_value_high(self_speed,
                                      other_speed,
                                      'locomotion.velocity.' + key + '.speed')
      if not same_speed:
        return False
        
      same_direction =  fc.corr_value_high(
                                      self_direction,
                                      other_direction,
                                      'locomotion.velocity.' + key + '.speed') 
      if not same_direction:
        return False
        
    #---------------------------------
     
    #Test motion events 
    #---------------------------------    
    motion_events_same = [self.motion_events[x].test_equality(\
      other.motion_events[x],'locomotion.motion_events.' + x)
                              for x in self.motion_events]  

    if not all(motion_events_same):
      return False
      
    #Test motion codes
    if not fc.corr_value_high(self.motion_mode,other.motion_mode,'locomotion.motion_mode'):
      return False

    
    #TODO: bends - Not Yet Implemented
    #--------------------
    #    foraging: [1x1 struct]
    #        head: [1x1 struct]
    #     midbody: [1x1 struct]
    #        tail: [1x1 struct]    
    
    #TODO: turns - Not Yet Implemented
    #--------------------
    #    omegas: [1x1 struct]
    #    upsilons: [1x1 struct]     
    

    return True

      
    
  @classmethod 
  def from_disk(cls, m_var):
    
    self = cls.__new__(cls)

    #bends
    #motion
    #turns

    velocity = {}
    
    #velocity
    #------------------------------
    
    velocity_ref = m_var['velocity']
    for key in velocity_ref:
        value = velocity_ref[key]
        temp_speed = _extract_time_from_disk(value,'speed')
        temp_direc = _extract_time_from_disk(value,'direction')
        velocity[key] = {'speed': temp_speed, 'direction': temp_direc}
     
    #NOTE: This only valid for MRC
     
    velocity['head_tip'] = velocity.pop('headTip') 
    velocity['tail_tip'] = velocity.pop('tailTip')      
     
    self.velocity = velocity

    #motion
    #------------------------------    
    self.motion_events = {}
    
    if 'motion' in m_var.keys():
      #Old MRC Format:
      # - forward, backward, paused, mode
      motion_ref = m_var['motion']
      for key in ['forward','backward','paused']:
        self.motion_events[key] = \
          EventFinder.EventListForOutput.from_disk(motion_ref[key],'MRC')
       
      self.motion_mode = _extract_time_from_disk(motion_ref,'mode')      
    else:
      raise Exception('Not yet implemented')
    
    #TODO: bends - Not Yet Implemented
    #--------------------
    #    foraging: [1x1 struct]
    #        head: [1x1 struct]
    #     midbody: [1x1 struct]
    #        tail: [1x1 struct]    
    
    #TODO: turns - Not Yet Implemented
    #--------------------
    #    omegas: [1x1 struct]
    #    upsilons: [1x1 struct] 
    
    return self



"""
===============================================================================
===============================================================================
"""
    

class WormPosture(object):
  """
  Worm posture feature class.
  
  Notes
  ---------------------------------------    
  Formerly SegwormMatlabClasses / 
  +seg_worm / @feature_calculator / getPostureFeatures.m

  Former usage: 
  
  posture = seg_worm.feature_calculator.getPostureFeatures(nw)

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

  def __init__(self, normalized_worm,midbody_distance):
    """
    Initialization method for WormPosture

    Parameters
    ---------------------------------------    
    normalized_worm: a NormalizedWorm instance

    """
    # Let's use a shorthand
    nw = normalized_worm    

    # *** 1. Bends *** DONE
    self.bends = posture_features.Bends(nw)
      
    # *** 2. Eccentricity & Orientation *** DONE, SLOW
    #This has not been optimized, that Matlab version has
    #NOTE: This is VERY slow, leaving commented for now
    #self.eccentricity,self.orientation = \
    #   posture_features.get_eccentricity_and_orientation(nw.contour_x,
    #                                                     nw.contour_y)

    
    #Temp input for next function ...
    self.orientation = np.zeros(nw.skeleton_x.shape[1])
    # *** 3. Amplitude, Wavelengths, TrackLength, Amplitude Ratio *** NOT DONE
    amp_wave_track = posture_features.get_amplitude_and_wavelength(
                          self.orientation,
                          nw.skeleton_x,
                          nw.skeleton_y,
                          nw.data_dict['lengths'])    

    self.amplitude_max        = amp_wave_track.amplitude_max
    self.amplitude_ratio      = amp_wave_track.amplitude_ratio 
    #self.primary_wavelength   = amp_wave_track.p_wavelength
    #self.secondary_wavelength = amp_wave_track.s_wavelength  
    self.track_length         = amp_wave_track.track_length

    # *** 4. Kinks *** DONE
    self.kinks = posture_features.get_worm_kinks(nw.data_dict['angles'])
        
    # *** 5. Coils ***
    frame_codes = nw.data_dict['frame_codes']
    self.coils = posture_features.get_worm_coils(frame_codes,midbody_distance)

    # *** 6. Directions *** DONE
    self.directions = posture_features.Directions(nw.skeleton_x,
                                                  nw.skeleton_y,
                                                  nw.worm_partitions)

    # *** 7. Skeleton *** DONE
    # (already in morphology, but Schafer Lab put it here too)
    nt = collections.namedtuple('skeleton',['x','y'])
    self.skeleton = nt(nw.skeleton_x,nw.skeleton_y)
    
    # *** 8. EigenProjection *** DONE
    eigen_worms = nw.eigen_worms

    self.eigen_projection = posture_features.get_eigenworms(
        nw.skeleton_x, nw.skeleton_y,
        np.transpose(eigen_worms),
        config.N_EIGENWORMS_USE)



  @classmethod 
  def from_disk(cls, p_var):
    
    self = cls.__new__(cls)
    self.bend = posture_features.Bends.from_disk(p_var['bends'])
      
    #NOTE: This will be considerably different for old vs new format. Currently
    #only the old is implemented
    temp_amp = p_var['amplitude']
    self.amplitude_max   = temp_amp['max'].value
    self.amplitude_ratio = temp_amp['ratio'].value
    
    temp_wave = p_var['wavelength']
    self.primary_wavelength   = temp_wave['primary']
    self.secondary_wavelength = temp_wave['secondary']

    self.track_length = p_var['tracklength'].value
    self.eccentricity = p_var['eccentricity'].value
    self.kinks        = p_var['kinks'].value
    
    EventFinder.EventListForOutput.from_disk(p_var['coils'],'MRC')
    
    self.directions   = posture_features.Directions.from_disk(
                                                      p_var['directions'])    
      
    #TODO: Add contours     
    skeleton = p_var['skeleton']
    nt = collections.namedtuple('skeleton',['x','y'])
    self.skeleton = nt(skeleton['x'].value,skeleton['y'].value)

    self.eigen_projection = p_var['eigenProjection'].value

    return self

  def __repr__(self):
    return utils.print_object(self)  



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
  
  def __init__(self, normalized_worm):
    """
    Initialization method for WormPosture

    Parameters
    ---------------------------------------    
    normalized_worm: a NormalizedWorm instance


    """    
    # Let's use a shorthand
    nw = normalized_worm    
    
    self.range = path_features.Range(nw.contour_x,nw.contour_y)
        
    #Duration (aka Dwelling)
    #---------------------------------------------------
    sx     = nw.skeleton_x
    sy     = nw.skeleton_y
    widths = nw.data_dict['widths']
    self.duration = path_features.Duration(nw, sx, sy, widths, config.FPS)
  
    #Coordinates (Done)
    #---------------------------------------------------    
    self.coordinates = self._create_coordinates(nw.contour_x.mean(axis=0),
                                                nw.contour_y.mean(axis=0))
       
    #Curvature (Done)
    #---------------------------------------------------
    self.curvature = path_features.worm_path_curvature(sx, sy,
                                                       config.FPS,
                                                       nw.ventral_mode)

  #TODO: Move to class in path_features
  @classmethod
  def _create_coordinates(cls, x, y):
    Coordinates = collections.namedtuple('Coordinates',['x','y'])
    return Coordinates(x, y)

  @classmethod 
  def from_disk(cls, path_var):
    
    self = cls.__new__(cls) 
    
    self.range       = path_features.Range.from_disk(path_var)
    self.duration    = path_features.Duration.from_disk(path_var['duration']) 

    #TODO: I'd like to have these also be objects with from_disk methods
    self.coordinates = self._create_coordinates(
                          path_var['coordinates']['x'].value[:,0],
                          path_var['coordinates']['y'].value[:,0])
                          
    self.curvature   = path_var['curvature'].value[:,0]   

    return self
    
  def __repr__(self):
    return utils.print_object(self)  
    
  def __eq__(self,other):

    return \
      self.range == other.range and \
      self.duration == other.duration and \
      fc.corr_value_high(self.coordinates.x, other.coordinates.x,
                         'path.coordinates.x') and \
      fc.corr_value_high(self.coordinates.y, other.coordinates.y,
                         'path.coordinates.y') and \
      fc.corr_value_high(self.curvature, other.curvature,
                         'path.curvature',
                         high_corr_value=0.95,
                         merge_nans=True)

      #NOTE: Unfortunately the curvature is slightly different. It looks the same
      #but I'm guessing there are a few off by 1 errors in it.



"""
===============================================================================
===============================================================================
"""

    
class WormFeatures(object):
  """ 
    WormFeatures: takes as input a NormalizedWorm instance, and
    during initialization calculates all the features of the worm.
    
    There are two ways to initialize a WormFeatures instance: 
    1. by passing a NormalizedWorm instance and generating the features, or
    2. by loading the already-calculated features from an HDF5 file.
       (via the from_disk method)
    
  """
  def __init__(self, nw):

    self.morphology = WormMorphology(nw)
    self.locomotion = WormLocomotion(nw)
    
    midbody_distance = np.abs(self.locomotion.velocity['midbody']['speed']/config.FPS)
    self.posture     = WormPosture(nw,midbody_distance)
    self.path        = WormPath(nw)
    
  @classmethod  
  def from_disk(cls, file_path):
    
    h = h5py.File(file_path,'r')
    worm = h['worm']
    
    self = cls.__new__(cls) 
    
    self.morphology = WormMorphology.from_disk(worm['morphology'])
    self.locomotion = WormLocomotion.from_disk(worm['locomotion'])
    self.posture    = WormPosture.from_disk(worm['posture'])
    self.path       = WormPath.from_disk(worm['path'])
    
    return self
    
  def __repr__(self):
    return utils.print_object(self)
    
  def __eq__(self,other):
    """
      Compare two WormFeatures instances by value
    
    """
    return \
      self.path       == other.path       and \
      self.morphology == other.morphology and \
      self.posture    == other.posture    and \
      self.locomotion == other.locomotion


"""
===============================================================================
===============================================================================
"""

        
def _extract_time_from_disk(parent_ref,name):
        
    """
    This is for handling Matlab save vs Python save when we get to that point.
    """        
        
    temp = parent_ref[name].value
    
    #Assuming vector, need to fix for eigenvectors
    if temp.shape[0] > temp.shape[1]:
      wtf = temp[:,0]
    else:
      wtf = temp[0,:]
    
    return wtf    

    