# -*- coding: utf-8 -*-
"""
  WormFeatures.py:
  
  @authors: @JimHokanson, @MichaelCurrie
  
  A translation of Matlab code written by Jim Hokanson,
  in the SegwormMatlabClasses GitHub repo.  Original code path:
  SegwormMatlabClasses / 
  +seg_worm / @feature_calculator / get_features_rewritten.m
  
  NOTE: REQUIRES numpy.version.full_version >= '1.8' 
  since numpy.nanmean is only available after that version.
  (http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.nanmean.html)
  Alternatively, you can just install nanfunctions.py 
  (see instructions in ..//README.md in this repo)
  
  *** For +seg_worm / @feature_calculator / getPostureFeatures.m,
  *** here are some renamed variables:
  
  SI = seg_worm.skeleton_indices is expressed here as self.skeleton_partitions
  ALL_INDICES = SI.ALL_NORMAL_INDICES is expressed here as 
                self.normal_partitions()
  FIELDS = SI.ALL_NORMAL_NAMES is expressed here as 
           self.normal_partitions().keys()
  n_fields = length(FIELDS) = len(self.normal_partitions().keys())

"""

from . import user_config as uconfig
import h5py #For loading from disk 
import numpy as np
import collections #For namedtuple
from wormpy import config
from wormpy import feature_helpers
from . import path_features
from . import posture_features
from . import utils

#import pdb

class WormMorphology(object):
  def __init__(self, nw):
    """
      Translation of: SegwormMatlabClasses / 
      +seg_worm / @feature_calculator / getMorphologyFeatures.m
      
      Nature Methods Description
      =======================================================================
       
      Morphology Features 
       
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
      
      get_morphology_features:
      * Takes nw, and generates a structure called "morphology"
      
      %Old files that served as a reference ...
      %------------------------------------------------------------
      %morphology_process.m
      %schaferFeatures_process.m
  
    """
    
    self.length = nw.data_dict['lengths']
    # each item in this sub-dictionary is the per-frame mean across some
    # part of the worm the head, midbody and tail.
    #
    # shape of resulting arrays are (2, n)
    width_dict = {k: np.mean(nw.get_partition(k, 'skeletons'), 0) \
                  for k in ('head', 'midbody', 'tail')}
            
    #Make named tuple instead of dict
    nt = collections.namedtuple('Widths',width_dict.keys())
    self.width = nt(**width_dict)
          
    #TODO: The access from nw should be cleaned up, e.g. nw.head_areas        
    self.area = nw.data_dict['head_areas'] + \
                nw.data_dict['vulva_areas'] + \
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
    #import pdb
    #pdb.set_trace()
    temp2 = {k: temp1[k].value[:,0] for k in ('head','midbody','tail')}
    
    #I'm not sure why this doesn't work, 
    #
    # unhashable type: 'numpy.ndarray'
    #
    #temp2 = {
    #'head',     temp1['head'].value[:,0],
    #'midbody',  temp1['midbody'].value[:,0],
    #'tail',     temp1['tail'].value[:,0]}
    nt = collections.namedtuple('Widths',['head','midbody','tail'])
    self.width  = nt(**temp2) 

    self.area             = m_var['area'].value[:,0]
    self.area_per_length  = m_var['areaPerLength'].value[:,0]
    self.width_per_length = m_var['widthPerLength'].value[:,0]

    return self

  def __eq__(self,other):
    
    import pdb
    pdb.set_trace()  
    
    return True

  def __repr__(self):
    return utils.print_object(self) 
    
  def save_for_gepetto(self):
    #See https://github.com/openworm/org.geppetto.recording/blob/master/org/geppetto/recording/CreateTestGeppettoRecording.py
    pass



class WormLocomotion():
  def __init__(self, nw):
    """
      Translation of: SegwormMatlabClasses / 
      +seg_worm / +features / @locomotion / locomotion.m
  
        properties
          velocity:
            (head_tip, head, midbody, tail, tail_tip) x (speed, direction)
          motion
          motion_mode
          is_paused
          bends
          foraging
          omegas
          upsilons
        end

    """
    self.locomotion = {}

    self.velocity = feature_helpers.get_worm_velocity(nw)

    midbody_distance = \
      abs(self.velocity['midbody']['speed'] / config.FPS)
    
    self.motion_codes = \
      feature_helpers.get_motion_codes(midbody_distance, 
                                       nw.data_dict['lengths'])
  
    self.motion_mode = 0
    
    self.is_paused = 0

    self.bends = 0

    self.foraging = 0

    self.omegas = 0

    self.upsilons = 0
    
    #.motion
    #  .forward
    #  .backward
    #  .paused
    #  .mode - time series   
    #.velocity
    #  .headTip
    #    .speed - time series   
    #    .direction - time series   
    #  .head - all have same format
    #  .midbody
    #  .tail
    #  .tailTip
    #.bends
    #  .foraging
    #    .amplitude - time series 
    #    .angleSpeed - time series 
    #  .head
    #    .amplitude - time series 
    #    .frequency - time series 
    #  .midbody - same as head
    #  .tail  - same as head
    #.turns
    #  .omegas
    #  .upsilons
    
  
  @classmethod 
  def from_disk(cls, m_var):
    
    self = cls.__new__(cls)

    import pdb
    pdb.set_trace()
    
    return self

    

class WormPosture():
  def __init__(self, nw):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getPostureFeatures.m

    %
    %   posture = seg_worm.feature_calculator.getPostureFeatures(nw)
    %
    %   Old Files
    %   - schaferFeatures_process
    %
    %   NOTES:
    %   - Indices were inconsistently defined for bends relative to other code
    %   - stdDev for bends is signed as well, based on means ...
    %
    %   UNFINISHED STATUS:
    %   - seg_worm.feature_helpers.posture.wormKinks - not yet examined
    %   - distance - missing input to function, need to process locomotion
    %   first
    %

    """    
 

    # *** 1. Bends *** DONE
    self.bends = posture_features.Bends(nw)
      
    # *** 2. Eccentricity & Orientation *** DONE, SLOW
    #This has not been optimized, that Matlab version has
    #NOTE: This is VERY slow, leaving commented for now
    #self.eccentricity,self.orientation = \
    #   posture_features.get_eccentricity_and_orientation(nw.contour_x,nw.contour_y)

    
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
    self.coils = posture_features.get_worm_coils()


    # *** 6. Directions *** DONE
    self.directions = posture_features.Directions(nw.skeleton_x,nw.skeleton_y,nw.worm_partitions)

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

    #TODO: Add contours

  @classmethod 
  def from_disk(cls, p_var):
    
    self = cls.__new__(cls)

    #bends
    #  .head
    #    .mean
    #    .std_dev
    #  .neck
    #  .midbody
    #  .hips
    #  .tail
    #.amplitude
    #  .max   - ts
    #  .ratio - ts
    #.wavelength
    #  .primary - ts
    #  .secondary - ts
    #.track_length - ts (OLD: tracklength)
    #.eccentricity - ts
    #.kinks - ts
    #.coils - event
    #.directions - 
    #  .tail2head - ts
    #  .head - ts
    #  .tail - ts
    #.skeleton
    #  .x - ts
    #  .y - ts
    #.eigen_projections [6 x frames] matrix (OLD:eigenProjection)

    import pdb
    pdb.set_trace()
    
    return self    

class WormPath():
  
  """
  
  Attributes:
  ------------------------
  range :
  duration :
  coordinates :
  curvature :
  
  """
  
  def __init__(self, nw):
    """
    Translation of: SegwormMatlabClasses / 
    +seg_worm / @feature_calculator / getPathFeatures.m

    """    
    
    #Pass in none to create from disk
    if nw is None:
      return

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
    self.curvature = path_features.worm_path_curvature(sx,sy,config.FPS,config.VENTRAL_MODE)

  #TODO: Move to class in path_features
  @classmethod
  def _create_coordinates(cls, x, y):
    Coordinates = collections.namedtuple('Coordinates',['x','y'])
    return Coordinates(x, y)

  @classmethod 
  def from_disk(cls, path_var):
    self = cls(None)   
    
    self.range       = path_features.Range.from_disk(path_var)
    self.duration    = path_features.Duration.from_disk(path_var['duration']) 

    #TODO: I'd like to have these also be objects with from_disk methods
    self.coordinates = self._create_coordinates(
                          path_var['coordinates']['x'].value,
                          path_var['coordinates']['y'].value)
    self.curvature   = path_var['curvature'].value   

    return self
    
  def __repr__(self):
    return utils.print_object(self)  
    
  def __eq__(self,other):
    #TODO: Ensure both are of this class - i.e. check other
    #TODO: Actually implement this
    return self.range == other.range #and \
    #self.duration == other.duration #and \
    #self.coordinates == other.cordinates #and \
    #self.curvature == other.curvature
    
class WormFeatures:
  """ 
    WormFeatures: takes as input a NormalizedWorm instance, and
    during initialization calculates all the features of the worm.
    
    There are two ways to initialize a WormFeatures instance: 
    1. by passing a NormalizedWorm instance and generating the features, or
    2. by loading the already-calculated features from an HDF5 file.
       (via the from_disk method)
    
  """
  def __init__(self, nw):

    if nw is None:
      return

    self.morphology = WormMorphology(nw)
    self.locomotion = WormLocomotion(nw)
    self.posture    = WormPosture(nw)
    #self.path       = WormPath(nw).path
    
  @classmethod  
  def from_disk(cls, file_path):
    
    h = h5py.File(file_path,'r')
    worm = h['worm']
    
    self = cls(None)
    
    self.morphology = WormMorphology.from_disk(worm['morphology'])
    #self.locomotion = WormLocomotion.from_disk(worm['locomotion'])
    #self.posture    = WormPosture.from_disk(worm['posture'])
    self.path = WormPath.from_disk(worm['path'])
    
    return self
    
  def __repr__(self):
    return utils.print_object(self)
    
  def __eq__(self,other):
    """
      Compare two WormFeatures instances by value
    
    """
    return \
      self.path       == other.path       and \
      self.morphology == other.morphology #and \
      #self.posture    == other.posture    and \
      #self.locomotion == other.locomotion and \
      #

        
    
    