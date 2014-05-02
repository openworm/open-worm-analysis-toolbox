# -*- coding: utf-8 -*-
"""

"""

from . import utils
import numpy as np
from . import feature_helpers
from . import config

class Range:

  """
  Attributes
  ------------------
  value :
  
  """

  def __init__(self,contour_x = None,contour_y = None):

    if contour_x is None:
      return

    #Get average per frame
    #------------------------------------------------
    mean_cx = contour_x.mean(axis=0)
    mean_cy = contour_y.mean(axis=0)
   
    #Average over all frames for subtracting
    #-------------------------------------------------
    x_centroid_cx = np.nanmean(mean_cx)
    y_centroid_cy = np.nanmean(mean_cy)
   
    self.value    =  np.sqrt((mean_cx - x_centroid_cx)**2 + (mean_cy - y_centroid_cy)**2)


  @staticmethod 
  def from_disk(path_var):
    
    #path_features.Range.from_disk(path_var)    
    
    temp = Range(None)
    
    #NOTE: This is of size nx1 for Matlab versions, might want to fix on loading
    #
    temp.value = np.squeeze(path_var['range'].value)
    
    return temp
    
  def __repr__(self):
    return utils.print_object(self)
    
  def __eq__(self,other):
    value1 = self.value
    value2 = other.value
 
 
    #TODO: This code should be moved elsewhere ...
    mask = ~(np.logical_or(np.isnan(value1),np.isnan(value2)))
    
    corr_mat = np.corrcoef(value1[mask],value2[mask])

    c_value = corr_mat[1][0]    
    
    return c_value > 0.99    

class Duration:

  """
  Attributes:
  --------------------------------------
  arena :
  worm  :
  head  :
  midbody :
  tail :
  
  """

  def __init__(self, nw=None, sx=None, sy=None, widths=None, fps=None):
    
    if nw is None:
      return
    
    s_points = [nw.worm_partitions[x] for x in ('all', 'head', 'body', 'tail')]    

    #TODO: d_opts not currently used
    #-------------------------------------------------------------------------
    #This is for the old version via d_opts, this is currently not used
    #i.e. if d_opts.mimic_old_behavior   #Then do the following ...
    #    s_points_temp = {SI.HEAD_INDICES SI.MID_INDICES SI.TAIL_INDICES};
    #
    #    all_widths = zeros(1,3);
    #    for iWidth = 1:3
    #        temp = widths(s_points_temp{iWidth},:);
    #        all_widths(iWidth) = nanmean(temp(:));
    #    end
    #    mean_width = mean(all_widths);    
    #end

    #Return early if necessary
    #------------------------------------------------------------------------
    if len(sx) == 0 or np.isnan(sx).all():
      raise Exception('This code is not yet translated')
      
      #ar = Arena(create_null = True)      
      
      #    NAN_cell  = repmat({NaN},1,n_points);
      #     durations = struct('indices',NAN_cell,'times',NAN_cell);  
      #    obj.duration = h__buildOutput(arena,durations);
      #    return;  
     
    mean_width = np.nanmean(widths)
    scale      = 2.0**0.5/mean_width;
     
    # Scale the skeleton and translate so that the minimum values are at 1
    #-------------------------------------------------------------------------
    with np.errstate(invalid='ignore'):
      scaled_sx = np.round(sx*scale)
      scaled_sy = np.round(sy*scale)  
  
    x_scaled_min = np.nanmin(scaled_sx)
    x_scaled_max = np.nanmax(scaled_sx)
    y_scaled_min = np.nanmin(scaled_sy)
    y_scaled_max = np.nanmax(scaled_sy)
   
    #Unfortunately needing to typecast to int for array indexing also
    #removes my ability to identify invalid values :/
    #Thus we precompute invalid values and then cast
    isnan_mask = np.isnan(scaled_sx) 
   
    scaled_zeroed_sx = (scaled_sx - x_scaled_min).astype(int)
    scaled_zeroed_sy = (scaled_sy - y_scaled_min).astype(int)     
    
    arena_size  = [y_scaled_max - y_scaled_min + 1, x_scaled_max - x_scaled_min + 1]    
    ar = Arena(sx, sy, arena_size)
  
    #--------------------------------------------------------------------------
    def h__populateArenas(arena_size, sys, sxs, s_points, isnan_mask):
      """
  
      Attributes:
      ----------------------------
      arena_size: list
        [2]
      sys : numpy.int32
        [49, n_frames]
      sxs : numpy.int32
        [49, n_frames]
      s_points: list
        [4]
      isnan_mask: bool
        [49, n_frames]
        
      
      """
      
      #NOTE: All skeleton points have been rounded to integer values for
      #assignment to the matrix based on their values being treated as indices
  
  
      #Filter out frames which have no valid values
      #----------------------------------------------------------
      frames_run   = np.flatnonzero(np.any(~isnan_mask,axis=0))
      n_frames_run = len(frames_run)
       
      #1 area for each set of skeleton indices
      #-----------------------------------------
      n_points = len(s_points)
      arenas   = [None]*n_points
      
      #Loop over the different regions of the body
      #------------------------------------------------
      for iPoint in range(n_points):
             
        temp_arena = np.zeros(arena_size)
        s_indices  = s_points[iPoint]
              
        #For each frame, add +1 to the arena each time a chunk of the skeleton
        #is located in that part
        #----------------------------------------------------------------
        for iFrame in range(n_frames_run):
          cur_frame = frames_run[iFrame]
          cur_x     = sxs[s_indices[0]:s_indices[1],cur_frame]
          cur_y     = sys[s_indices[0]:s_indices[1],cur_frame]
          temp_arena[cur_y,cur_x] += 1
      
        arenas[iPoint] = temp_arena[::-1,:] #FLip axis to maintain
        #consistency with Matlab
      
      return arenas
    #----------------------------------------------------------------------------  
    
    temp_arenas   = h__populateArenas(arena_size, scaled_zeroed_sy, scaled_zeroed_sx, s_points, isnan_mask)  

    #For looking at the data
    #------------------------------------
    #utils.imagesc(temp_arenas[0])

    temp_duration = [DurationElement(x,fps) for x in temp_arenas]

    self.arena   = ar
    self.worm    = temp_duration[0]
    self.head    = temp_duration[1]
    self.midbody = temp_duration[2]
    self.tail    = temp_duration[3]

  def __repr__(self):
    return utils.print_object(self)

  @staticmethod 
  def from_disk(duration_group):
    
    #path_features.Duration.from_disk(path_var)    
    
    temp = Duration(None)
    temp.arena    = Arena.from_disk(duration_group['arena'])
    temp.worm     = DurationElement.from_disk(duration_group['worm'])
    temp.head     = DurationElement.from_disk(duration_group['head'])
    temp.midbody  = DurationElement.from_disk(duration_group['midbody'])
    temp.tail     = DurationElement.from_disk(duration_group['tail'])

    return temp


class DurationElement:
  
  def __init__(self,arena_coverage=None,fps=None):

    if arena_coverage is None:
      return
    
    #transpose groups results by element rather than by dimension
    self.indices = np.transpose(np.nonzero(arena_coverage))
    self.times   = arena_coverage[self.indices[:,0],self.indices[:,1]]/fps

  def __repr__(self):
    return utils.print_object(self)
   
  @staticmethod 
  def from_disk(saved_duration_elem):
    temp = DurationElement(None)
    temp.indices = saved_duration_elem['indices'].value
    temp.times   = saved_duration_elem['times'].value
    
class Arena:
   
  def __init__(self, sx=None, sy=None, arena_size=None, create_null=False):
    
    if sx is None:
      return
    
    if create_null:
      self.height = np.nan
      self.width  = np.nan
      self.min_x  = np.nan
      self.min_y  = np.nan
      self.max_x  = np.nan
      self.max_y  = np.nan
    else:
      # Construct the empty arena(s).
        
      self.height = arena_size[0]
      self.width  = arena_size[1]
      self.min_x  = np.nanmin(sx)
      self.min_y  = np.nanmin(sy)
      self.max_x  = np.nanmax(sx)
      self.max_y  = np.nanmax(sy)    
    
  def __repr__(self):
    return utils.print_object(self)
    
  @staticmethod 
  def from_disk(saved_arena_elem):
    temp = Arena(None)
    temp.height = saved_arena_elem['height'].value
    temp.width  = saved_arena_elem['width'].value
    temp.min_x  = saved_arena_elem['min']['x'].value
    temp.min_y  = saved_arena_elem['min']['y'].value
    temp.max_x  = saved_arena_elem['max']['x'].value
    temp.max_y  = saved_arena_elem['max']['y'].value      
    
    return temp

def worm_path_curvature(x,y,fps,ventral_mode):
  
  """
  
  
  """
  
  #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40path/wormPathCurvature.m  
  
  BODY_I    = slice(44,3,-1)
  
#slice(*BODY_I)  
  
  #This was nanmean but I think mean will be fine. nanmean was
  #causing the program to crash
  diff_x = np.mean(np.diff(x[BODY_I,:],axis=0),axis=0)
  diff_y = np.mean(np.diff(y[BODY_I,:],axis=0),axis=0)
  avg_body_angles_d = np.arctan2(diff_y,diff_x)*180/np.pi  
  
  #compute_velocity - inputs don't make sense ...
  #???? - sample_time??
  #???? - bodyI, BODY_DIFF, 
  speed, motion_direction = feature_helpers.compute_velocity(x, y, avg_body_angles_d, config.BODY_DIFF, ventral_mode)

  frame_scale      = feature_helpers.get_frames_per_sample(config.BODY_DIFF)
  half_frame_scale = (frame_scale - 1) / 2

  #Compute the angle differentials and distances.
  speed = abs(speed);

  #At each frame, we'll compute the differences in motion direction using 
  #some frame in the future relative to the current frame
  #
  #i.e. diff_motion[current_frame] = motion_direction[current_frame + frame_scale] - motion_direction[current_frame]
  #------------------------------------------------
  diff_motion    = np.empty(speed.shape)
  diff_motion[:] = np.NAN
  
  right_max_I = len(diff_motion) - frame_scale
  diff_motion[0:right_max_I] = motion_direction[frame_scale:] - motion_direction[0:right_max_I]

  with np.errstate(invalid='ignore'):
    diff_motion[diff_motion >= 180]  -= 360;
    diff_motion[diff_motion <= -180] += 360;
  
  distance_I_base    = slice(half_frame_scale,-(frame_scale+1),1)
  distance_I_shifted = slice(half_frame_scale + frame_scale,-1,1)  
    
  distance    = np.empty(speed.shape)
  distance[:] = np.NaN

  distance[distance_I_base] = speed[distance_I_base] + speed[distance_I_shifted]*config.BODY_DIFF/2
  
  with np.errstate(invalid='ignore'):
    distance[distance < 1] = np.NAN
  
  return (diff_motion/distance) * (np.pi/180);    