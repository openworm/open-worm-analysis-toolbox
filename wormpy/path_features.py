# -*- coding: utf-8 -*-
"""

"""

import numpy as np

class Duration:

  """
  Attributes:
  --------------------------------------
    
  
  
  """

  def __init__(self, nw, sx, sy, widths, fps):
    
    s_points = [nw.worm_partitions[x] for x in ('all', 'head', 'body', 'tail')]    
    n_points = len(s_points)

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

    if len(sx) == 0 or np.isnan(sx).all():
      ar = Arena(create_null = True)      
      raise Exception('This code is not yet translated')
      #    NAN_cell  = repmat({NaN},1,n_points);
      #     durations = struct('indices',NAN_cell,'times',NAN_cell);  
      #    obj.duration = h__buildOutput(arena,durations);
      #    return;  
     
    mean_width = np.nanmean(widths)
    scale      = 2.0**0.5/mean_width;
     
    # Scale the skeleton and translate so that the minimum values are at 1
    #-------------------------------------------------------------------------
    #NOTE: These will throw warnings if NaN are created :/ , thanks Python
    scaled_sx = np.round(sx*scale)  #NOTE: I added the 1 just to avoid overwriting
    scaled_sy = np.round(sy*scale)  #Ideally these would be named better
  
    x_scaled_min = np.nanmin(scaled_sx)
    x_scaled_max = np.nanmax(scaled_sx)
    y_scaled_min = np.nanmin(scaled_sy)
    y_scaled_max = np.nanmax(scaled_sy)
   
    #Unfortunately needing to typecast to int for array indexing also
    #removes my ability to identify invalid values :/
    isnan_mask = np.isnan(scaled_sx) 
   
    scaled_zeroed_sx = (scaled_sx - x_scaled_min).astype(int)
    scaled_zeroed_sy = (scaled_sy - y_scaled_min).astype(int)     
    
    arena_size  = [y_scaled_max - y_scaled_min + 1, x_scaled_max - x_scaled_min + 1]    
    ar = Arena(sx, sy, arena_size)
  

  
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

    #********************************************************
    #JAH TODO: AT THIS POINT
    #********************************************************

     
#  n_points = len(s_points)      
#
#  temp_duration = [None]*n_points   
#
#  for iPoint in range(n_points): 
#    d = duration_element()
#    d.indices = np.transpose(np.nonzero(arenas[iPoint]))
#    d.times   = arenas[iPoint][d.indices[:,0],d.indices[:,1]]/fps      
#    temp_duration[iPoint] = d
#
#  d_out = durations()
#  d_out.arena   = ar
#  d_out.worm    = temp_duration[0]
#  d_out.head    = temp_duration[1]
#  d_out.midbody = temp_duration[2]
#  d_out.tail    = temp_duration[3]
#
#  return d_out





class DurationElement:
  
  def __init__(self):
    self.indices = []
    self.times   = []
    
class Arena:
   
  def __init__(self, sx, sy, arena_size, create_null = False):
    
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
    
    