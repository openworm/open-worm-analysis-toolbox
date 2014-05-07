# -*- coding: utf-8 -*-
"""
Posture features  ...
"""

from __future__ import division
from . import utils
import numpy as np
import pdb
import collections
import warnings
import time

#http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPoint
from shapely.geometry import Point

class Bends(object):

  def __init__(self,nw):
    
    p = nw.get_partition_subset('normal')
  
    for partition_key in p.keys():
      
      # retrieve the part of the worm we are currently looking at:
      bend_angles = nw.get_partition(partition_key, 'angles')

      #TODO: Should probably merge all three below ...

      # shape = (n):
      with warnings.catch_warnings(record=True) as w: #mean empty slice
        temp_mean = np.nanmean(a=bend_angles, axis = 0)       
        
      with warnings.catch_warnings(record=True) as w: #degrees of freedom <= 0 for slice
        temp_std  = np.nanstd(a=bend_angles, axis = 0)
      
      #Sign the standard deviation (to provide the bend's dorsal/ventral orientation):
      #-------------------------------
      with warnings.catch_warnings(record=True) as w:
        temp_std[temp_mean < 0] *= -1   
      
      setattr(self,partition_key,BendSection(temp_mean,temp_std))      
   
  def __repr__(self):
    return utils.print_object(self)     
   
class BendSection(object):
  
  def __init__(self,mean,std_dev):
    self.mean    = mean
    self.std_dev = std_dev
    
  def __repr__(self):
    return utils.print_object(self)

def get_eccentricity_and_orientation(contour_x, contour_y):
  """
    get_eccentricity   
   
      [eccentricity, orientation] = seg_worm.feature_helpers.posture.getEccentricity(xOutline, yOutline, gridSize)
   
      Given x and y coordinates of the outline of a region of interest, fill
      the outline with a grid of evenly spaced points and use these points in
      a center of mass calculation to calculate the eccentricity and
      orientation of the equivalent ellipse.
   
      Placing points in the contour is a well known computer science problem
      known as the Point-in-Polygon problem.
   
      http://en.wikipedia.org/wiki/Point_in_polygon
   
      This function became a lot more complicated in an attempt to make it 
      go much faster. The complication comes from the simplication that can
      be made when the worm doesn't bend back on itself at all.
   
   
      OldName: getEccentricity.m
    
   
      Inputs:
      =======================================================================
      xOutline : [96 x num_frames] The x coordinates of the contour. In particular the contour
                  starts at the head and goes to the tail and then back to
                  the head (although no points are redundant)
      yOutline : [96 x num_frames]  The y coordinates of the contour "  "
      
      N_ECCENTRICITY (a constant from config.py):
                 (scalar) The # of points to place in the long dimension. More points
                 gives a more accurate estimate of the ellipse but increases
                 the calculation time.
   
      Outputs: a namedtuple containing:
      =======================================================================
      eccentricity - [1 x num_frames] The eccentricity of the equivalent ellipse
      orientation  - [1 x num_frames] The orientation angle of the equivalent ellipse
   
      Nature Methods Description
      =======================================================================
      Eccentricity. 
      ------------------
      The eccentricity of the worm’s posture is measured using
      the eccentricity of an equivalent ellipse to the worm’s filled contour.
      The orientation of the major axis for the equivalent ellipse is used in
      computing the amplitude, wavelength, and track length (described
      below).
   
      Status
      =======================================================================
      The code below is finished although I want to break it up into smaller
      functions. I also need to submit a bug report for the inpoly FEX code.

  Translation of: SegwormMatlabClasses / 
  +seg_worm / +feature_helpers / +posture / getEccentricity.m
  """
  
  t_obj = time.time()
  
  N_GRID_POINTS = 50 #TODO: Get from config ...
  
  x_range_all       = np.ptp(contour_x,axis=0)
  y_range_all       = np.ptp(contour_y,axis=0)
  
  x_mc = contour_x - np.mean(contour_x,axis=0) #mc - mean centered
  y_mc = contour_y - np.mean(contour_y,axis=0)  
  
  grid_aspect_ratio = x_range_all/y_range_all
  
  #run_mask = np.logical_not(np.isnan(grid_aspect_ratio))

  n_frames = len(x_range_all)
  
  eccentricity    = np.empty(n_frames)
  eccentricity[:] = np.NAN
  orientation     = np.empty(n_frames)
  orientation[:]  = np.NAN
 
  #h__getEccentricityAndOrientation
  for iFrame in range(n_frames):
    cur_aspect_ratio = grid_aspect_ratio[iFrame]

    
    #------------------------------------------------------
    if not np.isnan(cur_aspect_ratio):
      
      cur_cx = x_mc[:,iFrame]
      cur_cy = y_mc[:,iFrame]
      poly = Polygon(zip(cur_cx,cur_cy))     
      
      if cur_aspect_ratio > 1:
        #x size is larger so scale down the number of grid points in the y direction
        n1 = N_GRID_POINTS
        n2 = np.round(N_GRID_POINTS / cur_aspect_ratio)
      else:
        #y size is larger so scale down the number of grid points in the x direction        
        n1 = np.round(N_GRID_POINTS * cur_aspect_ratio)
        n2 = N_GRID_POINTS
    
    
      wtf1 = np.linspace(np.min(x_mc[:,iFrame]), np.max(x_mc[:,iFrame]), num=n1);
      wtf2 = np.linspace(np.min(y_mc[:,iFrame]), np.max(y_mc[:,iFrame]), num=n2);    
    
      m,n = np.meshgrid( wtf1 , wtf2 );


    
      n_points = m.size
      m_lin    = m.reshape(n_points)
      n_lin    = n.reshape(n_points)  
      in_worm  = np.zeros(n_points,dtype=np.bool)
      for i in range(n_points):
        p = Point(m_lin[i],n_lin[i])
#        try:
        in_worm[i] = poly.contains(p)
#        except ValueError:
#          import pdb
#          pdb.set_trace()
        
      
        x = m_lin[in_worm]
        y = n_lin[in_worm]
      
      """
        TODO: Finish this
        plot(xOutline_mc(:,iFrame),yOutline_mc(:,iFrame),'g-o')
        hold on
        scatter(x,y,'r')
        hold off
        axis equal
        title(sprintf('%d',iFrame))
        pause
      """
    
    
      #First eccentricity value should be: 0.9743

      #h__calculateSingleValues
      N = float(len(x))
      # Calculate normalized second central moments for the region.
      uxx = np.sum(x*x)/N
      uyy = np.sum(y*y)/N
      uxy = np.sum(x*y)/N
  
      # Calculate major axis length, minor axis length, and eccentricity.
      common               = np.sqrt((uxx - uyy)**2 + 4*(uxy**2))
      majorAxisLength      = 2*np.sqrt(2)*np.sqrt(uxx + uyy + common)
      minorAxisLength      = 2*np.sqrt(2)*np.sqrt(uxx + uyy - common)
      eccentricity[iFrame] = 2*np.sqrt((majorAxisLength/2)**2 - (minorAxisLength/2)**2) / majorAxisLength
  
      # Calculate orientation.
      if (uyy > uxx):
        num = uyy - uxx + np.sqrt((uyy - uxx)**2 + 4*uxy**2)
        den = 2*uxy
      else:
        num = 2*uxy
        den = uxx - uyy + np.sqrt((uxx - uyy)**2 + 4*uxy**2)
  
      orientation[iFrame] = (180/np.pi) * np.arctan(num/den)

    #[eccentricity(iFrame),orientation(iFrame)] = h__calculateSingleValues(x,y);  
  
  
  elapsed_time = time.time() - t_obj
  print('Elapsed time in seconds for eccentricity: %d' % elapsed_time)
  
  return (eccentricity,orientation)

def get_amplitude_and_wavelength(theta_d, sx, sy, worm_lengths):

  #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getAmplitudeAndWavelength.m
  
  N_POINTS_FFT   = 512
  HALF_N_FFT     = N_POINTS_FFT/2
  MIN_DIST_PEAKS = 5  
  WAVELENGTH_PCT_MAX_CUTOFF = 0.5 #TODO: Describe
  WAVELENGTH_PCT_CUTOFF     = 2
  
  #TODO: Write in Python
  #assert(size(sx,1) <= N_POINTS_FFT,'# of points used in the FFT must be more than the # of points in the skeleton')  
  
  theta_r = theta_d*(np.pi/180);  
  
  #Unrotate worm
  #------------------------------
  
  import pdb
  pdb.set_trace()
  
  
  
  
  
  
  amp_wave_track = \
    collections.namedtuple('amp_wave_track', 
                           ['amplitude', 'wavelength', 'track_length'])
  amp_wave_track.amplitude = 'yay1'
  amp_wave_track.wavelength = 'yay2'
  amp_wave_track.track_length = 'yay3'

  onw = nw.re_orient_and_centre()  

  return amp_wave_track
  
  
def get_eigenworms(sx,sy,eigen_worms,N_EIGENWORMS_USE):
  
  """
  
  Parameters:
  ---------------------------------
  eigen_worms: [7,48]  

  """  

  angles   = np.arctan2(np.diff(sy,n=1,axis=0),np.diff(sx,n=1,axis=0))

  n_frames = sx.shape[1]
  
  # need to deal with cases where angle changes discontinuously from -pi
  # to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
  # respectively to all remaining points.  This effectively extends the
  # range outside the -pi to pi range.  Everything is re-centred later
  # when we subtract off the mean.
  false_row = np.zeros((1,n_frames),dtype=bool)
  
  #NOTE: By adding the row of falses, we shift the trues
  #to the next value, which allows indices to match. Otherwise after every
  #find statement we would need to add 1, I think this is a bit faster ...
  
  with np.errstate(invalid='ignore'):
    mask_pos = np.concatenate((false_row,np.diff(angles,n=1,axis=0) > np.pi),axis=0) 
    mask_neg = np.concatenate((false_row,np.diff(angles,n=1,axis=0) < -np.pi),axis=0)   

  #Only fix the frames we need to, in which there is a jump in going from one
  #segment to the next ...
  fix_frames_I = (np.any(np.logical_or(mask_pos,mask_neg),axis=0)).nonzero()[0]
    
  for cur_frame in fix_frames_I:
    
    positive_jump_I = (mask_pos[:,cur_frame]).nonzero()[0]
    negative_jump_I = (mask_neg[:,cur_frame]).nonzero()[0]
  
    # subtract 2pi from remainging data after positive jumps
    # Note that the jumps impact all subsequent frames
    for cur_pos_jump in positive_jump_I:
      angles[cur_pos_jump:,cur_frame] -= 2*np.pi
      
    # add 2pi to remaining data after negative jumps
    for cur_neg_jump in negative_jump_I:
      angles[cur_neg_jump:,cur_frame] += 2*np.pi

  angles = angles - np.mean(angles,axis=0)  
  
  return np.dot(eigen_worms[0:N_EIGENWORMS_USE,:],angles)