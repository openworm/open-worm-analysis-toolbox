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

#http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
from shapely.geometry.polygon import LinearRing
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
  
  
  N_GRID_POINTS = 50 #TODO: Get from config ...
  
  x_range_all = np.ptp(contour_x,axis=0)
  y_range_all = np.ptp(contour_y,axis=0)
  grid_aspect_ratio = x_range_all/y_range_all
  
  #run_mask = np.logical_not(np.isnan(grid_aspect_ratio))

  n_frames = length(x_range_all)
  
  eccentricity    = np.empty(n_frames)
  eccentricity[:] = np.NAN
  orientation     = np.empty(n_frames)
  orientation[:]  = np.NAN

  oh_yeah = 0  
  for iFrame in range(n_frames):
    cur_aspect_ratio = grid_aspect_ratio[iFrame]
    if not np.isnan(cur_aspect_ratio):
      if cur_aspect_ratio > 1:
        #x size is larger so scale down the number of grid points in the y direction
        wtf1 = np.linspace(np.min(contour_x[:,iFrame]), np.max(contour_x[:,iFrame]), num=N_GRID_POINTS);
        wtf2 = np.linspace(np.min(contour_y[:,iFrame]), np.max(contour_y[:,iFrame]), num=np.round(N_GRID_POINTS / cur_aspect_ratio));
      else:
        #y size is larger so scale down the number of grid points in the x direction
        #wtf1 = linspace(min(xOutline_mc(:,iFrame)), max(xOutline_mc(:,iFrame)), round(gridSize * gridAspectRatio));
        #wtf2 = linspace(min(yOutline_mc(:,iFrame)), max(yOutline_mc(:,iFrame)), gridSize);  
    
    #[m,n] = meshgrid( wtf1 , wtf2 );
    
    # get the indices of the points inside of the polygon
    #inPointInds = helper__inpolyNew([m(:) n(:)], [xOutline_mc(:,iFrame) yOutline_mc(:,iFrame)]);
    
    # get the x and y coordinates of the new set of points to be used in calculating eccentricity.
    #x = m(inPointInds);
    #y = n(inPointInds);    
  
    """
        plot(xOutline_mc(:,iFrame),yOutline_mc(:,iFrame),'g-o')
        hold on
        scatter(x,y,'r')
        hold off
        axis equal
        title(sprintf('%d',iFrame))
        pause
    """
    
    #[eccentricity(iFrame),orientation(iFrame)] = h__calculateSingleValues(x,y);  
  
  
  import pdb
  pdb.set_trace()  

  return (eccentricity,orientation)


"""
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
"""