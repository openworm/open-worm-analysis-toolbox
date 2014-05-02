# -*- coding: utf-8 -*-
"""
Posture features  ...
"""

import numpy as np
import pdb
import collections
import warnings

class Bends(object):

  def __init__(self,nw):
    
    p = nw.get_partition_subset('normal')
  
    for partition_key in p.keys():
      
      # retrieve the part of the worm we are currently looking at:
      bend_angles = nw.get_partition(partition_key, 'angles')
      
      # shape = (n):
      with warnings.catch_warnings(): #mean empty slice
        temp_mean = np.nanmean(a=bend_angles, axis = 0) 
      with warnings.catch_warnings(): #degrees of freedom <= 0 for slice
        temp_std  = np.nanstd(a=bend_angles, axis = 0)
      
      #Sign the standard deviation (to provide the bend's dorsal/ventral orientation):
      #-------------------------------
      
      pdb.set_trace()
      # First find all entries where the mean is negative
      mask = np.ma.masked_where(condition=temp_mean < 0, a=temp_std).mask
      # Now create a numpy array of -1 where the mask is True and 1 otherwise
      sign_array = -np.ones(np.shape(mask)) * mask + \
                    np.ones(np.shape(mask)) * (~mask)
                    
      # Finally, multiply the std_dev array by our sign_array
      temp_std   = temp_std * sign_array
      
      setattr(self,partition_key,BendSection(temp_mean,temp_std))      
    
class BendSection(object):
  
  def __init__(self,mean,std_dev):
    self.mean    = mean
    self.std_dev = std_dev

def get_amplitude_and_wavelength(theta_d, sx, sy, worm_lengths):
  """
  
  
     Inputs
     =======================================================================
     theta_d      : worm orientation based on fitting to an ellipse, in
                     degrees
     sx           : [49 x num_frames]
     sy           : [49 x num_frames]
     worm_lengths : [1 x num_frames], total length of each worm
  
  
     Output: A dictionary with three elements:
     =======================================================================
     amplitude    :
         .max       - [1 x num_frames] max y deviation after rotating major axis to x-axis
         .ratio     - [1 x num_frames] ratio of y-deviations (+y and -y) with worm centered
                      on the y-axis, ratio is computed to be less than 1
     wavelength   :
         .primary   - [1 x num_frames]
         .secondary - [1 x num_frames] this might not always be valid, even 
                       when the primary wavelength is defined
     track_length  : [1 x num_frames]
  
     
     Old Name: getAmpWavelength.m
     TODO: This function was missing from some of the original code
     distributions. I need to make sure I upload it.
  
  
     Nature Methods Description
     =======================================================================
     Amplitude. 
     ------------------
     Worm amplitude is expressed in two forms: a) the maximum
     amplitude found along the worm body and, b) the ratio of the maximum
     amplitudes found on opposing sides of the worm body (wherein the smaller of
     these two amplitudes is used as the numerator). The formula and code originate
     from the publication “An automated system for measuring parameters of
     nematode sinusoidal movement”6.
     The worm skeleton is rotated to the horizontal axis using the orientation of the
     equivalent ellipse and the skeleton’s centroid is positioned at the origin. The
     maximum amplitude is defined as the maximum y coordinate minus the minimum
     y coordinate. The amplitude ratio is defined as the maximum positive y coordinate
     divided by the absolute value of the minimum negative y coordinate. If the
     amplitude ratio is greater than 1, we use its reciprocal.
  
     Wavelength
     ------------------------
     Wavelength. The worm’s primary and secondary wavelength are computed by
     treating the worm’s skeleton as a periodic signal. The formula and code
     originate from the publication “An automated system for measuring
     parameters of nematode sinusoidal movement”6. The worm’s skeleton is
     rotated as described above for the amplitude. If there are any
     overlapping skeleton points (the skeleton’s x coordinates are not
     monotonically increasing or decreasing in sequence -- e.g., the worm is
     in an S shape) then the shape is rejected, otherwise the Fourier
     transform computed. The primary wavelength is the wavelength associated
     with the largest peak in the transformed data. The secondary wavelength
     is computed as the wavelength associated with the second largest
     amplitude (as long as it exceeds half the amplitude of the primary
     wavelength). The wavelength is capped at twice the value of the worm’s
     length. In other words, a worm can never achieve a wavelength more than
     double its size.
  
     Tracklength
     -----------------------------
     Track Length. The worm’s track length is the range of the skeleton’s
     horizontal projection (as opposed to the skeleton’s arc length) after
     rotating the worm to align it with the horizontal axis. The formula and
     code originate from the publication “An automated system for measuring
     parameters of nematode sinusoidal movement”.
  
  
     Code based on:
     ------------------------------------------------
     BMC Genetics, 2005
     C.J. Cronin, J.E. Mendel, S. Mukhtar, Young-Mee Kim, R.C. Stirb, J. Bruck,
     P.W. Sternberg
     "An automated system for measuring parameters of nematode
     sinusoidal movement" BMC Genetics 2005, 6:5
  
  """
  
  """
  amp_wave_track = \
    collections.namedtuple('amp_wave_track', 
                           ['amplitude', 'wavelength', 'track_length'])
  amp_wave_track.amplitude = 'yay1'
  amp_wave_track.wavelength = 'yay2'
  amp_wave_track.track_length = 'yay3'

  onw = nw.re_orient_and_centre()  


  return amp_wave_track
  """
  pass