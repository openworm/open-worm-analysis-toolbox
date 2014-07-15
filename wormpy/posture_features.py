# -*- coding: utf-8 -*-
"""
Posture features  ...
"""


from __future__ import division
from . import Events
from . import utils
from . import config
import numpy as np
import warnings
import time
import scipy.ndimage.filters as filters
import collections
from . import feature_helpers

#http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

class Bends(object):

  def __init__(self,nw):
    
    #TODO: I don't like this being in normalized worm
    p = nw.get_partition_subset('normal')
  
    for partition_key in p.keys():
      
      # retrieve the part of the worm we are currently looking at:
      bend_angles = nw.get_partition(partition_key, 'angles')

      #TODO: Should probably merge all three below ...

      # shape = (n):
      with warnings.catch_warnings(record=True): #mean empty slice
        temp_mean = np.nanmean(a=bend_angles, axis = 0)       
        
      with warnings.catch_warnings(record=True): #degrees of freedom <= 0 for slice
        temp_std  = np.nanstd(a=bend_angles, axis = 0)
      
      #Sign the standard deviation (to provide the bend's dorsal/ventral orientation):
      #-------------------------------
      with warnings.catch_warnings(record=True):
        temp_std[temp_mean < 0] *= -1   
      
      setattr(self,partition_key,BendSection(temp_mean,temp_std))      
   
  def __repr__(self):
    return utils.print_object(self)     
   
  @classmethod 
  def from_disk(cls,saved_bend_data):
    
    self = cls.__new__(cls) 
  
    for partition_key in saved_bend_data.keys():
      setattr(self,partition_key,BendSection.from_disk(saved_bend_data[partition_key]))   

    return self
   
class BendSection(object):
  
  def __init__(self,mean,std_dev):
    self.mean    = mean
    self.std_dev = std_dev
   
  @classmethod 
  def from_disk(cls,saved_bend_data):
    
    self = cls.__new__(cls) 

    self.mean   = saved_bend_data['mean'].value
    self.stdDev = saved_bend_data['stdDev'].value
    
    return self
   
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
  
  N_GRID_POINTS = config.N_ECCENTRICITY #TODO: Get from config ...
  
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

def h__centerAndRotateOutlines(x_outline,y_outline):
  """
  #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getEccentricity.m#L391
  """
  pass

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
  
  wwx = sx*np.cos(theta_r)  + sy*np.sin(theta_r)
  wwy = sx*-np.sin(theta_r) + sy*np.cos(theta_r)

  #Subtract mean
  #-----------------------------------------------------------------
  wwx = wwx - np.mean(wwx,axis=0)
  wwy = wwy - np.mean(wwy,axis=0)
  
  # Calculate track amplitude
  #--------------------------------------------------------------------------
  amp1 = np.amax(wwy,axis=0)
  amp2 = np.amin(wwy,axis=0)
  amplitude_max = amp1 - amp2
  amp2 = np.abs(amp2)
  with np.errstate(invalid='ignore'):
    amplitude_ratio = np.divide(np.minimum(amp1,amp2),np.maximum(amp1,amp2))  
  
  
  # Calculate track length
  #--------------------------------------------------------------------------
  #NOTE: This is the x distance after rotation, and is different from the worm
  #length which follows the skeleton. This will always be smaller than the
  #worm length.
  track_length = np.amax(wwx,axis=0) - np.amin(wwx,axis=0)
    
  # Wavelength calculation
  #--------------------------------------------------------------------------
  dwwx = np.diff(wwx,1,axis=0)
  
  #Does the sign change? This is a check to make sure that the change in x is
  #always going one way or the other. Is sign of all differences the same as
  #the sign of the first, or rather, are any of the signs not the same as the
  #first sign, indicating a "bad worm orientation".
  #
  # NOT: This means that within a frame, if the worm x direction changes, then
  # it is considered a bad worm and is not evaluated for wavelength
  #
  
  with np.errstate(invalid='ignore'):
    bad_worm_orientation = np.any(np.not_equal(np.sign(dwwx),np.sign(dwwx[0,:])),axis=0)

  n_frames = bad_worm_orientation.size

  primary_wavelength    = np.zeros(n_frames)
  primary_wavelength[:] = np.NaN  
  secondary_wavelength    = np.zeros(n_frames)
  secondary_wavelength[:] = np.NaN


  #NOTE: Right now this varies from worm to worm which means the spectral
  #resolution varies as well from worm to worm
  spatial_sampling_frequency = (wwx.shape[0]-1)/track_length
  
  ds = 1/spatial_sampling_frequency;

  frames_to_calculate = (np.logical_not(bad_worm_orientation)).nonzero()[0]

  for cur_frame in frames_to_calculate:
    
    #Create an evenly sampled x-axis, note that ds varies
    x1 = wwx[0,cur_frame]
    x2 = wwx[-1,cur_frame]
    if x1 > x2:
      iwwx = utils.colon(x1,-ds[cur_frame],x2)
      iwwy = np.interp(iwwx,wwx[::-1,cur_frame],wwy[::-1,cur_frame])
      iwwy = iwwy[::-1]
    else:
      iwwx = utils.colon(x1,ds[cur_frame],x2)
      iwwy = np.interp(iwwx,wwx[:,cur_frame],wwy[:,cur_frame])
      iwwy = iwwy[::-1]
    
    temp = np.fft.fft(iwwy,N_POINTS_FFT)
       
    if config.MIMIC_OLD_BEHAVIOUR:
      iY = temp[0:HALF_N_FFT]
      iY = iY*np.conjugate(iY)/N_POINTS_FFT
    else:
      iY = np.abs(temp[0:HALF_N_FFT])
      
    #Find peaks that are greater than the cutoff  
    peaks, indx = utils.separated_peaks(iY, MIN_DIST_PEAKS,True,WAVELENGTH_PCT_MAX_CUTOFF*np.amax(iY))  
      
    # This is what the supplemental says, not what was done in the previous
    # code. I'm not sure what was done for the actual paper, but I would
    # guess they used power.
    #
    # This gets used when determining the secondary wavelength, as it must
    # be greater than half the maximum to be considered a secondary
    # wavelength.
    
    # NOTE: True Amplitude = 2*abs(fft)/(length_real_data i.e. 48 or 49, not 512)
    #
    # i.e. for a sinusoid of a given amplitude, the above formula would give
    # you the amplitude of the sinusoid
  
    # We sort the peaks so that the largest is at the first index and will
    # be primary, this was not done in the previous version of the code
    I = np.argsort(-1*peaks)
    indx = indx[I]

    frequency_values = (indx - 1)/N_POINTS_FFT*spatial_sampling_frequency[cur_frame]
    
    all_wavelengths = 1/frequency_values
    
    p_temp = all_wavelengths[0]
    
    if indx.size > 1:
      s_temp = all_wavelengths[1]
    else:
      s_temp = np.NaN
      
    worm_wavelength_max = WAVELENGTH_PCT_CUTOFF*worm_lengths[cur_frame]
    
    # Cap wavelengths ...
    if p_temp > worm_wavelength_max:
      p_temp = worm_wavelength_max
      
    
    # ??? Do we really want to keep this as well if p_temp == worm_2x?
    # i.e., should the secondary wavelength be valid if the primary is also
    # limited in this way ?????
    if s_temp > worm_wavelength_max:
        s_temp = worm_wavelength_max

    primary_wavelength[cur_frame] = p_temp
    secondary_wavelength[cur_frame] = s_temp

  if config.MIMIC_OLD_BEHAVIOUR:
    # Suppress warnings so we can compare a numpy array that may contain NaNs
    # without triggering a Runtime Warning
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      mask = secondary_wavelength > primary_wavelength

    temp = secondary_wavelength[mask]
    secondary_wavelength[mask] = primary_wavelength[mask]
    primary_wavelength[mask] = temp
            
  amp_wave_track = \
    collections.namedtuple('amp_wave_track', 
                           ['amplitude_max', 'amplitude_ratio', 'primary_wavelength', 
                            'secondary_wavelength', 'track_length'])
                            
  amp_wave_track.amplitude_max   = amplitude_max
  amp_wave_track.amplitude_ratio = amplitude_ratio 
  amp_wave_track.primary_wavelength   = primary_wavelength
  amp_wave_track.secondary_wavelength = secondary_wavelength  
  amp_wave_track.track_length = track_length
  
  return amp_wave_track

"""

Old Vs New Code:
  - power instead of magnitude is used for comparison
  - primary and secondary wavelength may be switched ...
  - error in maxPeaksDist for distance threshold, not sure where in code
        - see frame 880 for example
        - minus 1 just gives new problem - see 1794

"""

def get_worm_kinks(bend_angles):
  #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getWormKinks.m

  # Determine the bend segment length threshold.
  n_angles = bend_angles.shape[0]
  length_threshold = np.round(n_angles*config.KINK_LENGTH_THRESHOLD_PCT)  
  
  # Compute a gaussian filter for the angles.
  #--------------------------------------------------------------------------
  #JAH NOTE: This is a nice way of getting the appropriate odd value
  #unlike the other code with so many if statements ...
  #- see window code which tries to get an odd value ...
  #- I'd like to go back and fix that code ...
  half_length_thr = np.round(length_threshold / 2)
  gauss_filter    = feature_helpers.gausswin(half_length_thr * 2 + 1) \
                    / half_length_thr
  
  # Compute the kinks for the worms.
  n_frames       = bend_angles.shape[1]
  n_kinks_all    = np.zeros(n_frames,dtype=float)
  n_kinks_all[:] = np.NaN

  #(np.any(np.logical_or(mask_pos,mask_neg),axis=0)).nonzero()[0]

  nan_mask = np.isnan(bend_angles)

  for iFrame in (~np.all(nan_mask,axis=0)).nonzero()[0]:
    smoothed_bend_angles = filters.convolve1d(bend_angles[:,iFrame],gauss_filter,cval=0,mode='constant')
  
    #This code is nearly identical in getForaging
    #-------------------------------------------------------
    n_frames = smoothed_bend_angles.shape[0]

    with np.errstate(invalid='ignore'):
      dataSign = np.sign(smoothed_bend_angles)
    
    if np.any(np.equal(dataSign,0)):
        #I don't expect that we'll ever actually reach 0
        #The code for zero was a bit weird, it keeps counting if no sign
        #change i.e. + + + 0 + + + => all +
        #
        #but if counts for both if sign change
        # + + 0 - - - => 3 +s and 4 -s    
        raise Exception("Unhandled code case")
    
    sign_change_I = (np.not_equal(dataSign[1:],dataSign[0:-1])).nonzero()[0]

    end_I   = np.concatenate((sign_change_I,n_frames*np.ones(1,dtype=np.result_type(sign_change_I))))
    
    wtf1    = np.zeros(1,dtype=np.result_type(sign_change_I))
    wtf2    = sign_change_I+1
    start_I = np.concatenate((wtf1,wtf2)) #+2? due to inclusion rules???

    #All NaN values are considered sign changes, remove these ...
    keep_mask = np.logical_not(np.isnan(smoothed_bend_angles[start_I]))

    start_I = start_I[keep_mask]
    end_I   = end_I[keep_mask]
    
    #The old code had a provision for having NaN values in the middle
    #of the worm. I have not translated that feature to the newer code. I
    #don't think it will ever happen though for a valid frame, only on the
    #edges should you have NaN values.
    if start_I.size != 0 and np.any(np.isnan(smoothed_bend_angles[start_I[0]:end_I[-1]])):
       raise Exception("Unhandled code case")
       
    #-------------------------------------------------------
    #End of identical code ...
    
    lengths = end_I - start_I + 1

    #Adjust lengths for first and last:
    #Basically we allow NaN values to count towards the length for the
    #first and last stretches
    if lengths.size != 0:
       if start_I[0] != 0: #Due to leading NaNs
          lengths[0] = end_I[0] + 1
       if end_I[-1] != n_frames: #Due to trailing NaNs
          lengths[-1] = n_frames - start_I[-1]
    
    n_kinks_all[iFrame] = np.sum(lengths >= length_threshold)
    
  return n_kinks_all

def get_worm_coils(frame_codes,midbody_distance):
  
  #This function is very reliant on the MRC processor  
  
  #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getCoils.m
  
  
  COIL_FRAME_THRESHOLD = np.round(1/5 * config.FPS)  
  COIL_START_CODES = [105, 106];
  FRAME_SEGMENTED  = 1 #Go back 1 frame, this is the end of the coil ...  
  
  #Algorithm: Whenever a new start is found, find the first segmented frame, 
  #that's the end.

  #Add on a frame to allow closing a coil at the end ...
  coil_start_mask = (frame_codes == COIL_START_CODES[0]) | (frame_codes == COIL_START_CODES[1])
  np_false = np.zeros((1,),dtype=bool)
  coil_start_mask = np.concatenate((coil_start_mask,np_false))
  
  
  #NOTE: These are not guaranteed ends, just possible ends ...
  end_coil_mask = frame_codes == FRAME_SEGMENTED
  np_true = ~np_false
  end_coil_mask = np.concatenate((end_coil_mask,np_true)) 
  
  in_coil = False
  coil_frame_start = -1
  n_coils = 0
  n_frames_plus1 = len(frame_codes) + 1
  
  starts = []
  ends   = []  
  
  for iFrame in range(n_frames_plus1):
    if in_coil:
      if end_coil_mask[iFrame]:
        n_coil_frames = iFrame - coil_frame_start
        if n_coil_frames >= COIL_FRAME_THRESHOLD:
          n_coils += 1
          
          starts.append(coil_frame_start)
          ends.append(iFrame-1)
        
        in_coil = False
    elif coil_start_mask[iFrame]:
      in_coil = True
      coil_frame_start = iFrame
  
  
  if config.MIMIC_OLD_BEHAVIOUR:
    if (len(starts) > 0) & (ends[-1] == len(frame_codes)-1):
      ends[-1]   += -1
      starts[-1] += -1
       
        
  temp = Events.EventList(np.transpose(np.vstack((starts, ends))))
  
  return Events.EventListWithFeatures(temp, midbody_distance)
  
  """
  coiled_frames = h__getWormTouchFrames(frame_codes, config.FPS);

  COIL_FRAME_THRESHOLD = np.round(1/5 * config.FPS);
  """
  
  
  
  #Algorithm: Whenever a new start is found, find the first segmented frame, 
  #that's the end.
  
  #Add on a frame to allow closing a coil at the end ...
  
#  pdb.set_trace()    # DEBUG
  
  """
  coil_start_mask = [frameCodes == COIL_START_CODES(1) | frameCodes == COIL_START_CODES(2) false];
  
  #NOTE: These are not guaranteed ends, just possible ends ...
  end_coil_mask   = [frameCodes == FRAME_SEGMENTED true];
  
  in_coil = false;
  coil_frame_start = 0;
  
  n_coils = 0;
  
  n_frames_p1 = length(frameCodes) + 1;
  
  for iFrame = 1:n_frames_p1
      if in_coil
          if end_coil_mask(iFrame)
              
              n_coil_frames = iFrame - coil_frame_start;
              if n_coil_frames >= COIL_FRAME_THRESHOLD
                  n_coils = n_coils + 1;
                  
                  touchFrames(n_coils).start = coil_frame_start; 
                  touchFrames(n_coils).end   = iFrame - 1;
              end
              in_coil = false;
          end
      elseif coil_start_mask(iFrame)
          in_coil = true;
          coil_frame_start = iFrame;
      end
  end
  
  
  
  
  
  if d_opts.mimic_old_behavior
      if ~isempty(coiled_frames) && coiled_frames(end).end == length(frame_codes)
         coiled_frames(end).end   = coiled_frames(end).end - 1;
         coiled_frames(end).start = coiled_frames(end).start - 1;
      end
  end
  
  coiled_events = seg_worm.feature.event(coiled_frames,FPS,midbody_distance,DATA_NAME,INTER_DATA_NAME);
  
  return coiled_events.getFeatureStruct;
   
  """

  return None   
   
class Directions(object):
  
  """

  tail2head
  head
  tail
  
  """
  
  def __init__(self,sx,sy,wp):
    
    """
    
    wp : (worm paritions) from normalized worm, 
    
    """
        
    #These are the names of the final fields
    NAMES = ['tail2head', 'head', 'tail']
    
    #For each set of indices, compute the centroids of the tip and tail then
    #compute a direction vector between them (tip - tail)

    TIP_I  = [wp['head'], wp['head_tip'], wp['tail_tip']] #I - "indices" - really a tuple of start,stop
    TAIL_I = [wp['tail'], wp['head_base'], wp['tail_base']]

    TIP_S  = [slice(*x) for x in TIP_I] #S - slice
    TAIL_S = [slice(*x) for x in TAIL_I]
      
    for iVector in range(3):
      tip_x  = np.mean(sx[TIP_S[iVector], :],axis=0)
      tip_y  = np.mean(sy[TIP_S[iVector], :],axis=0)
      tail_x = np.mean(sx[TAIL_S[iVector], :],axis=0)
      tail_y = np.mean(sy[TAIL_S[iVector], :],axis=0)
      
      dir_value = 180/np.pi*np.arctan2(tip_y - tail_y, tip_x - tail_x)
      setattr(self,NAMES[iVector],dir_value)
  
  @classmethod 
  def from_disk(cls,data):
    
    self = cls.__new__(cls)
    
    for key in data:
      setattr(self,key,data[key])
    
    return self
    
        
  def __repr__(self):
    return utils.print_object(self)         
      
def get_eigenworms(sx,sy,eigen_worms,N_EIGENWORMS_USE):
  """
  
  Parameters:
  -----------
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
  
  # DEBUG: hiding this error for now - @MichaelCurrie
  return None # np.dot(eigen_worms[0:N_EIGENWORMS_USE,:],angles)
  
