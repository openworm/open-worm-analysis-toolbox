# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import numpy as np
import pdb

#Training wheels for Jim :/
def scatter(x,y):
  plt.scatter(x,y)
  plt.show()

def plotxy(x,y):
  plt.plot(x,y)
  plt.show()

def plotx(data):
  plt.plot(data)
  plt.show()

def imagesc(data):
  #http://matplotlib.org/api/pyplot_api.html?highlight=imshow#matplotlib.pyplot.imshow
  plt.imshow(data,aspect='auto')
  plt.show()

def max_peaks_dist(x, dist, use_max, value_cutoff):
  """
  MAXPEAKSDIST Find the maximum peaks in a vector. The peaks
  are separated by, at least, the given distance.
  
     seg_worm.util.maxPeaksDist
     
     [PEAKS INDICES] = seg_worm.util.maxPeaksDist(x, dist,use_max,value_cutoff,*chain_code_lengths)
  
     Inputs:
         x                - the vector of values
         dist             - the minimum distance between peaks
         chainCodeLengths - the chain-code length at each index;
                            if empty, the array indices are used instead
  
     Outputs:
         peaks   - the maximum peaks
         indices - the indices for the peaks
  
     See also MINPEAKSDIST, COMPUTECHAINCODELENGTHS
  
     ****************
     Used in seg_worm.feature_helpers.posture.getAmplitudeAndWavelength
  
     NOTE: Outputs ARE NOT SORTED 
  """
  #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Butil/maxPeaksDist.m
 
  chain_code_lengths = colon(1,1,x.size)

  # Is the vector larger than the search window?
  winSize = 2*dist + 1
  if chain_code_lengths[-1] < winSize:
    temp_I = np.argmax(x)
    return (x[temp_I],temp_I)
    
  #xt - "x for testing" in some places in the code below
  #it will be quicker (and/or easier) to assume that we want the largest
  #value. By negating the data we can look for maxima (which will tell us
  #where the minima are)
  if not use_max:
    xt = -1*x
  else:
    xt = x
    
  #NOTE: I added left/right neighbor comparisions which really helped with
  #the fft ..., a point can't be a peak if it is smaller than either of its
  #neighbors
  if use_max:    
    #                                           %> left                     > right
    #Matlab version:    
    #could_be_a_peak = x > value_cutoff & [true x(2:end) > x(1:end-1)] & [x(1:end-1) > x(2:end) true];
    #
    #TODO: Simplify how this code looks
    could_be_a_peak = np.logical_and(x > value_cutoff,np.concatenate((np.ones(1,dtype=np.bool),x[1:] > x[0:-1])))
    could_be_a_peak = np.logical_and(could_be_a_peak,np.concatenate((x[0:-1] > x[1:],np.ones(1,dtype=np.bool))))

    I1 = could_be_a_peak.nonzero()[0]    
    I2 = np.argsort(-1*x[I1]) #-1 => we want largest first
    I  = I1[I2]
  else:
    raise Exception("Not yet implemented")
    pdb.set_trace()
    #could_be_a_peak = x < value_cutoff & [true x(2:end) < x(1:end-1)] & [x(1:end-1) < x(2:end) true];
    #I1     = find(could_be_a_peak);
    #[~,I2] = sort(x(I1));
    #I = I1(I2);



  n_points = x.size

  #This code would need to be fixed if real distances
  #are input ...
  too_close = dist - 1

  temp_I  = colon(0,1,n_points-1)
  start_I = temp_I - too_close #Note, separated by dist is ok
  #This sets the off limits area, so we go in by 1
  end_I   = temp_I + too_close
  
  start_I[start_I < 0] = 0
  end_I[end_I > n_points] = n_points


  is_peak_mask = np.zeros(n_points,dtype=bool)
  #a peak and thus can not be used as a peak
    
  for cur_index in I:
    
    #NOTE: This gets updated in the loop so we can't just iterate
    #over these values
    if could_be_a_peak[cur_index]:
      #NOTE: Even if a point isn't the local max, it is greater
      #than anything that is by it that is currently not taken
      #(because of sorting), so it prevents these points
      #from undergoing the expensive search of determining
      #whether they are the min or max within their
      #else from being used, so we might as well mark those indices
      #within it's distance as taken as well
      temp_indices = slice(start_I[cur_index],end_I[cur_index])
      could_be_a_peak[temp_indices] = False
      
      #This line is really slow ...
      #It would be better to precompute the max within a window
      #for all windows ...
      is_peak_mask[cur_index] = np.max(xt[temp_indices]) == xt[cur_index]
  
  indices = is_peak_mask.nonzero()[0]
  peaks   = x[indices]
  
  """
  is_peak_mask   = false(1,n_points);
  %a peak and thus can not be used as a peak
  n_sort = length(I);
  for iElem = 1:n_sort
      cur_index = I(iElem);
      if could_be_a_peak(cur_index)
          %NOTE: Even if a point isn't the local max, it is greater
          %than anything that is by it that is currently not taken
          %(because of sorting), so it prevents these points
          %from undergoing the expensive search of determining
          %whether they are the min or max within their
          %else from being used, so we might as well mark those indices
          %within it's distance as taken as well
          temp_indices = start_I(cur_index):end_I(cur_index);
          could_be_a_peak(temp_indices) = false;
          
          %This line is really slow ...
          %It would be better to precompute the max within a window
          %for all windows ...
          is_peak_mask(cur_index) = max(xt(temp_indices)) == xt(cur_index);
      end
  end
  
  indices = find(is_peak_mask);
  peaks   = x(indices);
  """
  
  return (peaks,indices)
  
def colon(r1,inc,r2):
  
  """
    Matlab's colon operator, althought it doesn't although inc is required
  """
  s = np.sign(inc)

  if s == 0:
    return np.zeros(1)
  elif s == 1:
    n = ((r2-r1)+2*np.spacing(r2-r1))//inc
    return np.linspace(r1,r1+inc*n,n+1)
  else: #s == -1:
    #NOTE: I think this is slightly off as we start on the wrong end
    #r1 should be exact, not r2
    n  = ((r1-r2)+2*np.spacing(r1-r2))//np.abs(inc)
    temp = np.linspace(r2,r2+np.abs(inc)*n,n+1)    
    return temp[::-1]  

def print_object(obj):

    """ Goal is to eventually mimic Matlab's default display behavior for objects """

    #TODO - have some way of indicating nested function and not doing fancy
    #print for nested objects ...

    MAX_WIDTH = 70

    """
    Example output from Matlab    
    
    morphology: [1x1 seg_worm.features.morphology]
       posture: [1x1 seg_worm.features.posture]
    locomotion: [1x1 seg_worm.features.locomotion]
          path: [1x1 seg_worm.features.path]
          info: [1x1 seg_worm.info]
    """

    dict_local = obj.__dict__

    key_names      = [x for x in dict_local]    
    key_lengths    = [len(x) for x in key_names]
    
    if len(key_lengths) == 0:
      return ""
    
    max_key_length = max(key_lengths)
    key_padding    = [max_key_length - x for x in key_lengths]
    
    max_leadin_length = max_key_length + 2
    max_value_length  = MAX_WIDTH - max_leadin_length
 
 
    lead_strings   = [' '*x + y + ': ' for x,y in zip(key_padding,key_names)]    
    
    #TODO: Alphabatize the results ????
    #Could pass in as a option
    #TODO: It might be better to test for built in types
    #   Class::Bio.Entrez.Parser.DictionaryElement
    #   => show actual dictionary, not what is above
    
    
    value_strings = []
    for key in dict_local:
        value = dict_local[key]
        try: #Not sure how to test for classes :/
            class_name  = value.__class__.__name__
            module_name = inspect.getmodule(value).__name__
            temp_str    = 'Class::' + module_name + '.' + class_name
        except:
            temp_str    = repr(value)
            if len(temp_str) > max_value_length:
                #type_str = str(type(value))
                #type_str = type_str[7:-2]
                try:
                  len_value = len(value)
                except:
                  len_value = 1
                temp_str = str.format('Type::{}, Len: {}',type(value).__name__,len_value)                
  
        value_strings.append(temp_str)    
    
    final_str = ''
    for cur_lead_str, cur_value in zip(lead_strings,value_strings):
        final_str += (cur_lead_str + cur_value + '\n')


    return final_str
