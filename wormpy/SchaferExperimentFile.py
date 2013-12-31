""" wormpy\SchaferExperimentFile.py

    Authors: @MichaelCurrie, @JimHokanson

    This module defines the SchaferExperimentFile class, which encapsulates
    the data contained in the Shafer lab worm experiment data files
    
    To use it, instantiate the class by specifying a particular worm 
    HDF5 data file.  Then use the functions to extract information 
    about the worm.

    Once you've created your SchaferExperimentFile object, you
    will likely want to convert it to a wormpy.WormFeatures object using
    WormFeatures.load_schafer_experiment_file, so you can do things
    like animate the data.

"""
import os
import numpy as np
import h5py

#SPEED_UP = 4
#DT = 0.05

"""
   DIFFERENCES BETWEEN wormpy.SchaferExperimentFile and 
                       wormpy.WormFeatures:
   1. SchaferExperimentFile expects skeletons to be 
      in the shape (n, 49, 2), but data_dict['skeletons'] is in the 
      shape (49, 2, n), so we must "roll the axis" twice.
      self.skeletons = np.rollaxis(worm_features.data_dict['skeletons'], 2)
      
"""


class SchaferExperimentFile:
  """
      This class encapsulates the data in the Shafer lab
      Worm feature files, which store the final features data.
      
      This is provided as a legacy support to interpret their data.
      
      The latest equivalent is wormpy.WormFeatures.
      
      Some aspects of the data here are different in that later version.
  """

  # A 3-d numpy array containing the worm's position
  # this array's shape is:
  #   (
  #    # of frames, 
  #    # of worm points in each frame, 
  #    coordinate (0=x, 1=y)
  #   )
  # so shape is approx (23000, 49, 2):
  skeletons = None   
  name = ''        # TODO: add to __init__, grab from wormFile["info"]
  # this contains our animation data
  # TODO: avoid making this a member data element, by
  # referencing the parent scope when referencing animation_points
  animation_points = None
  # this also contains our animation data, as an animation
  animation_data = None      
  
  def __init__(self):
    """ Let's not initialize the worm data, leaving it open
        to an HDF5 load or a load from the more basic, block data.
    """
    pass

  def load_HDF5_data(self, worm_file_path):
    """ 
      load_HDF5_data:
        Load the worm data, including the skeleton data
    
    """
    if(not os.path.isfile(worm_file_path)):
      raise Exception("Worm file not found: " + worm_file_path)
    else:
      worm_file = h5py.File(worm_file_path, 'r')
      
      x_data = worm_file["worm"]["posture"]["skeleton"]["x"].value
      y_data = worm_file["worm"]["posture"]["skeleton"]["y"].value

      worm_file.close()

      self.skeletons = self.combine_skeleton_axes(x_data, y_data)


  def combine_skeleton_axes(self, x_data, y_data):
    """ We want to "concatenate" the values of the skeletons_x and 
        skeletons_y 2D arrays into a 3D array
        First let's create a temporary python list of numpy arrays
        
    """
    skeletons_TEMP = []
    
    # Loop over all frames; frames are the first dimension of skeletons_x
    for frame_index in range(x_data.shape[0]):
        skeletons_TEMP.append(np.column_stack((x_data[frame_index], 
                                              y_data[frame_index])))

    # Return our list as a numpy array
    return np.array(skeletons_TEMP)

  def skeletons_x(self):
    """ returns a numpy array of shape (23135, 49) with just X coordinate
        data
    """
    return np.rollaxis(self.skeletons, 2)[0]

  def skeletons_y(self):
    """ returns a numpy array of shape (23135, 49) with just X coordinate
        data
    """
    return np.rollaxis(self.skeletons, 2)[1]

  def dropped_frames_mask(self):
    """ decide which frames are "dropped" by seeing which frames 
        have the first skeleton X-coordinate set to NaN
        returned shape is approx (23000) and gives True if frame 
        was dropped in experiment file
    """
    return np.isnan(list(frame[0] for frame in self.skeletons_x()))


  def interpolate_dropped_frames(self):
    """ Fixes the "dropped" frames (i.e. frames that are NaN) 
        by doing linear interpolation between the nearest valid frames.
        Which frames are interpolated and not genuine data is
        given by self.dropped_frames_mask
    """
    # ATTEMPT #1 (LOOP-BASED. VERY SLOW)
    # we will amend entries in this list to false as we patch up
    # the skeletons
    #s = list(self.dropped_frames_mask)
    #counter = 0 
    #while(max(s) == True and counter < 500):
    #  current_frame_to_fix = s.index(True)
    #  self.skeletons[current_frame_to_fix] = \
    #    self.skeletons[current_frame_to_fix - 1]
    #  s[current_frame_to_fix] = False
    #  counter += 1

    # ATTEMPT #2 (using the numpy.interp function)
    dropped_frames_mask = self.dropped_frames_mask()
    
    # this numpy function returns the array indices of all the True
    # fields in our mask, giving us a list of just the dropped frames
    dropped_frames = np.flatnonzero(dropped_frames_mask)
    # note that the tilde operator flips the True/False values elementwise
    good_frames = np.flatnonzero(~dropped_frames_mask)
    
    # extract just the x-coordinates.  x_data has shape (49, 23135)
    x_data = np.rollaxis(self.skeletons_x(), 1)
    y_data = np.rollaxis(self.skeletons_y(), 1)
    
    # interpolate missing data points for each of the worm's 49 
    # skeleton points (np.shape(self.skeletons)[1] is just telling 
    # us the shape of the skeleton in the dimension that contains the worm's 
    # points for each given frame)
    for i in range(0, np.shape(self.skeletons)[1]):
      # in each of the x and y axes, replace the NaN entries with 
      # interpolated entries taken from data in nearby frames
      x_data[i][dropped_frames_mask] = \
        np.interp(dropped_frames,
                  good_frames, 
                  x_data[i][~dropped_frames_mask])
      y_data[i][dropped_frames_mask] = \
        np.interp(dropped_frames,
                  good_frames,
                  y_data[i][~dropped_frames_mask])

    # change x_data and y_data so their shape is the more 
    # familiar (23135, 49)
    # this is the shape expected by combine_skeleton_axes()
    x_data = np.rollaxis(x_data, 1)
    y_data = np.rollaxis(y_data, 1)

    # Create a new instance, with the interpolated results    
    w = SchaferExperimentFile()
    w.combine_skeleton_axes(x_data, y_data)
    
    return w
     
  def position_limits(self, dimension):  
    """ Maximum extent of worm's travels projected onto a given axis
        PARAMETERS:
          dimension: specify 0 for X axis, or 1 for Y axis.
    NOTE: Dropped frames show up as NaN.  
          nanmin returns the min ignoring such NaNs.        
    
    """
    return (np.nanmin(self.skeletons[:,:,dimension]), 
            np.nanmax(self.skeletons[:,:,dimension]))

  def num_frames(self): 
    """ the number of frames in the video
        ndarray.shape returns a tuple of array dimensions.
        the frames are along the first dimension i.e. [0].
    """
    return self.skeletons.shape[0]


  def num_skeletons_points(self): 
    """ the number of points in the skeletons of the worm
        ndarray.shape returns a tuple of array dimensions.
        the skeletal points are along the first dimension i.e. [1].
    """
    return self.skeletons.shape[1]