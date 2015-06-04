"""
**********************************************
**********************************************
**********************************************
**********************************************
DEPRECATED.   Do not use.

TODO: extract the one useful method from this, 
interpolate_dropped_frames, put it in NormalizedWorm, and then delete this file
**********************************************
**********************************************
**********************************************
**********************************************


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

"""
   DIFFERENCES BETWEEN wormpy.SchaferExperimentFile and 
                       wormpy.WormFeatures:
   1. SchaferExperimentFile expects skeletons to be 
      in the shape (n, 49, 2), but data_dict['skeletons'] is in the 
      shape (49, 2, n), so we must "roll the axis" twice.
      self.skeletons = np.rollaxis(worm_features.data_dict['skeletons'], 2)
      
"""

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
        """ Returns a numpy array of shape (23135, 49) with just X coordinate
            data
        """
        return np.rollaxis(self.skeletons, 2)[0]

    def skeletons_y(self):
        """ Returns a numpy array of shape (23135, 49) with just X coordinate
            data
        """
        return np.rollaxis(self.skeletons, 2)[1]

    def dropped_frames_mask(self):
        """ Decide which frames are "dropped" by seeing which frames 
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

