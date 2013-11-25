""" experiment_file.py: A python module in the wormpy package

    Originally based on Jim Hokanson's gist:
    https://gist.github.com/JimHokanson/6425605

    This module defines the WormExperimentFile class, which encapsulates
    the data contained in the Shafer lab worm experiment data files
    
    To use it, instantiate the class by specifying a particular worm 
    video data file.  Then use the functions to extract information 
    about the worm and to do things like animate it and save that 
    animation to mp4.

"""
import math
import numpy as np
from matplotlib import pyplot
from matplotlib import animation
import h5py

#SPEED_UP = 4
#DT = 0.05
#TODO: try-catch block for savetomp3 in case anim is still None

class WormExperimentFile:
    """
        This class encapsulates the data in the Shafer lab
        Worm files.  Also, some of the worm data can be 
        animated using functions in this class.
        
    """

    # A 3-d numpy array containing the worm's position
    # this array's shape is:
    #   (
    #    # of frames, 
    #    # of worm points in each frame, 
    #    coordinate (0=x, 1=y)
    #   )
    # so shape is approx (23000, 49, 2):
    skeleton = None   
    # shape is approx (23000) and gives True if frame was dropped in 
    # experiment file
    dropped_frames_mask = None
    name = ''        # TODO: add to __init__, grab from wormFile["info"]
    points = None    # this contains our animation data
    anim = None      # this also contains our animation data, as an animation
    
    def __init__(self):
      pass

    def load_worm(self, worm_file_path):
        """ Load the worm data into wormFile """
        wormFile = h5py.File(worm_file_path, 'r')
        
        x_data = wormFile["worm"]["posture"]["skeleton"]["x"].value
        y_data = wormFile["worm"]["posture"]["skeleton"]["y"].value

        wormFile.close()

        self.combine_skeleton_axes(x_data, y_data)


    def combine_skeleton_axes(self, x_data, y_data):
        # We want to "concatenate" the values of the skeletonX and 
        # skeletonY 2D arrays into a 3D array
        # First let's create a temporary python list of numpy arrays
        skeletonTEMP = []
        
        # loop over all frames; frames are the first dimension of skeletonX
        for frameIndex in range(x_data.shape[0]):
            skeletonTEMP.append(np.column_stack((x_data[frameIndex], 
                                                     y_data[frameIndex])))

        # Then let's transform our list into a numpy array
        self.skeleton = np.array(skeletonTEMP)


    def skeletonX(self):
      """ returns a numpy array of shape (23135, 49) with just X coordinate
          data
      """
      return np.rollaxis(self.skeleton, 2)[0]

    def skeletonY(self):
      """ returns a numpy array of shape (23135, 49) with just X coordinate
          data
      """
      return np.rollaxis(self.skeleton, 2)[1]

    def dropped_frames_mask(self):
        # decide which frames are "dropped" by seeing which frames 
        # have the first skeleton X-coordinate set to NaN
        return np.isnan(list(frame[0] for frame in self.skeletonX()))


    def interpolate_dropped_frames(self):
      """ Fixes the dropped frames populated by NaN by inserting
          the most recent valid frame.
          Which frames are stale-dated (i.e. formerly dropped) is
          given by self.dropped_frames_mask
      """
      # ATTEMPT #1 (LOOP-BASED. VERY SLOW)
      # we will amend entries in this list to false as we patch up
      # the skeleton
      #s = list(self.dropped_frames_mask)
      #counter = 0 
      #while(max(s) == True and counter < 500):
      #  current_frame_to_fix = s.index(True)
      #  self.skeleton[current_frame_to_fix] = \
      #    self.skeleton[current_frame_to_fix - 1]
      #  s[current_frame_to_fix] = False
      #  counter += 1

      # ATTEMPT #2 (using the numpy.interp function)
      dropped_frames_mask = self.dropped_frames_mask()
      
      # this numpy function returns the array indices of all the True
      # fields in our mask, giving us a list of just the dropped frames
      dropped_frames = np.flatnonzero(dropped_frames_mask)
      # note that the tilde operator flips the True/False values elementwise
      good_frames = np.flatnonzero(~dropped_frames_mask)
      
      # extract just the x-coordinates.  dataX has shape (49, 23135)
      x_data = np.rollaxis(self.skeletonX(), 1)
      y_data = np.rollaxis(self.skeletonY(), 1)
      
      # interpolate missing data points for each of the worm's 49 
      # skeleton points
      for i in range(0, 48):
        # in each of the x and y axes, replace the NaN entries with 
        # interpolated entries taken from data in nearby frames
        x_data[i][dropped_frames_mask] = np.interp(dropped_frames, good_frames, x_data[i][~dropped_frames_mask])
        y_data[i][dropped_frames_mask] = np.interp(dropped_frames, good_frames, y_data[i][~dropped_frames_mask])

      # change dataX and dataY so their shape is the more familiar (23135, 49)
      # this is the shape expected by combine_skeleton_axes()
      x_data = np.rollaxis(x_data, 1)
      y_data = np.rollaxis(y_data, 1)

      # Create a new instance, with the interpolated results    
      w = WormExperimentFile()
      w.combine_skeleton_axes(x_data, y_data)
      
      return w
      
    
    def create_animation(self):
        """ Creates an animation of the worm's position over time.
        
        """
        fig = pyplot.figure()
        
        fig.suptitle('Worm position over time', fontsize=20)
        pyplot.xlabel('x coordinates', fontsize=18)
        pyplot.ylabel('y coordinates', fontsize=16)

        # Set the axes to the maximum extent of the worm's travels
        ax = pyplot.axes(xLim=self.position_limits(0), 
                         yLim=self.position_limits(1))
        
        
        # Alternatively: marker='o', linestyle='None'
        # the plot starts with all worm position points from frame 0
        points, = ax.plot(self.skeleton[0,:,0], self.skeleton[0,:,1], 
                          color='green', linestyle='point marker', 
                          marker='o', markersize=5) 

        # inline initialization function: plot the background of each frame
        def init():
            points.set_data([], [])
            return points,
        
        # inline animation function.  This is called sequentially
        def animate_frame(iFrame):
            points.set_data(self.skeleton[iFrame,:,0], 
                            self.skeleton[iFrame,:,1])
            return points,
        
        # let's just run it for a 1/60th subset of the complete number of 
        # frames to make it faster to save the mp4.
        self.anim = \
            animation.FuncAnimation(fig, func=animate_frame, init_func=init,
                                    # total frames / 60 = about 15 seconds
                                    frames=math.floor(self.num_frames()/60), 
                                    interval=20, blit=True, repeat_delay=100)  
    
    
    def position_limits(self, dimension):  
        """ Maximum extent of worm's travels projected onto a given axis
            dimension is 0 for X axis, or 1 for Y axis.
        NOTE: Dropped frames show up as NaN.  
              nanmin returns the min ignoring such NaNs.        
        
        """
        return (np.nanmin(self.skeleton[:,:,dimension]), 
                np.nanmax(self.skeleton[:,:,dimension]))
    
    def num_frames(self): 
        # the number of frames in the video
        # ndarray.shape returns a tuple of array dimensions.
        # the frames are along the first dimension i.e. [0].
        return self.skeleton.shape[0]


    def num_skeleton_points(self): 
        # the number of points in the skeleton of the worm
        # ndarray.shape returns a tuple of array dimensions.
        # the skeletal points are along the first dimension i.e. [1].
        return self.skeleton.shape[1]

    def num_valid_frames(self):
        pass  # TODO:

    def position(self): 
        # return a two-dimensional array with worm's position
        pass  # TODO:

    def save_to_mp4(self, filename):
        """  Save the animation as an mp4.
        This requires ffmpeg or mencoder to be installed.
        The extra_args ensure that the x264 codec is used, so that 
        the video can be embedded in html5.  You may need to adjust 
        this for your system: for more information, see
        http://matplotlib.sourceforge.net/api/animation_api.html
                
        to install ffmpeg on windows, see
        http://www.wikihow.com/Install-FFmpeg-on-Windows
        
        """
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='C. elegans movement video', artist='Matplotlib',
                        comment='C. elegans movement video from Shafer lab')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        self.anim.save(filename, writer=writer, fps=15, 
                       extra_args=['-vcodec', 'libx264'])
