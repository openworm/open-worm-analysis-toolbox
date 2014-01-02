# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:04:02 2013

WormPlotter: A class that renders matplotlib plots of worm measurement and
             feature data.

This follows the [animated subplots example]
(http://matplotlib.org/1.3.0/examples/animation/subplots.html)
in creating a class derivation from TimedAnimation rather than 
using pylab shell-type calls to construct the animation.

@authors: @MichaelCurrie, @JimHokanson

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import wormpy_example

class WormPlotter(animation.TimedAnimation):
  normalized_worm = None
  skeletons = None  
  
  
  def __init__(self, normalized_worm):
    """
      __init__: Initialize the animation of the worm's attributes.
        To initialize the animation, we must do six things:
        
        1. set up the data to be used from the normalized_worm
        2. create the figure
        3. create subplots in the figure, assigning them Axis handles
        4. create Line2D objects for all objects in the subplots 
        5. assign the Line2D objects to the correct Axis handle
        6. call the base class __init__
     
    """

    # 1. set up the data to be used
    self.normalized_worm = normalized_worm
    # TODO: eventually we'll put this in a nicer place
    self.skeletons = self.normalized_worm.data_dict['skeletons']  

    self.t = np.linspace(0, 80, 400)
    self.x = np.cos(2 * np.pi * self.t / 10.) + 10000
    self.y = np.sin(2 * np.pi * self.t / 10.) + 13000

    
      
    # 2. create the figure
    fig = plt.figure()
    fig.suptitle('C. elegans attributes', fontsize=20)    
    
    # 3. add the subplots    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Worm Position over time')    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(self.position_limits(0))  # DON'T USE set_xbound, it changes dynmically
    ax1.set_ylim(self.position_limits(1))
    ax1.set_aspect(aspect='equal', adjustable='datalim')
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Worm Position over time 2')    
    ax2.set_xlim(self.position_limits(0))  # DON'T USE set_xbound, it changes dynmically
    ax2.set_ylim(self.position_limits(1))
    ax2.set_aspect(aspect='equal', adjustable='datalim')

    # 4. create Line2D objects
    self.line1 = Line2D([], [], color='black')
    self.line1a = Line2D([], [], color='green', linewidth=2)
    self.line1e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
    self.line1W = Line2D([], [], color='green', linestyle='point marker', 
                         marker='o', markersize=5) 

    self.line2W = Line2D([], [], color='black', marker='o', markersize=5)

    # 5. assign Line2D objects to the relevant subplot
    ax1.add_line(self.line1)
    ax1.add_line(self.line1a)
    ax1.add_line(self.line1e)
    ax1.add_line(self.line1W)
    
    ax2.add_line(self.line2W)


    # 6. call the base class __init__
    return animation.TimedAnimation.__init__(self, fig, interval=15, blit=True)

  def _draw_frame(self, framedata):
    i = framedata
    head = i - 1
    head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

    self.line1.set_data(self.x[:i], self.y[:i])
    self.line1a.set_data(self.x[head_slice], self.y[head_slice])
    self.line1e.set_data(self.x[head], self.y[head])

    self.line1W.set_data(self.skeletons[:,0,i],
                         self.skeletons[:,1,i])

    self.line2W.set_data(self.skeletons[10,0,i],
                         self.skeletons[10,1,i])

    self._drawn_artists = [self.line1, self.line1a, self.line1e, self.line1W, self.line2W]

  def new_frame_seq(self):
    return iter(range(self.t.size))

  def _init_draw(self):
    lines =  [self.line1, self.line1a, self.line1e, self.line1W, self.line2W]

    for l in lines:
      l.set_data([], [])

  def position_limits(self, dimension):  
    """ Maximum extent of worm's travels projected onto a given axis
        PARAMETERS:
          dimension: specify 0 for X axis, or 1 for Y axis.
    NOTE: Dropped frames show up as NaN.  
          nanmin returns the min ignoring such NaNs.        
    
    """
    return (np.nanmin(self.skeletons[dimension,:,:]), 
            np.nanmax(self.skeletons[dimension,:,:]))


  def animate_OLD(self, portion = 0.1):
    """ Creates an animation of the worm's position over time.
    
        optional parameter portion is a figure between 0 and 1 of frames
        to animate.  default is 10%.
    """
    fig = plt.figure()
    
    fig.suptitle('Worm position over time', fontsize=20)
    plt.xlabel('x coordinates', fontsize=18)
    plt.ylabel('y coordinates', fontsize=16)

    # Set the axes to the maximum extent of the worm's travels
    ax = plt.axes(xLim=self.position_limits(0), 
                  yLim=self.position_limits(1))
    

    
    
    # Alternatively: marker='o', linestyle='None'
    # the plot starts with all worm position animation_points from frame 0
    animation_points, = ax.plot(self.skeletons[0,:,0], 
                                self.skeletons[0,:,1],
                                color='green', 
                                linestyle='point marker', 
                                marker='o', 
                                markersize=5) 

    # inline initialization function: plot the background of each frame
    def init():
      animation_points.set_data([], [])
      return animation_points,
    
    # inline animation function.  This is called sequentially
    def animate_frame(iFrame):
      animation_points.set_data(self.skeletons[iFrame,:,0], 
                                self.skeletons[iFrame,:,1])
      return animation_points,
    
    # create animation of a certain number of frames.
    self.animation_data = \
        animation.FuncAnimation(fig, func=animate_frame, 
                                init_func=init,
                                # animate only a portion of the frames.
                                frames=400, interval=20, 
                                blit=True, repeat_delay=100)  


  def save(self, filename):
    """ Save the animation as an mp4.
        This requires ffmpeg or mencoder to be installed.
        The extra_args ensure that the x264 codec is used, so that 
        the video can be embedded in html5.  You may need to adjust 
        this for your system: for more information, see
        http://matplotlib.sourceforge.net/api/animation_api.html
            
        To install ffmpeg on windows, see
        http://www.wikihow.com/Install-FFmpeg-on-Windows
    """
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='C. elegans movement video', artist='matplotlib',
                    comment='C. elegans movement video from Shafer lab')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    animation.TimedAnimation.save(self, filename, writer=writer, 
                                  fps=15, extra_args=['-vcodec', 'libx264'])



nw = wormpy_example.example_nw()

ani = WormPlotter(nw)
#ani.save('test_sub.mp4')
plt.show()