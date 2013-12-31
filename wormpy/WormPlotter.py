# -*- coding: utf-8 -*-
"""
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
from matplotlib.patches import Ellipse
import matplotlib.animation as animation


class WormPlotter(animation.TimedAnimation):
  normalized_worm = None
  skeletons = None 
  skeletons_centred = None
  
  
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
    self.skeletons_centred = self.normalized_worm.translate_to_centre()
    
    self.t = np.linspace(0, 80, 400)
    self.x = np.cos(2 * np.pi * self.t / 10.) + 10000
    self.y = np.sin(2 * np.pi * self.t / 10.) + 13000

    
      
    # 2. create the figure
    fig = plt.figure()
    fig.suptitle('C. elegans attributes', fontsize=20)    
    
    # 3. add the subplots    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Position')    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(self.position_limits(0))  # DON'T USE set_xbound, it changes dynmically
    ax1.set_ylim(self.position_limits(1))
    ax1.set_aspect(aspect='equal', adjustable='datalim')
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Morphology')    
    ax2.set_xlim((-500, 500))  # DON'T USE set_xbound, it changes dynmically
    ax2.set_ylim((-500, 500))
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

    self.line2W.set_data(self.skeletons_centred[:,0,i],
                         self.skeletons_centred[:,1,i])

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


