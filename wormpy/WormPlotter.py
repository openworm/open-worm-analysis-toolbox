# -*- coding: utf-8 -*-
"""
WormPlotter: A class that renders matplotlib plots of worm measurement and
             feature data.

@authors: @JimHokanson, @MichaelCurrie

This follows the [animated subplots example]
(http://matplotlib.org/1.3.0/examples/animation/subplots.html)
in creating a class derivation from TimedAnimation rather than 
using pylab shell-type calls to construct the animation.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from wormpy import config




class WormPlotter(animation.TimedAnimation):
  
  def __init__(self, normalized_worm, interactive=False):
    """
      __init__: Initialize the animation of the worm's attributes.

        INPUT: 
          normalized_worm: the NormalizedWorm object to be plotted.
          
          interactive: boolean, if interactive is set to False, then:
          suppress the drawing of the figure, until an explicit plt.show()
          is called.  this allows WormPlotter to be instantiated without 
          just automatically being displayed.  Instead, the user must call
          WormPlotter.show() to have plt.show() be called.

        To initialize the animation, we must do six things:
        
        1. set up the data to be used from the normalized_worm
        2. create the figure
        3. create subplots in the figure, assigning them Axis handles
        4. create Artist objects for all objects in the subplots 
        5. assign the Artist objects to the correct Axis handle
        6. call the base class __init__

    """
    # A WormPlotter instance can be instantiated to be interactive,
    # or by default it is set to NOT interactive, which means that 
    # WormPlotter.show() must be called to get it to display.
    plt.interactive(interactive)
    
    # 1. set up the data to be used
    self.normalized_worm = normalized_worm
    # TODO: eventually we'll put this in a nicer place
    self.vulva_contours = self.normalized_worm.data_dict['vulva_contours']
    self.non_vulva_contours = self.normalized_worm.data_dict['non_vulva_contours']
    self.skeletons = self.normalized_worm.data_dict['skeletons']  
    self.skeletons_centred = self.normalized_worm.translate_to_centre()
    self.skeleton_centres = self.normalized_worm.centre()    
    self.orientation = self.normalized_worm.angle()    
    self.skeletons_rotated = self.normalized_worm.rotate_and_translate()
        
      
    # 2. create the figure
    fig = plt.figure(figsize=(5,5))

    # We have blit=True, so the animation only redraws the elements that have
    # changed.  This means that if the window is resized, everything other
    # than the plot area will be black.  To fix this, here we have matplotlib
    # explicitly redraw everything if a resizing event occurs.
    # DEBUG: this actually doesn't appear to work.
    def refresh_plot(event):
      fig.canvas.draw()

    self.refresh_connection_id = \
      fig.canvas.mpl_connect('resize_event', refresh_plot)

    fig.suptitle('C. elegans attributes', fontsize=20)
    
    # 3. add the subplots    
    ax1 = plt.subplot2grid((3,3), (0,0), rowspan=2, colspan=2)
    ax1.set_title('Position')    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(self.normalized_worm.position_limits(0))
    ax1.set_ylim(self.normalized_worm.position_limits(1))
    #ax1.set_aspect(aspect='equal', adjustable='datalim')
    #ax1.set_autoscale_on()    
    
    ax2 = plt.subplot2grid((3,3), (2,0))
    ax2.set_title('Morphology')    
    ax2.set_xlim((-500, 500))  # DON'T USE set_xbound, it changes dynmically
    ax2.set_ylim((-500, 500))
    ax2.set_aspect(aspect='equal', adjustable='datalim')
    self.annotation2 = ax2.annotate("Worm head",
                                    xy=(0,0), xycoords='data',
                                    xytext=(10, 10), textcoords='data',
                                    arrowprops=dict(arrowstyle="fancy",
                                                    connectionstyle="arc3,rad=.2"))


    ax3 = plt.subplot2grid((3,3), (2,1))
    ax3.set_title('Orientation-free')
    ax3.set_xlim((-500, 500))  # DON'T USE set_xbound, it changes dynmically
    ax3.set_ylim((-500, 500))
    ax3.set_aspect(aspect='equal', adjustable='datalim')

    ax4 = plt.subplot2grid((3,3), (2,2))
    self.annotation4 = ax4.annotate("Segmentation status: ",
                                    xy=(0,0), xycoords='data',
                                    xytext=(10, 10), textcoords='data',
                                    arrowprops=dict(arrowstyle="fancy",
                                                    connectionstyle="arc3,rad=.2"))

    # 4. create Artist objects 
    self.line1W = Line2D([], [], color='green', linestyle='point marker', 
                         marker='o', markersize=5) 
    self.line1W_head = Line2D([], [], color='red', linestyle='point marker', 
                              marker='o', markersize=7) 
    self.line1C = Line2D([], [], color='yellow', linestyle='point marker', 
                         marker='o', markersize=5) 
    self.patch1E = Ellipse(xy=(0,0), width=1000, height=500, angle=0, alpha=0.3)

    self.line2W = Line2D([], [], color='black', marker='o', markersize=5)
    self.line2W_head = Line2D([], [], color='red', linestyle='point marker', 
                              marker='o', markersize=7) 
    self.line2C = Line2D([], [], color='blue') 
    self.line2C2 = Line2D([], [], color='orange') 

    self.line3W = Line2D([], [], color='black', marker='o', markersize=5)
    self.line3W_head = Line2D([], [], color='red', linestyle='point marker', 
                              marker='o', markersize=7) 
    
    # 5. assign Artist objects to the relevant subplot
    ax1.add_line(self.line1W)
    ax1.add_line(self.line1W_head)
    ax1.add_line(self.line1C)
    ax1.add_artist(self.patch1E)
    
    ax2.add_line(self.line2W)
    ax2.add_line(self.line2W_head)
    ax2.add_line(self.line2C)
    ax2.add_line(self.line2C2)
    
    ax3.add_line(self.line3W)
    ax3.add_line(self.line3W_head)


    # So labels don't overlap:    
    plt.tight_layout()

    # 6. call the base class __init__

    # TimedAnimation draws a new frame every *interval* milliseconds.
    # so this is how we convert from FPS to interval:
    interval = 1000 / config.FPS

    return animation.TimedAnimation.__init__(self, 
                                             fig, 
                                             interval=interval, 
                                             blit=True)

  def show(self):
    """
      show:
        draw the figure in a window on the screen
      
    """
    plt.show()

  def _draw_frame(self, framedata):
    """
      _draw_frame:
        Called sequentially for each frame of the animation.  Thus
        we must set our plot to look like it should for the given frame.
        
      INPUT
        framedata: 
          an integer between 0 and the number of frames, giving
          the current frame.
      
    """

    i = framedata

    self.line1W.set_data(self.skeletons[:,0,i],
                         self.skeletons[:,1,i])
    self.line1W_head.set_data(self.skeletons[0,0,i],
                              self.skeletons[0,1,i])
    self.line1C.set_data(self.vulva_contours[:,0,i],
                         self.vulva_contours[:,1,i])                              
    self.patch1E.center = (self.skeleton_centres[:,i])
    self.patch1E.angle = self.orientation[i]
    

    self.line2W.set_data(self.skeletons_centred[:,0,i],
                         self.skeletons_centred[:,1,i])
    self.line2W_head.set_data(self.skeletons_centred[0,0,i],
                              self.skeletons_centred[0,1,i])
    self.line2C.set_data(self.skeletons_centred[:,0,i] + (self.vulva_contours[:,0,i] - self.skeletons[:,0,i]),
                         self.skeletons_centred[:,1,i] + (self.vulva_contours[:,1,i] - self.skeletons[:,1,i]))
    self.line2C2.set_data(self.skeletons_centred[:,0,i] + (self.non_vulva_contours[:,0,i] - self.skeletons[:,0,i]),
                          self.skeletons_centred[:,1,i] + (self.non_vulva_contours[:,1,i] - self.skeletons[:,1,i]))
    self.annotation2.xy = (self.skeletons_centred[0,0,i],
                           self.skeletons_centred[0,1,i])
    self.annotation2.label = 'label'  # DEBUG
    self.annotation2.text = 'text'    # DEBUG 

                            
    self.line3W.set_data(self.skeletons_rotated[:,0,i],
                         self.skeletons_rotated[:,1,i])
    self.line3W_head.set_data(self.skeletons_rotated[0,0,i],
                              self.skeletons_rotated[0,1,i])

    
    self._drawn_artists = [self.line1W, self.line1C, self.line1W_head, self.patch1E,
                           self.line2W, self.line2C, self.line2C2, self.line2W_head, self.annotation2,
                           self.line3W, self.line3W_head,
                           self.annotation4]

  def new_frame_seq(self):
    """
      returns an iterator that iterates over the frames 
      in the animation
      
    """
    return iter(range(self.normalized_worm.num_frames()))

  def _init_draw(self):
    """
      _init_draw:
        Called when first drawing the animation.
        It is an abstract method in Animation, to be implemented here for
        the first time.
      
    """
    artists =  [self.line1W, self.line1W_head, self.line1C,
              self.line2W, self.line2W_head, self.line2C, self.line2C2,
              self.line3W, self.line3W_head]

    for l in artists:
      l.set_data([], [])
    
    # TODO: figure out how to clear the non-line elements
    #artists.extend([self.annotation2, self.patch1E])
    

  def save(self, filename):
    """ 
      save:
        Save the animation as an mp4.
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
    animation.TimedAnimation.save(self, filename, 
                                  writer=writer, fps=config.FPS, 
                                  extra_args=['-vcodec', 'libx264'])



def plot_frame_codes(normalized_worm):
  """
    Plot a pie chart of the frame codes of a normalized worm.
    
  """  
  # TODO: someday make this pie chart look nicer with:
  # http://nxn.se/post/46440196846/making-nicer-looking-pie-charts-with-matplotlib
  nw = normalized_worm
  fc = nw.data_dict['frame_codes']
  # create a dictionary of    frame code : frame code title   pairs
  fc_desc = {b[0]: b[2] for b in nw.frame_codes_descriptions}
  
  # a dictionary with the count for each frame code type
  counts = {i:np.bincount(fc)[i] for i in np.unique(fc)}

  # display the pie chart  
  patches, texts, autotexts = plt.pie(x=list(counts.values()), 
          labels=list(fc_desc[d] for d in np.unique(fc)), 
          autopct='%1.1f%%',
          startangle=90,
          colors=['g','r','c', 'y', 'm'], labeldistance=1.2)
  plt.suptitle("Proportion of frames segmented")
  
  for t in texts:
    t.set_size('smaller')
  for t in autotexts:
    t.set_size('x-small')
  
  
  