# -*- coding: utf-8 -*-
"""
A class that renders matplotlib plots of worm measurement and feature data.
  
This follows the [animated subplots example]
(http://matplotlib.org/1.3.0/examples/animation/subplots.html)
in inheriting from TimedAnimation rather than 
using PyLab-shell-type calls to construct the animation.

Usage
----------------------------

Within the movement_validation repository, the plotting code is encapsulated 
in the WormPlotter class.

If you already have an instance nw of the NormalizedWorm class, you could 
plot it as follows:

    wp = wormpy.WormPlotter(nw, interactive=False)
    wp.show()

Another plotting function is the number of dropped frames in a given 
normalized worm's segmented video. To show a pie chart breaking down this 
information, you could run the following:

wormpy.plot_frame_codes(nw)

Creating such an instance nw of the NormalizedWorm class requires access 
to worm data files. The easiest way to obtain such data files is to sync
with the Google Drive folder example_movement_validation_data, which is 
discussed in the installation instructions: 
https://github.com/openworm/movement_validation/blob/master/INSTALL.md

Once you've joined the worm_data shared folder, you'll need to clone and 
configure the movement_validation repository for yourself. Here is an 
installation guide for you.

I note in the code for WormPlotter that the visualization module follows 
the animated subplots example in inheriting from TimedAnimation rather 
than using PyLab shell-type calls to construct the animation, which is 
the more typical approach. I used this approach so that it would be 
extensible into other visualization possibilities.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

from . import config


class WormPlotter(animation.TimedAnimation):

    def __init__(self, normalized_worm, motion_mode=None, interactive=False):
        """ 
        Initialize the animation of the worm's attributes.

        Parameters
        ---------------------------------------
        normalized_worm: NormalizedWorm
          the NormalizedWorm object to be plotted.

        motion_mode: 1-dimensional numpy array (optional)
          The motion mode of the worm over time.  Must have 
          length = normalized_worm.num_frames

        interactive: boolean (optional)
          if interactive is set to False, then:
            suppress the drawing of the figure, until an explicit plt.show()
            is called.  this allows WormPlotter to be instantiated without 
            just automatically being displayed.  Instead, the user must call
            WormPlotter.show() to have plt.show() be called.

        Notes
        ---------------------------------------
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

        self.motion_mode = motion_mode
        self.motion_mode_options = {-1: 'backward',
                                    0: 'paused',
                                    1: 'forward'}
        self.motion_mode_colours = {-1: 'b',      # blue
                                    0: 'r',       # red
                                    1: 'g'}      # green

        self.normalized_worm = normalized_worm
        # TODO: eventually we'll put this in a nicer place
        self.vulva_contours = self.normalized_worm.vulva_contours
        self.non_vulva_contours = self.normalized_worm.non_vulva_contours
        self.skeletons = self.normalized_worm.skeletons
        self.skeletons_centred = self.normalized_worm.translate_to_centre()
        self.skeleton_centres = self.normalized_worm.centre
        self.orientation = self.normalized_worm.angle
        self.skeletons_rotated = self.normalized_worm.rotate_and_translate()

        # 2. create the figure
        fig = plt.figure(figsize=(5, 5))

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
        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        ax1.set_title('Position')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_xlim(self.normalized_worm.position_limits(0))
        ax1.set_ylim(self.normalized_worm.position_limits(1))
        #ax1.set_aspect(aspect='equal', adjustable='datalim')
        # ax1.set_autoscale_on()

        self.annotation1a = ax1.annotate("(motion mode data not available)",
                                         xy=(-10, 10), xycoords='axes points',
                                         horizontalalignment='right',
                                         verticalalignment='top',
                                         fontsize=10)

        self.annotation1b = ax1.annotate("bottom right (points)",
                                         xy=(-10, 10), xycoords='axes points',
                                         horizontalalignment='right',
                                         verticalalignment='bottom',
                                         fontsize=10)

        # DEBUG: this doesn't appear to do anything,
        # it was an attempt to get it to refresh with diff titles
        #self.annotation1.animated = True

        ax2 = plt.subplot2grid((3, 3), (2, 0))
        ax2.set_title('Morphology')
        # DON'T USE set_xbound, it changes dynmically
        ax2.set_xlim((-500, 500))
        ax2.set_ylim((-500, 500))
        ax2.set_aspect(aspect='equal', adjustable='datalim')
        self.annotation2 = ax2.annotate("Worm head",
                                        xy=(0, 0), xycoords='data',
                                        xytext=(10, 10), textcoords='data',
                                        arrowprops=dict(arrowstyle="fancy",
                                                        connectionstyle="arc3,rad=.2"))

        ax3 = plt.subplot2grid((3, 3), (2, 1))
        ax3.set_title('Orientation-free')
        # DON'T USE set_xbound, it changes dynmically
        ax3.set_xlim((-500, 500))
        ax3.set_ylim((-500, 500))
        ax3.set_aspect(aspect='equal', adjustable='datalim')

        ax4 = plt.subplot2grid((3, 3), (2, 2))
        # self.annotation4 = ax4.annotate("Segmentation status: ",
        #                                xy=(0,0), xycoords='data',
        #                                xytext=(10, 10), textcoords='data',
        #                                arrowprops=dict(arrowstyle="fancy",
        #                                                connectionstyle="arc3,rad=.2"))

        # 4. create Artist objects
        self.line1W = Line2D([], [], color='green', linestyle='point marker',
                             marker='o', markersize=5)
        self.line1W_head = Line2D([], [], color='red', linestyle='point marker',
                                  marker='o', markersize=7)
        self.line1C = Line2D([], [], color='yellow', linestyle='point marker',
                             marker='o', markersize=5)
        self.patch1E = Ellipse(
            xy=(0, 0), width=1000, height=500, angle=0, alpha=0.3)

        self.line2W = Line2D([], [], color='black', marker='o', markersize=5)
        self.line2W_head = Line2D([], [], color='red', linestyle='point marker',
                                  marker='o', markersize=7)
        self.line2C = Line2D([], [], color='blue')
        self.line2C2 = Line2D([], [], color='orange')

        self.line3W = Line2D([], [], color='black', marker='o', markersize=5)
        self.line3W_head = Line2D([], [], color='red', linestyle='point marker',
                                  marker='o', markersize=7)

        self.artists_to_be_drawn = \
            [self.line1W, self.line1C, self.line1W_head, self.patch1E, self.annotation1a, self.annotation1b,
             self.line2W, self.line2C, self.line2C2, self.line2W_head, self.annotation2,
             self.line3W, self.line3W_head]

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

    @property
    def num_frames(self):
        """
          Return the total number of frames in the animation

        """
        return self.normalized_worm.num_frames

    """
    motion_mode is a numpy array of length num_frames giving the motion
    type over time.  it must be loaded from features data as it is not
    a property of the NormalizedWorm.
  """
    @property
    def motion_mode(self):
        return self._motion_mode

    @motion_mode.setter
    def motion_mode(self, m):
        self._motion_mode = m

    @motion_mode.deleter
    def motion_mode(self):
        del self._motion_mode

    def show(self):
        """ 
        Draw the figure in a window on the screen

        """
        plt.show()

    def _draw_frame(self, framedata):
        """ 
        Called sequentially for each frame of the animation.  Thus
        we must set our plot to look like it should for the given frame.

        Parameters
        ---------------------------------------
        framedata: int
          An integer between 0 and the number of frames, giving
          the current frame.

        """

        i = framedata

        # Make the canvas elements visible now
        # (We made them invisible in _init_draw() so that the first
        # frame of the animation wouldn't remain stuck on the canvas)
        for l in self.artists_to_be_drawn:
            l.set_visible(True)

        self.line1W.set_data(self.skeletons[:, 0, i],
                             self.skeletons[:, 1, i])
        self.line1W_head.set_data(self.skeletons[0, 0, i],
                                  self.skeletons[0, 1, i])
        self.line1C.set_data(self.vulva_contours[:, 0, i],
                             self.vulva_contours[:, 1, i])
        self.patch1E.center = (self.skeleton_centres[:, i])
        self.patch1E.angle = self.orientation[i]

        # Set the values of our annotation text in the main subplot:

        if self.motion_mode != None:
            if np.isnan(self.motion_mode[i]):
                self.patch1E.set_facecolor('w')
                self.annotation1a.set_text("Motion mode: {}".format('NaN'))
            else:
                # Set the colour of the ellipse surrounding the worm to a colour
                # corresponding to the current motion mode of the worm
                self.patch1E.set_facecolor(self.motion_mode_colours[
                    self.motion_mode[i]])
                # Annotate the current motion mode of the worm in text
                self.annotation1a.set_text("Motion mode: {}".format(
                    self.motion_mode_options[self.motion_mode[i]]))

        self.annotation1b.set_text("Frame {} of {}".format(i, self.num_frames))

        self.line2W.set_data(self.skeletons_centred[:, 0, i],
                             self.skeletons_centred[:, 1, i])
        self.line2W_head.set_data(self.skeletons_centred[0, 0, i],
                                  self.skeletons_centred[0, 1, i])
        self.line2C.set_data(self.skeletons_centred[:, 0, i] + (self.vulva_contours[:, 0, i] - self.skeletons[:, 0, i]),
                             self.skeletons_centred[:, 1, i] + (self.vulva_contours[:, 1, i] - self.skeletons[:, 1, i]))
        self.line2C2.set_data(self.skeletons_centred[:, 0, i] + (self.non_vulva_contours[:, 0, i] - self.skeletons[:, 0, i]),
                              self.skeletons_centred[:, 1, i] + (self.non_vulva_contours[:, 1, i] - self.skeletons[:, 1, i]))
        self.annotation2.xy = (self.skeletons_centred[0, 0, i],
                               self.skeletons_centred[0, 1, i])

        self.line3W.set_data(self.skeletons_rotated[:, 0, i],
                             self.skeletons_rotated[:, 1, i])
        self.line3W_head.set_data(self.skeletons_rotated[0, 0, i],
                                  self.skeletons_rotated[0, 1, i])

        self._drawn_artists = self.artists_to_be_drawn

    def new_frame_seq(self):
        """ 
        Returns an iterator that iterates over the frames 
        in the animation

        """
        return iter(range(self.normalized_worm.num_frames))

    def _init_draw(self):
        """ 
        Called when first drawing the animation.
        It is an abstract method in Animation, to be implemented here for
        the first time.

        """
        artists = [self.line1W, self.line1W_head, self.line1C,
                   self.line2W, self.line2W_head, self.line2C, self.line2C2,
                   self.line3W, self.line3W_head]

        for l in artists:
            l.set_data([], [])

        # Keep the drawing elements invisible for the first frame to avoid
        # the first frame remaining on the canvas
        for l in self.artists_to_be_drawn:
            l.set_visible(False)

        # TODO: figure out how to clear the non-line elements
        #artists.extend([self.annotation2, self.patch1E])

    def save(self, filename,
             file_title='C. elegans movement video',
             file_comment='C. elegans movement video from Schafer lab'):
        """ 
        Save the animation as an mp4.

        Parameters
        ---------------------------------------
        filename: string
          The name of the file to be saved as an mp4.

        file_title: string (optional)
          A title to be embedded in the file's saved metadata.

        file_comment: string (optional)
          A comment to be embedded in the file's saved metadata.

        Notes
        ---------------------------------------
        Requires ffmpeg or mencoder to be installed.  To install ffmpeg 
        on Windows, see:
        http://www.wikihow.com/Install-FFmpeg-on-Windows

        The code specifies extra_args=['-vcodec', 'libx264'], to ensure
        that the x264 codec is used, so that the video can be embedded 
        in html5.  You may need to adjust this for your system.  For more 
        information, see:
        http://matplotlib.sourceforge.net/api/animation_api.html

        """
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title=file_title,
                        artist='matplotlib',
                        comment=file_comment)
        writer = FFMpegWriter(fps=15, metadata=metadata)
        animation.TimedAnimation.save(self, filename,
                                      writer=writer, fps=config.FPS,
                                      extra_args=['-vcodec', 'libx264'])


def plot_frame_codes(normalized_worm):
    """ 
    Plot a pie chart of the frame codes of a normalized worm.  (An 
    attempt at replicating /documentation/Video%20Segmentation.gif)

    Parameters
    ---------------------------------------
    normalized_worm: NormalizedWorm instance

    """
    # TODO: someday make this pie chart look nicer with:
    # http://nxn.se/post/46440196846/

    nw = normalized_worm
    fc = nw.frame_codes

    # Create a dictionary of    frame code : frame code title   pairs
    fc_desc = {b[0]: b[2] for b in nw.frame_codes_descriptions}

    # A dictionary with the count for each frame code type
    counts = {i: np.bincount(fc)[i] for i in np.unique(fc)}

    # Display the pie chart
    patches, texts, autotexts = plt.pie(x=list(counts.values()),
                                        labels=list(fc_desc[d]
                                                    for d in np.unique(fc)),
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=['g', 'r', 'c', 'y', 'm'], labeldistance=1.2)
    plt.suptitle("Proportion of frames segmented")

    for t in texts:
        t.set_size('smaller')
    for t in autotexts:
        t.set_size('x-small')
