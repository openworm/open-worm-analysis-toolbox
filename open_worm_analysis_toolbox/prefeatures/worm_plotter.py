# -*- coding: utf-8 -*-
"""
Plotting some of the calculated values of open-worm-analysis-toolbox for
illustrative purposes.

Usage
----------------------------

Within the open-worm-analysis-toolbox repository, the plotting code is encapsulated
in the NormalizedWormPlottable class.

If you already have an instance nw of the NormalizedWorm class, you could
plot it as follows:

    wp = NormalizedWormPlottable(nw, interactive=False)
    wp.show()

Another plotting function is the number of dropped frames in a given
normalized worm's segmented video. To show a pie chart breaking down this
information, you could run the following:

    plot_frame_codes(nw)

I note in the code for NormalizedWormPlottable that the visualization module
follows the animated subplots example in inheriting from TimedAnimation rather
than using PyLab shell-type calls to construct the animation, which is
the more typical approach. I used this approach so that it would be
extensible into other visualization possibilities.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation


class NormalizedWormPlottable(animation.TimedAnimation):
    """
    A class that renders matplotlib plots of worm measurement and
    feature data.

    This follows the [animated subplots example]
    (http://matplotlib.org/1.3.0/examples/animation/subplots.html)
    in inheriting from TimedAnimation rather than
    using PyLab-shell-type calls to construct the animation.

    """

    def __init__(self, normalized_worm, motion_mode=None, interactive=False,
                 interpolate_nan_frames=False):
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
            is called.  this allows NormalizedWormPlottable to be instantiated
            without just automatically being displayed.  Instead, the user
            must call NormalizedWormPlottable.show() to have plt.show() be
            called.

        interpolate_nan_frames: interpolate the flickering nan frames
            Note: this is currently not implemented

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
        # A NormalizedWormPlottable instance can be instantiated to be
        # interactive, or by default it is set to NOT interactive, which
        # means that NormalizedWormPlottable.show() must be called to get
        # it to display.
        plt.interactive(interactive)

        self._paused = False

        # 1. set up the data to be used
        self.nw = normalized_worm
        self.motion_mode = motion_mode
        self.motion_mode_options = {-1: 'backward',
                                    0: 'paused',
                                    1: 'forward'}
        self.motion_mode_colours = {-1: 'b',      # blue
                                    0: 'r',       # red
                                    1: 'g'}       # green

        # 2. create the figure
        fig = plt.figure(figsize=(5, 5))

        # We have blit=True, so the animation only redraws the elements that
        # have changed.  This means that if the window is resized, everything
        # other than the plot area will be black.  To fix this, here we have
        # matplotlib explicitly redraw everything if a resizing event occurs.
        # DEBUG: this actually doesn't appear to work.
        def refresh_plot(event):
            print("refresh_plot called")

            # We must reset this or else repetitive expanding and contracting
            # of the window will cause the view to zoom in on the subplots
            self.set_axes_extents()

            fig.canvas.draw()

        self.refresh_connection_id = fig.canvas.mpl_connect('resize_event',
                                                            refresh_plot)

        fig.suptitle('C. elegans attributes', fontsize=20)

        # 3. Add the subplots
        self.ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        self.ax1.set_title('Position')
        self.ax1.plot(self.nw.skeleton[0, 0, :], self.nw.skeleton[0, 1, :])
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        #ax1.set_aspect(aspect='equal', adjustable='datalim')
        # ax1.set_autoscale_on()

        self.annotation1a = \
            self.ax1.annotate("(motion mode data not available)",
                              xy=(-10, 10), xycoords='axes points',
                              horizontalalignment='right',
                              verticalalignment='top',
                              fontsize=10)

        self.annotation1b = \
            self.ax1.annotate("bottom right (points)",
                              xy=(-10, 10), xycoords='axes points',
                              horizontalalignment='right',
                              verticalalignment='bottom',
                              fontsize=10)

        # WIDGETS
        # [left,botton,width,height] as a proportion of
        # figure width and height
        frame_slider_axes = fig.add_axes([0.2, 0.02, 0.33, 0.02],
                                         axisbg='lightgoldenrodyellow')
        self.frame_slider = Slider(frame_slider_axes, label='Frame#',
                                   valmin=1, valmax=self.nw.num_frames,
                                   valinit=1, valfmt=u'%d')

        def frame_slider_update(val):
            print("Slider value: %d" % round(val, 0))
            self.frame_seq = self.new_frame_seq(int(round(val, 0)))

            fig.canvas.draw()

        self.frame_slider.on_changed(frame_slider_update)

        button_axes = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        #ax4 = plt.subplot2grid((3, 3), (2, 2))
        button = Button(button_axes, 'Pause',
                        color='lightgoldenrodyellow', hovercolor='0.975')

        def pause(event):
            if not self._paused:
                self._stop()
            else:
                self.event_source = fig.canvas.new_timer()
                self.event_source.interval = self._interval
                self._start()

            # Toggle the paused state
            self._paused = 1 - self._paused

        button.on_clicked(pause)

        self.ax2 = plt.subplot2grid((3, 3), (0, 2))
        self.ax2.set_title('Morphology')
        self.ax2.set_aspect(aspect='equal', adjustable='datalim')
        self.annotation2 = self.ax2.annotate(
            "Worm head", xy=(
                0, 0), xycoords='data', xytext=(
                10, 10), textcoords='data', arrowprops=dict(
                arrowstyle="fancy", connectionstyle="arc3,rad=.2"))

        self.ax3 = plt.subplot2grid((3, 3), (1, 2))
        self.ax3.set_title('Orientation-free')
        # DON'T USE set_xbound, it changes dynmically
        self.ax3.set_aspect(aspect='equal', adjustable='datalim')

        # Length and Area over time
        self.ax4a = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=2)
        self.ax4a.plot(self.nw.length, 'o-')
        self.ax4a.set_title('Length and Area over time')
        self.ax4a.set_ylabel('Microns')

        self.ax4b = self.ax4a.twinx()
        self.ax4b.plot(self.nw.area, 'xr-')
        self.ax4b.set_ylabel('Microns ^ 2')

        # Widths and Angles
        self.ax5a = plt.subplot2grid((3, 3), (2, 2))
        self.ax5a.set_title('Widths and Angles')
        self.ax5a.set_xlabel('Skeleton point')
        self.widths = Line2D([], [])
        self.ax5a.set_ylabel('Microns')

        self.ax5b = self.ax5a.twinx()
        self.angles = Line2D([], [])
        self.ax5b.set_ylabel('Degrees')

        # 4. create Artist objects
        self.time_marker = Line2D([], [])

        self.line1W = Line2D([], [], color='green',
                             linestyle=':',
                             marker='o', markersize=5)
        self.line1W_head = Line2D([], [], color='red',
                                  linestyle=':',
                                  marker='o', markersize=7)
        self.line1C = Line2D([], [], color='yellow',
                             linestyle=':',
                             marker='o', markersize=5)
        self.patch1E = Ellipse(xy=(0, 0), width=1000, height=500,
                               angle=0, alpha=0.3)

        self.line2W = Line2D([], [], color='black', marker='o', markersize=5)
        self.line2W_head = Line2D([], [], color='red',
                                  linestyle=':',
                                  marker='o', markersize=7)
        self.line2C = Line2D([], [], color='blue')
        self.line2C2 = Line2D([], [], color='orange')

        self.line3W = Line2D([], [], color='black', marker='o', markersize=5)
        self.line3W_head = Line2D([], [], color='red',
                                  linestyle=':',
                                  marker='o', markersize=7)

        self.artists_with_data = [self.line1W, self.line1W_head, self.line1C,
                                  self.line2W, self.line2W_head, self.line2C,
                                  self.line2C2, self.line3W, self.line3W_head,
                                  self.widths, self.angles, self.time_marker]

        # This list is a superset of self.artists_with_data
        self.artists_to_be_drawn = ([self.patch1E, self.annotation1a,
                                     self.annotation1b, self.annotation2] +
                                    self.artists_with_data)

        self.set_axes_extents()

        # 5. assign Artist objects to the relevant subplot
        self.ax1.add_line(self.line1W)
        self.ax1.add_line(self.line1W_head)
        self.ax1.add_line(self.line1C)
        self.ax1.add_artist(self.patch1E)

        self.ax2.add_line(self.line2W)
        self.ax2.add_line(self.line2W_head)
        self.ax2.add_line(self.line2C)
        self.ax2.add_line(self.line2C2)

        self.ax3.add_line(self.line3W)
        self.ax3.add_line(self.line3W_head)

        self.ax4a.add_line(self.time_marker)

        self.ax5a.add_line(self.widths)
        self.ax5b.add_line(self.angles)

        # So labels don't overlap:
        # plt.tight_layout()

        # 6. call the base class __init__

        # TimedAnimation draws a new frame every *interval* milliseconds.
        # so this is how we convert from FPS to interval:
        interval = 1000 / self.nw.video_info.fps

        return animation.TimedAnimation.__init__(self,
                                                 fig,
                                                 interval=interval,
                                                 blit=True)

    def set_axes_extents(self):
        # DON'T USE set_xbound; it changes dynamically
        self.ax1.set_xlim(self.nw.position_limits(0))
        self.ax1.set_ylim(self.nw.position_limits(1))
        self.ax2.set_xlim((-500, 500))
        self.ax2.set_ylim((-500, 500))
        self.ax3.set_xlim((-800, 800))
        self.ax3.set_ylim((-500, 500))
        self.ax4a.set_xlim((0, self.nw.num_frames))
        self.ax5a.set_xlim((0, 49))
        self.ax5a.set_ylim((0, np.nanmax(self.nw.widths)))
        self.ax5b.set_ylim((np.nanmin(self.nw.angles),
                            np.nanmax(self.nw.angles)))

    def set_frame_data(self, frame_index):
        self._current_frame = frame_index

        i = frame_index
        if self._paused:
            return

        self.line1W.set_data(self.nw.skeleton[:, 0, i],
                             self.nw.skeleton[:, 1, i])
        self.line1W_head.set_data(self.nw.skeleton[0, 0, i],
                                  self.nw.skeleton[0, 1, i])
        self.line1C.set_data(self.nw.ventral_contour[:, 0, i],
                             self.nw.ventral_contour[:, 1, i])
        self.patch1E.center = (self.nw.centre[:, i])  # skeleton centre
        self.patch1E.angle = self.nw.angle[i]    # orientation

        # Set the values of our annotation text in the main subplot:

        if self.motion_mode is not None:
            if np.isnan(self.motion_mode[i]):
                self.patch1E.set_facecolor('w')
                self.annotation1a.set_text("Motion mode: {}".format('NaN'))
            else:
                # Set the colour of the ellipse surrounding the worm to a
                # colour corresponding to the current motion mode of the worm
                self.patch1E.set_facecolor(self.motion_mode_colours[
                    self.motion_mode[i]])
                # Annotate the current motion mode of the worm in text
                self.annotation1a.set_text("Motion mode: {}".format(
                    self.motion_mode_options[self.motion_mode[i]]))

        self.annotation1b.set_text("Frame {} of {}".format(i,
                                                           self.num_frames))

        self.line2W.set_data(self.nw.centred_skeleton[:, 0, i],
                             self.nw.centred_skeleton[:, 1, i])
        self.line2W_head.set_data(self.nw.centred_skeleton[0, 0, i],
                                  self.nw.centred_skeleton[0, 1, i])
        self.line2C.set_data(self.nw.centred_skeleton[:, 0, i] +
                             (self.nw.ventral_contour[:, 0, i] - self.nw.skeleton[:, 0, i]),
                             self.nw.centred_skeleton[:, 1, i] +
                             (self.nw.ventral_contour[:, 1, i] - self.nw.skeleton[:, 1, i]))
        self.line2C2.set_data(self.nw.centred_skeleton[:, 0, i] +
                              (self.nw.dorsal_contour[:, 0, i] - self.nw.skeleton[:, 0, i]),
                              self.nw.centred_skeleton[:, 1, i] +
                              (self.nw.dorsal_contour[:, 1, i] - self.nw.skeleton[:, 1, i]))
        self.annotation2.xy = (self.nw.centred_skeleton[0, 0, i],
                               self.nw.centred_skeleton[0, 1, i])

        self.line3W.set_data(self.nw.orientation_free_skeleton[:, 0, i],
                             self.nw.orientation_free_skeleton[:, 1, i])
        self.line3W_head.set_data(self.nw.orientation_free_skeleton[0, 0, i],
                                  self.nw.orientation_free_skeleton[0, 1, i])

        self.widths.set_data(np.arange(49), self.nw.widths[:, i])
        self.angles.set_data(np.arange(49), self.nw.angles[:, i])

        # Draw a vertical line to mark the passage of time
        self.time_marker.set_data([i, i], [0, 10000])

    @property
    def num_frames(self):
        """
          Return the total number of frames in the animation

        """
        return self.nw.num_frames

    def show(self):
        """
        Draw the figure in a window on the screen

        """
        plt.show()

    def _draw_frame(self, frame_index):
        """
        Called sequentially for each frame of the animation.  Thus
        we must set our plot to look like it should for the given frame.

        Parameters
        ---------------------------------------
        frame_index: int
          An integer between 0 and the number of frames, giving
          the current frame.

        """
        # Make the canvas elements visible now
        # (We made them invisible in _init_draw() so that the first
        # frame of the animation wouldn't remain stuck on the canvas)
        for l in self.artists_to_be_drawn:
            l.set_visible(True)

        self.set_frame_data(frame_index)

        self._drawn_artists = self.artists_to_be_drawn

    def new_frame_seq(self, start_frame=0):
        """
        Returns an iterator that iterates over the frames
        in the animation

        Parameters
        ---------------
        start_frame: int
            Start the sequence at this frame then loop back around
            to start_frame-1.

        """
        s = start_frame
        n = self.num_frames
        return iter(np.mod(np.arange(s, s + n), n))

    def _init_draw(self):
        """
        Called when first drawing the animation.
        It is an abstract method in Animation, to be implemented here for
        the first time.

        """
        for l in self.artists_with_data:
            l.set_data([], [])

        # Keep the drawing elements invisible for the first frame to avoid
        # the first frame remaining on the canvas
        for l in self.artists_to_be_drawn:
            l.set_visible(False)

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
        fps = self.nw.video_info.fps
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title=file_title,
                        artist='matplotlib',
                        comment=file_comment)
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        animation.TimedAnimation.save(self, filename,
                                      writer=writer, fps=fps,
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
    fc = nw.frame_code

    # Create a dictionary of    frame code : frame code title   pairs
    # TODO: this currently doesn't work since I no longer load frame_codes
    # into NormalizedWorm.  The data is at documentation/frame_codes.csv
    # though
    #fc_desc = {b[0]: b[2] for b in nw.frame_codes_descriptions}

    # A dictionary with the count for each frame code type
    counts = {i: np.bincount(fc)[i] for i in np.unique(fc)}

    # Display the pie chart
    patches, texts, autotexts = plt.pie(x=list(counts.values()),
                                        # labels=list(fc_desc[d]
                                        #            for d in np.unique(fc)),
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=['g', 'r', 'c', 'y', 'm'],
                                        labeldistance=1.2)
    plt.suptitle("Proportion of frames segmented")

    for t in texts:
        t.set_size('smaller')
    for t in autotexts:
        t.set_size('x-small')
