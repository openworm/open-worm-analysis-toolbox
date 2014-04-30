(Written in March 2014 by @MichaelCurrie)

Within the [movement_validation repository](https://github.com/openworm/movement_validation), the plotting code is encapsulated in the WormPlotter class.

If you already have an instance nw of the NormalizedWorm class, you could plot it as follows:

wp = wormpy.WormPlotter(nw, interactive=False)

wp.show()

Another plotting function is the number of dropped frames in a given normalized worm's segmented video.  To show a pie chart breaking down this information, you could run the following:

wormpy.plot_frame_codes(nw)

Creating such an instance nw of the NormalizedWorm class requires access to worm data files.  The easiest way to obtain such data files is to join the shared DropBox folder, worm_data, where we've stored some examples.  I've sent you an invitation by email; let me know if you did not get it.

Once you've joined the worm_data shared folder, you'll need to clone and configure the movement_validation repository for yourself.  Here is an installation guide for you.

I note in the code for WormPlotter that the visualization module follows the animated subplots example in inheriting from TimedAnimation rather than using PyLab shell-type calls to construct the animation, which is the more typical approach.  I used this approach so that it would be extensible into other visualizations possibilities.
