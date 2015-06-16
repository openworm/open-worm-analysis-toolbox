# -*- coding: utf-8 -*-
"""
Insert description here
"""

class VideoInfo(object):
    """
    Encapsulates the metadata for a worm video
    
    """
    def __init__(self, video_name, fps, 
                 height=None, width=None, 
                 microns_per_pixel_x=None, 
                 microns_per_pixel_y=None,
                 fourcc=None,
                 length_in_seconds=None,
                 length_in_frames=None):

        # Mandatory
        self.fps = fps
        self.video_name = video_name

        # Optional (i.e. not used in future processing?)
        # TODO: use this info for pixel-to-micron scaling
        self.height = height
        self.width = width
        self.microns_per_pixel_x = microns_per_pixel_x
        self.microns_per_pixel_y = microns_per_pixel_y
        # Not sure what this is, but it's in the example Schafer feature file
        # Likely the "four-character code" 
        # (https://en.wikipedia.org/wiki/FourCC)
        # "One of the most well-known uses of FourCCs is to identify the 
        #  video codec used in AVI files."
        # - Michael Currie
        self.fourcc = fourcc

        # TODO: We'll have to do some integrity checks since 
        # length_in_frames = len(skeleton[0,0,:]) = len(contour[0,0,:]) and
        # length_in_frames = fps * length_in_seconds
        self.length_in_seconds = length_in_seconds
        self.length_in_frames = length_in_frames


        
class ExperimentInfo(object):
    def __init__(self):
        pass
        # just have dictionaries of information, on:
        # environment
        # worm
        # lab
        #
        # I need ventral_mode !!!