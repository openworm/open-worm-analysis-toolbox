# -*- coding: utf-8 -*-
"""
Insert description here
"""

class VideoInfo(object):
    """
    Encapsulates the metadata for a worm video
    
    """
    def __init__(self, video_name=None, fps=None):
        self.fps = fps
        self.video_name = video_name