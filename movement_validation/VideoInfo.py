# -*- coding: utf-8 -*-
"""
This code may eventually change or may take on an inheritance structure as we
start to clarify differences between simulated and real worms
"""

class VideoInfo(object):

    def __init__(self,video_name,fps):
        self.fps = fps
        self.video_name = video_name