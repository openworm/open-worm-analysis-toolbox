# -*- coding: utf-8 -*-
"""
Metadata (i.e. not frame-by-frame information)

The following attributes are used in downstream processing:

fps:
    - several features

frame_code:
    - 1, 105, 106 is used in posture.coils
    -            2 is used in locomotion.turns

ventral_mode:
    Needed to sign (give + or -) several features:
        - locomotion.velocity, the motion direction.
        - The amplitude and frequency of foraging.bends
        - path.curvature

"""
import os
import numpy as np
import pandas as pd

from .. import config


class VideoInfo(object):
    """
    Metadata associated with a a worm video.

    video_name
    fps
    height
    width
    microns_per_pixel_x
    microns_per_pixel_y
    fourcc
    length_in_seconds
    length_in_frames

    frame_code : numpy array of codes for each frame of the video
    frame_code_info : Descriptions of the frame codes, lazy-loaded from csv
    ventral_mode : int
        The ventral side mode:
        0 = unknown
        1 = clockwise
        2 = anticlockwise
    video_type: in ['Schafer Lab', 'Not specified']

    This can also be used as a base class for a new team that might need
    different annotations on their videos.

    """

    def __init__(self, video_name='', fps=None,
                 height=None, width=None,
                 microns_per_pixel_x=None,
                 microns_per_pixel_y=None,
                 fourcc=None,
                 length_in_seconds=None,
                 length_in_frames=None):

        # Mandatory
        if fps is None:
            self.fps = config.DEFAULT_FPS
        else:
            self.fps = fps
        self.video_name = video_name

        # Optional (i.e. not used in future processing?)
        # TODO: use this info for pixel-to-micron scaling
        self.height = height
        self.width = width
        self.microns_per_pixel_x = microns_per_pixel_x
        self.microns_per_pixel_y = microns_per_pixel_y
        # The "four-character code"
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

        # DEBUG: (Note from @MichaelCurrie:)
        # This should be set by the normalized worm file, since each
        # worm subjected to an experiment is manually examined to find the
        # vulva so the ventral mode can be determined.  Here we just set
        # the ventral mode to a default value as a stopgap measure
        self.ventral_mode = config.DEFAULT_VENTRAL_MODE
        self.video_type = 'Not specified'

    def set_ventral_mode(self, ventral_side):
        '''
        Set the ventral side mode. Valid options are "clockwise", "anticlockwise" and "unknown".
        '''
        if ventral_side == 'clockwise':
            self.ventral_mode = 1
        elif ventral_side == 'anticlockwise':
            self.ventral_mode = 2
        elif ventral_side == 'unknown':
            self.ventral_mode = 0
        else:
            raise ValueError('{} is not a recognizable ventral_mode.'.format(ventral_side))


    @staticmethod
    def sniff_video_properties(file_path):
        """
        A utility method to find a video's resolution, frame rate, codec, etc
        in case this isn't passed to us and we need to populate it here.

        """
        # TODO
        pass

    @property
    def is_stage_movement(self):
        """
        Returns a mask for all frames with frame code == 2, that is,
        with a stage movement.
        """
        return self.frame_code == 2

    @property
    def frame_code_info(self):
        """
        Frame code descriptions

        I'd like to make this a static property but I don't think
        that's possible.

        """
        try:
            return self._frame_code_info
        except AttributeError:
            self.load_frame_code_info()
            return self._frame_code_info

    def load_frame_code_info(self):
        """
        Load the frame code descriptions

        """
        # Obtain this computer's path to
        # open-worm-analysis-toolbox\documentation\frame_codes.csv
        cur_file_folder = os.path.dirname(__file__)
        package_path = os.path.abspath(os.path.dirname(cur_file_folder))
        frame_codes_path = os.path.join(package_path,
                                        'documentation',
                                        'frame_codes.csv')

        # Load frame code information
        self._frame_code_info = pd.read_csv(frame_codes_path,
                                            delimiter=';',
                                            quotechar="'")
        # Convert the 'Frame Codes' column, which is all int, to int.
        self._frame_code_info = \
            self._frame_code_info.convert_objects(convert_numeric=True)

    @property
    def is_segmented(self):
        """
        Returns a 1-d boolean numpy array of whether
        or not, frame-by-frame, the given frame was segmented

        """
        return self.frame_code == 1

    @property
    def segmentation_status(self):
        """
        Deprecated in favour of using self.frame_code directly.

        A numpy array of characters 's', 'm', 'd', 'f', where:
            s = Segmented           (aka frame code 1)
            m = Stage movement      (aka frame code 2)
            d = Dropped frame       (aka frame code 3)
            f = Segmentation failed (aka frame codes 100+)

        """
        try:
            return self._segmentation_status
        except AttributeError:
            s = self.frame_code == 1
            m = self.frame_code == 2
            d = self.frame_code == 3

            self._segmentation_status = np.empty(self.num_frames, dtype='<U1')
            for frame_index in range(self.num_frames):
                if s[frame_index]:
                    self._segmentation_status[frame_index] = 's'
                elif m[frame_index]:
                    self._segmentation_status[frame_index] = 'm'
                elif d[frame_index]:
                    self._segmentation_status[frame_index] = 'd'
                else:
                    self._segmentation_status[frame_index] = 'f'

            return self._segmentation_status


class ExperimentInfo(object):

    def __init__(self):
        pass
        # just have dictionaries of information, on:
        # environment
        # worm
        # lab
        #
        # I need ventral_mode !!!
