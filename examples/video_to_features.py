# -*- coding: utf-8 -*-
"""
An example showing how to use the movement_validation package to go from a 
raw video to a feature file.

"""
import sys, os, warnings

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
import movement_validation as mv


def main():
    pass


def geppetto_to_features(minimal_worm_spec_path):
    pass


def video_to_features(video_path, output_path):
    """
    Go from a path to a .avi file to features

    Parameters
    --------------------
    vid_path: string
        Path to the video file.
    output_path: string
        Path to save the feature file to.  Will overwrite if necessary.
    
    Returns
    --------------------
    A StatisticsManager instance
    
    """
    # The segmentation algorithm requires a temporary folder to generate 
    # some intermediate files.  The folder will be deleted at the end.
    tmp_root = os.path.join(os.path.abspath(output_path), 'TEMP_DATA')

    video = mv.VideoFile(video_path, tmp_root)
    frame_iterator = video.obtain_frame_iterator()

    worm_frame_list = []

    for frame in frame_iterator:
        bool_frame = frame.process()
        segmented_worm_frame = mv.SegmentedWorm(bool_frame)
        worm_frame_list.append(segmented_worm_frame)
    
    worm_spec = mv.MinimalWormSpecification(worm_frame_list)
    
    worm_pre_features = mv.WormPreFeatures(worm_spec)
    
    nw = mv.NormalizedWorm(worm_spec.pre_features)
    
    if hasattr(video, 'video_info'):
        worm_features = mv.WormFeatures(nw, video.video_info)
    else:
        warnings.warn("VideoFile has not yet implemented video_info, " + \
                      "using default values.")
        #The frame rate is somewhere in the video info. Ideally this would all come
        #from the video parser eventually
        vi = mv.VideoInfo('Example Video File', 25.8398)
        worm_features = mv.WormFeatures(nw, vi)
    
    worm_features.write_to_disk(output_path)


if __name__ == '__main__':
    main()
