# -*- coding: utf-8 -*-
"""
An example showing how to use the movement_validation package to go from a 
raw video .avi file to a fitness function result.

"""
import sys, os, warnings

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
import movement_validation as mv


def main():
    # TODO:
    #h_ventral_contour, h_dorsal_contour, video_info = \
    #    Kezhi_CV_Algorithm('test.avi')

    #experiment_info = mv.ExperimentInfo.from_CSV_factory('test.csv')
   
    #bw = BasicWorm.from_h_contour_factory(h_ventral_contour, h_dorsal_contour)
    #bw.video_info = video_info

    # TEMPORARY----------------------------
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    schafer_bw_file_path = os.path.join(base_path, 
                                     "example_contour_and_skeleton_info.mat")  
    bw = mv.BasicWorm.from_schafer_file_factory(schafer_bw_file_path)
    # -------------------------------------

    nw = mv.NormalizedWorm.from_BasicWorm_factory(bw)

    wf = mv.WormFeatures(nw)

    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    control_path = os.path.join(base_path, '30m_wait', 'R')
    
    experiment_files = [wf]
    control_files = get_matlab_filepaths(control_path)

    # Compute histograms on our files
    experiment_histograms = mv.HistogramManager(experiment_files)
    control_histograms = mv.HistogramManager(control_files)

    # TODO: show all histograms on a tabular format on an HTML page.
    for i in range(1):
        experiment_histograms.hists[i].plot_versus(control_histograms.hists[i])     

    
def get_matlab_filepaths(root_path):
    """
    Recursively traverses from root_path to find all .mat files
    Returns a list of .mat files, with full path

    Parameters
    -----------------------
    root_path: string
        The absolute path to start searching from

    """
    matlab_filepaths = []
    for root, dirs, files in os.walk(root_path):
        mat_files = [f for f in files if f[-4:] == '.mat']
        for f in mat_files:
            matlab_filepaths.append(os.path.join(root, f))

    return matlab_filepaths

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
    start = mv.utils.timing_function()
    main()
    print("Time elapsed: %.2fs" % 
          (utils.timing_function() - start_time))
        