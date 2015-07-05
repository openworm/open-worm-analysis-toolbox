# -*- coding: utf-8 -*-
"""
An example showing how to use the movement_validation package to go from a 
raw video .avi file to a fitness function result.

"""
import sys, os

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
import movement_validation as mv

#%%
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

    # Plot some histograms
    """
    fig = plt.figure(1)
    rows = 5; cols = 4
    #for i in range(0, 700, 100):
    for i in range(rows * cols):
        ax = plt.subplot2grid((rows, cols), (i // cols, i % cols))
        mv.Histogram.plot_versus(ax,
                                 experiment_histograms.hists[i],
                                 control_histograms.hists[i])
    #plt.tight_layout()
    """                                 
    stat = mv.StatisticsManager(experiment_histograms, control_histograms)

    print("Comparison p and q values are %.2f and %.2f, respectively." %
    #     (stat.p_worm, stat.q_worm))
          (stat.p_worm, 0))
    


#%%
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

#%%
def geppetto_to_features(minimal_worm_spec_path):
    pass


#%%
if __name__ == '__main__':
    start_time = mv.utils.timing_function()
    main()
    print("Time elapsed: %.2fs" % 
          (mv.utils.timing_function() - start_time))
        