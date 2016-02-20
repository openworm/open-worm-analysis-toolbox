# -*- coding: utf-8 -*-
"""
An example showing how to use the open-worm-analysis-toolbox package to go from a
raw video .avi file to a fitness function result.

"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We must add .. to the path so that we can perform the
# import of open_worm_analysis_toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv

#%%


def example_worms(num_frames=1000):
    """
    Construct a simple set of worm positions over time, for testing purposes.

    Returns
    ------------
    Tuple of BasicWorm objects
        One moving, one not

    """
    # Start the worm at position 10000 for 1000 microns.
    skeleton_x = np.linspace(10000, 11000, mv.config.N_POINTS_NORMALIZED)
    skeleton_y = np.linspace(5000, 5000, mv.config.N_POINTS_NORMALIZED)
    # Shape is (49,2,1):
    skeleton_frame1 = np.rollaxis(np.dstack([skeleton_x,
                                             skeleton_y]),
                                  axis=0, start=3)
    # Shape is (49,2,1000):
    skeleton = np.repeat(skeleton_frame1, num_frames, axis=2)

    bw = mv.BasicWorm.from_skeleton_factory(skeleton)

    # Have the worm move in a square
    motion_overlay_x = np.linspace(0, 100, num_frames)
    motion_overlay_y = np.linspace(0, 0, num_frames)

    # Shape is (1000,2,1):   (DEBUG: we need it to be (1,2,1000))
    motion_overlay = np.rollaxis(np.dstack([motion_overlay_x,
                                            motion_overlay_y]),
                                 axis=0, start=3)

    # Broadcast the motion_overlay across axis 0 (i.e. apply the
    # motion_overlay evenly across all skeleton points)
    # skeleton_moving = skeleton + motion_overlay  # DEBUG

    return bw

#%%


def main():
    # TODO:
    # h_ventral_contour, h_dorsal_contour, video_info = \
    #    Kezhi_CV_Algorithm('test.avi')

    #experiment_info = mv.ExperimentInfo.from_CSV_factory('test.csv')

    #bw = BasicWorm.from_h_contour_factory(h_ventral_contour, h_dorsal_contour)
    #bw.video_info = video_info

    # TEMPORARY----------------------------
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    schafer_bw_file_path = os.path.join(
        base_path, "example_contour_and_skeleton_info.mat")
    bw = mv.BasicWorm.from_schafer_file_factory(schafer_bw_file_path)
    # -------------------------------------

    # TODO: get this line to work:
    #bw = example_worms()

    nw = mv.NormalizedWorm.from_BasicWorm_factory(bw)

    # DEBUG
    #wp = mv.NormalizedWormPlottable(nw, interactive=False)
    # wp.show()
    # return

    wf = mv.WormFeaturesDos(nw)

    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    control_path = os.path.join(base_path, '30m_wait', 'R')

    experiment_files = [wf, wf]
    control_files = get_matlab_filepaths(control_path)

    # Compute histograms on our files
    experiment_histograms = mv.HistogramManager(experiment_files)
    control_histograms = mv.HistogramManager(control_files)
    experiment_histograms.plot_information()

    # Compute statistics
    stat = mv.StatisticsManager(experiment_histograms, control_histograms)
    stat[0].plot(ax=None, use_alternate_plot=True)

    print("Nonparametric p and q values are %.2f and %.2f, respectively." %
          (stat.min_p_wilcoxon, stat.min_q_wilcoxon))


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

pd.set_option('expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
#fs = mv.WormFeaturesDos.get_feature_spec()
#%%
if __name__ == '__main__':
    start_time = mv.utils.timing_function()
    main()
    print("Time elapsed: %.2fs" %
          (mv.utils.timing_function() - start_time))
