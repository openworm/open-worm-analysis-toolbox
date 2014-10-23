# -*- coding: utf-8 -*-
"""
This should replicate the behaviour of the compute "new" histograms code from
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Btesting/%2Bstats/t001_oldVsNewStats.m

We are just trying to achieve the creation of an in-memory object that contains
the statistics generated from comparing a set of 20 Feature .mat Files:
- 10 "Experiment" files and
- 10 "Control" files.

"""


import sys, os

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
import movement_validation as mv


def main():
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)

    root_path = os.path.join(base_path, '30m_wait')
    
    compute_histograms(root_path)    
    
    # TODO: do something with the histograms
    pass


def get_matlab_filepaths(root_path):
    """
    Recursively traverses from root_path to find all .mat files
    Returns a list of .mat files, with full path

    """

    matlab_filepaths = []
    for root, dirs, files in os.walk(root_path):
        mat_files = [f for f in files if f[-4:] == '.mat']
        for f in mat_files:
            matlab_filepaths.append(os.path.join(root, f))

    return matlab_filepaths


def compute_histograms(root_path):
    """
    Compute histograms for 10 experiment and 10 control feature files
    
    """
    experiment_path = os.path.join(root_path, 'L')
    control_path = os.path.join(root_path, 'R')

    experiment_files = get_matlab_filepaths(experiment_path)
    control_files = get_matlab_filepaths(control_path)

    # We need at least 10 files in each
    assert(len(experiment_files) >= 10)
    assert(len(control_files) >= 10)

    # Compute histograms on our files
    # DEBUG: I've dialled this down to just 3 files each, for speed.  Really
    #        this should be [:10] each
    histogram_manager_experiment = mv.HistogramManager(experiment_files[:3])
    histogram_manager_control = mv.HistogramManager(control_files[:3])

    # TODO: Translate from matlab:
    # % stats_manager = seg_worm.stats.manager(hist_man_exp,hist_man_ctl);

    # TODO:
    # now somehow display the stats to prove that we generated them!
    # maybe compare to the segwormmatlabclasses-generated stats somehow?


if __name__ == '__main__':
    main()
