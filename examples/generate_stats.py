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
import matplotlib.pyplot as plt

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


def plot_histogram(histogram):
    """
    Use matplotlib to plot a Histogram instance.
    
    Parameters
    -----------------------
    histogram: a Histogram instance
    
    """
    bins = histogram.bin_midpoints[:-1]  # because there are for some reason one too many
    y_values = histogram.counts
    
    plt.bar(bins, y_values)
    
    plt.xlabel('Counts')
    plt.ylabel(histogram.field)
    plt.title("Histogram of a worm's " + histogram.field)
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    
    plt.show()    
    
    
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
    experiment_histograms = mv.HistogramManager(experiment_files[:3])
    control_histograms = mv.HistogramManager(control_files[:3])

    plot_histogram(experiment_histograms.hists[1])
    
    # DEBUG: why does len(experiment_histograms.hists[0].bin_midpoints) - 
    #                 len(experiment_histograms.hists[0].counts) = 1?
    # it should equal 0!

    # TODO: test this:
    # stats = mv.StatsManager(experiment_histograms, control_histograms)

    # TODO:
    # now somehow display the stats to prove that we generated them!

    # TODO:
    # maybe compare to the segwormmatlabclasses-generated stats somehow?


if __name__ == '__main__':
    main()
