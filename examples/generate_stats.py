# -*- coding: utf-8 -*-
"""
Create an in-memory object, statistics_manager, that contains the 
statistics generated from comparing a set of 20 Feature .mat Files:
- 10 "Experiment" files and
- 10 "Control" files.

Notes
--------------
Formerly:
https://github.com/JimHokanson/SegwormMatlabClasses/
blob/master/%2Bseg_worm/%2Btesting/%2Bstats/t001_oldVsNewStats.m

"""
import sys, os, pickle
import matplotlib.pyplot as plt

# We must add .. to the path so that we can perform the
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import movement_validation as mv


def main():
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    root_path = os.path.join(base_path, '30m_wait')

    exp_histogram_manager, ctl_histogram_manager = \
        obtain_histograms(root_path, "pickled_histograms.dat")

    print("Done with Histogram generation.  Now let's calculate statistics.")

    statistics_manager = \
        mv.StatisticsManager(exp_histogram_manager, ctl_histogram_manager)

    print("Comparison p and q values are %.2f and %.2f, respectively." %
          (statistics_manager.min_p_wilcoxon, 
           statistics_manager.min_q_wilcoxon))

    statistics_manager.plot()
    plt.savefig('michael.png')
    
    """
    # Plot the p-values, ranked.
    # TODO: add a line at the 0.01 and 0.05 thresholds, with annotation for
    #       the intercept.
    plt.plot(np.sort(statistics_manager.p_wilcoxon_array), 
             label="Sorted p-values by Wilcoxon's signed rank test")
    plt.plot(np.sort(statistics_manager.q_wilcoxon_array),
             label="Sorted q-values by Wilcoxon's signed rank test")
    plt.ylabel("Probability", fontsize=10)
    plt.xlabel("Feature", fontsize=10)
    plt.legend(loc='best', shadow=True)
    #plt.gca().set_axis_bgcolor('m')
    plt.show()
    """    
    
    # TODO:
    # maybe compare to the segwormmatlabclasses-generated stats somehow?

    # TODO:
    # visualize the data in a grid
    # http://stackoverflow.com/questions/19407950

    # Y-axis is features, labeled
    # X-axis is worm videos
    # then list the p and q values
    # List if the mean is vailable or golor red if not.



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


def obtain_histograms(root_path, pickle_file_path):
    """
    Compute histograms for 10 experiment and 10 control feature files.
    
    Uses Python's pickle module to save results to disk to save time
    on future times the function is run.
    
    Parameters
    ----------
    root_path: string
        A path that has two subfolders, L and R, containing some .mat files,
        for the experiment and control samples, respectively.
    pickle_file_path: string
        A relative path, to the pickle file that has serialized the 
        histograms.  This is generally found in the examples folder 
        if one wishes to delete it to rerun the code fresh.
    
    Returns
    -------
    exp_histogram_manager, ctl_histogram_manager
        Both instances of HistogramManager
    
    """
    if os.path.isfile(pickle_file_path):
        print("Found a pickled version of the histogram managers "
              "at:\n%s\n" % pickle_file_path + "Let's attempt to "
              "unpickle rather than re-calculate, to save time...")
        with open(pickle_file_path, "rb") as pickle_file:
            exp_histogram_manager = pickle.load(pickle_file)
            ctl_histogram_manager = pickle.load(pickle_file)
    else:
        print("Could not find a pickled version of the histogram "
              "managers so let's calculate from scratch and then pickle")

        experiment_path = os.path.join(root_path, 'L')
        control_path = os.path.join(root_path, 'R')

        experiment_files = get_matlab_filepaths(experiment_path)
        control_files = get_matlab_filepaths(control_path)

        # We need at least 10 files in each
        assert(len(experiment_files) >= 10)
        assert(len(control_files) >= 10)

        # Compute histograms on our files
        exp_histogram_manager = mv.HistogramManager(experiment_files[:10])
        ctl_histogram_manager = mv.HistogramManager(control_files[:10])
        
        # Store a pickle file in the same folder as this script 
        # (i.e. movement_validation/examples/)
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(exp_histogram_manager, pickle_file)
            pickle.dump(ctl_histogram_manager, pickle_file)

    print("Experiment has a total of " + \
          str(len(exp_histogram_manager.merged_histograms)) + " histograms")

    return exp_histogram_manager, ctl_histogram_manager


if __name__ == '__main__':
    main()
