# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:47:16 2015

@author: mcurrie
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pickle

sys.path.append('..')
import movement_validation as mv

# Use pandas to load the features specification
feature_spec_path = os.path.join('..', 'documentation', 'database schema',
                                 'Features Specifications.xlsx')

feature_spec = pd.ExcelFile(feature_spec_path).parse('FeatureSpecifications')


def prepare_plots():

    # Assigned numbers are the sub-extended feature ID.
    # any features that have feature_type = 'movement' use the standard
    # 6-figure movement type plot.

    # PAGE 0: First an introductory page
    # Table of contents, description of experiment and control, 
    # Heatmap of phenotype
    # a heatmap of available features for each of the worms
    # 
    page = [None]*94
    page[0] = 'Table of Contents.'

    # maps (row,col) to sub-extended feature ID
    page[1] = {(0,0): '', (0,1):  7, (0,2): 12, (0,3): 14, (0,4): 16,
               (1,0):  5, (1,1):  6, (1,2): 17, (1,3): 19, (0,4): 21,
               (2,0): 26, (2,1):  8, (2,2): 50, (2,3): 51, (0,4): 52,
               (3,0): 22, (3,1): 24, (3,2): 53, (3,3): 54, (0,4): 55}
    page[1][(0,0)] = 'legend'

    page[2] = {(0,0): 48, (0,1): 49, (0,2): 39, (0,3): 40, (0,4): 41,
               (1,0): 32, (1,1): 33, (1,2): 44, (1,3): 45, (0,4): 46,
               (2,0): 34, (2,1): 35, (2,2): 63, (2,3): 64, (0,4): 65,
               (3,0): 36, (3,1): 37, (3,2): 68, (3,3): 69, (0,4): 70}

    page[3] = {(0,0): 56, (0,1): 28, (0,2): 58, (0,3): 59, (0,4): 60,
               (1,0):  2, (1,1): 73, (1,2): 74, (1,3): 75, (0,4): 76,
               (2,0):  3, (2,1): 80, (2,2): 81, (2,3): 82, (0,4): 83,
               (3,0):  4, (3,1): 87, (3,2): 88, (3,3): 89, (0,4): 90}

    # THE SIX-FIGURE SECTIOIN    
    
    for i in range(4,27):
        # Movement features
        page[i] = i + 1

    page[27] = 'bend count'   #(features[58:63])
    page[28] = 'coil time'  #(features[58:63])

    for i in range(29,38):
        # More movement features
        page[i] = i

    page[38] = 'locomotion.motion_events.forward'
    page[39] = 'locomotion.motion_events.paused'
    page[40] = 'locomotion.motion_events.backward'

    for i in range(41,52):
        # More movement features
        page[i] = i - 3

    page[52] = 50 # Crawling amplitude
    page[53] = 51
    page[54] = 52

    page[55] = 49 # Foraging speed

    page[56] = 53 # Crawling frequency
    page[57] = 54
    page[58] = 55

    page[59] = 'Omega turns (just four plots)'
    page[60] = 'Upsilon turns (just four plots)'
    page[61] = 56   # ANOTHER 6-figure plot (Path range)
    page[62] = [1,2,3,4]  # Worm dwelling four-grid; worm, had, midbody, tail.
    page[63] = 67   # ANOTHER 6-figure plot (Path curvature)

    # PATH PLOTS, all annotated with omegas and coils:
    # i.e. exactly 30 pages of path and dwelling charts

    # 10 pages for Midbody/Head/Tail colors
    page[64] = 'two charts, 24 experiment, 24 control, blended'
    page[65] = ('two charts, 24 experiment, 24 control, split out into '
               '24 little plots, 5 columns, 6 rows')
    page[66] = '6 plots of experiment worms 0-6'
    page[67] = '6 plots of experiment worms 6-12'
    page[68] = '6 plots of experiment worms 12-18'
    page[69] = '6 plots of experiment worms 18-24'
    page[70] = '6 plots of control worms 0-6'
    page[71] = '6 plots of control worms 6-12'
    page[72] = '6 plots of control worms 12-18'
    page[73] = '6 plots of control worms 18-24'

    page[74:84] = 'SAME 10 pages AGAIN, BUT NOW FOR MIDBODY SPEED'
    page[84:94] = 'SAME 10 pages AGAIN, but now for FORAGING AMPLITUDE'

    # METHODS TABLE OF CONTENTS
    
    
    """
    
    for feature in master_feature_list:
        title = feature['title']
        #three legends:
            #experiment / control
            #backward / forward / paused
            #q-values
        if feature['is_time_series']:
            # Show 2 rows and 3 columns of plots, with:
            rows = 2
            cols = 3

            (0,0) is motion_type = 'all'
                experiment is brown
            (1,0) is motion_type = 'forward'
                experiment is purple
            (1,1) is motion_type = 'paused'
                experiment is green
            (1,2) is motion_type = 'backward'
                experiment is blue
            ax = plt.subplot2grid((rows, cols), (i // cols, i % cols))
    
    # Then link to feature descriptions
    # perhaps this could be stored in the FeatureInfo table
    
    """
    return page

def prepare_pages():
    # Assigned numbers are the 'sub-extended feature ID'
    # any features that have feature_type = 'movement' use the standard
    # 6-figure movement type plot.

    # PAGE 0: First an introductory page
    # Table of contents, description of experiment and control, 
    # Heatmap of phenotype
    # a heatmap of available features for each of the worms
    # 
    pages = [None]*94
    pages[0] = 'Table of Contents.'

    # maps (row,col) to sub-extended feature ID
    pages[1] = {(0,0): '', (0,1):  7, (0,2): 12, (0,3): 14, (0,4): 16,
               (1,0):  5, (1,1):  6, (1,2): 17, (1,3): 19, (1,4): 21,
               (2,0): 26, (2,1):  8, (2,2): 50, (2,3): 51, (2,4): 52,
               (3,0): 22, (3,1): 24, (3,2): 53, (3,3): 54, (3,4): 55}
    pages[1][(0,0)] = 'legend'

    pages[2] = {(0,0): 48, (0,1): 49, (0,2): 39, (0,3): 40, (0,4): 41,
               (1,0): 32, (1,1): 33, (1,2): 44, (1,3): 45, (1,4): 46,
               (2,0): 34, (2,1): 35, (2,2): 63, (2,3): 64, (2,4): 65,
               (3,0): 36, (3,1): 37, (3,2): 68, (3,3): 69, (3,4): 70}

    pages[3] = {(0,0): 56, (0,1): 28, (0,2): 58, (0,3): 59, (0,4): 60,
               (1,0):  2, (1,1): 73, (1,2): 74, (1,3): 75, (1,4): 76,
               (2,0):  3, (2,1): 80, (2,2): 81, (2,3): 82, (2,4): 83,
               (3,0):  4, (3,1): 87, (3,2): 88, (3,3): 89, (3,4): 90}

    # THE SIX-FIGURE SECTIOIN    
    
    for i in range(4,27):
        # Movement features
        pages[i] = i + 1

    pages[27] = 'bend count'   #(features[58:63])
    pages[28] = 'coil time'  #(features[58:63])

    for i in range(29,38):
        # More movement features
        pages[i] = i

    pages[38] = 'locomotion.motion_events.forward'
    pages[39] = 'locomotion.motion_events.paused'
    pages[40] = 'locomotion.motion_events.backward'

    for i in range(41,52):
        # More movement features
        pages[i] = i - 3

    pages[52] = 50 # Crawling amplitude
    pages[53] = 51
    pages[54] = 52

    pages[55] = 49 # Foraging speed

    pages[56] = 53 # Crawling frequency
    pages[57] = 54
    pages[58] = 55

    pages[59] = 'Omega turns (just four plots)'
    pages[60] = 'Upsilon turns (just four plots)'
    pages[61] = 56   # ANOTHER 6-figure plot (Path range)
    pages[62] = {(0,0):1,(0,1):2,(1,0):3,(1,1):4}  # Worm dwelling four-grid; worm, head, midbody, tail.
    pages[63] = 67   # ANOTHER 6-figure plot (Path curvature)

    # PATH PLOTS, all annotated with omegas and coils:
    # i.e. exactly 30 pages of path and dwelling charts

    # 10 pages for Midbody/Head/Tail colors
    # pages[64] = 'two charts, 24 experiment, 24 control, blended'
    # pages[65] = ('two charts, 24 experiment, 24 control, split out into '
    #            '24 little plots, 5 columns, 6 rows')
    # pages[66] = '6 plots of experiment worms 0-6'
    # pages[67] = '6 plots of experiment worms 6-12'
    # pages[68] = '6 plots of experiment worms 12-18'
    # pages[69] = '6 plots of experiment worms 18-24'
    # pages[70] = '6 plots of control worms 0-6'
    # pages[71] = '6 plots of control worms 6-12'
    # pages[72] = '6 plots of control worms 12-18'
    # pages[73] = '6 plots of control worms 18-24'

    # pages[74:84] = 'SAME 10 pages AGAIN, BUT NOW FOR MIDBODY SPEED'
    # pages[84:94] = 'SAME 10 pages AGAIN, but now for FORAGING AMPLITUDE'

    # METHODS TABLE OF CONTENTS
    
    
    # """
    
    # for feature in master_feature_list:
    #     title = feature['title']
    #     #three legends:
    #         #experiment / control
    #         #backward / forward / paused
    #         #q-values
    #     if feature['is_time_series']:
    #         # Show 2 rows and 3 columns of plots, with:
    #         rows = 2
    #         cols = 3

    #         (0,0) is motion_type = 'all'
    #             experiment is brown
    #         (1,0) is motion_type = 'forward'
    #             experiment is purple
    #         (1,1) is motion_type = 'paused'
    #             experiment is green
    #         (1,2) is motion_type = 'backward'
    #             experiment is blue
    #         ax = plt.subplot2grid((rows, cols), (i // cols, i % cols))
    
    # # Then link to feature descriptions
    # # perhaps this could be stored in the FeatureInfo table
    
    # """
    return pages

def add_plots(pages, statistics_manager, plot_pdf=None):
    '''
    >>> pages = [None]*94
    >>> add_plots(pages)
    >>> pages = ['Test', 'Test', 'Test']
    >>> add_plots(pages)
    'Test'
    'Test'
    'Test'
    >>> pages = [{(0,0): '', (0,1):  7, (0,2): 12, (0,3): 14, (0,4): 16,
               (1,0):  5, (1,1):  6, (1,2): 17, (1,3): 19, (0,4): 21,
               (2,0): 26, (2,1):  8, (2,2): 50, (2,3): 51, (0,4): 52,
               (3,0): 22, (3,1): 24, (3,2): 53, (3,3): 54, (0,4): 55}]
    >>> add_plots(pages) # doctest: +ELLIPSIS
    ...
    '''
    figure_size = (17, 11)

    page_count = 0
    for page in pages:
        #check if page has any content
        if page:
            #print page if it is a string
            if type(page) is type('string'):
                fig = plt.figure(figsize=figure_size)
                fig.suptitle(page)

            #otherwise it should be a dictionary of for subplot position:feature which indicates plotting order
            elif type(page) is type({}):
                subplot_tuples = sorted(page.keys())
                subplot_rows = subplot_tuples[-1][0] + 1
                subplot_cols = subplot_tuples[-1][1] + 1

                fig = plt.figure(figsize=figure_size)
                for subplot_tuple in reversed(subplot_tuples):
                    feature = page[subplot_tuple]
                    subplot_num = (subplot_tuple[0])*subplot_cols + subplot_tuple[1] + 1 
                    ax = fig.add_subplot(subplot_rows, subplot_cols, subplot_num)
                    subplot_fontsize = int(round(10 - 0.2*(subplot_rows*subplot_cols)))
                    if type(feature) is type(1):
                        try:
                            statistics_manager[feature].plot(ax)
                            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                                item.set_fontsize(subplot_fontsize)
                        except AttributeError:
                            ax.text(0.5,0.5, 'Not Available')
                    elif type(feature) is type('string'):
                        ax.set_title(feature)
                fig.subplots_adjust(wspace=0.4, hspace=0.4)
            elif type(page) is type(1):
                feature = page
                fig = plt.figure(figsize=figure_size)
                ax = fig.gca()
                if type(feature) is type(1):
                    try: 
                        statistics_manager[feature].plot(ax)
                    except AttributeError:
                        ax.text(0.5,0.5, 'Not Available')
                elif type(feature) is type('string'):
                    ax.set_title(feature)
            
            if plot_pdf:
                plt.savefig(plot_pdf, format='pdf')
                plt.close()
            else:
                plt.show()
        page_count += 1
    
    if plot_pdf:
        plot_pdf.close()

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

def main(output_filename=None):
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

    plt.ioff()
    pages = prepare_pages()

    if output_filename:
        plot_pdf = PdfPages(output_filename)
        add_plots(pages, statistics_manager, plot_pdf)
    else:
        add_plots(pages, statistics_manager)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('test_pdf.pdf')
