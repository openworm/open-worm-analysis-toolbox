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

DEBUG=False

class PlotPage(object):
    def __init__(self, content, page_title='', grid_shape=None, page_legend=None):
        self.content = content
        self.title = page_title

        if grid_shape:
            self.type = 'GridPlot'
            self.grid_shape = grid_shape
            self.figure, self.axes = plt.subplots(self.grid_shape[0],self.grid_shape[1])
            self.content = content
            self.title = page_title
            self.page_legend = page_legend
        else:
            self.type = 'Text'
            self.figure = plt.figure()

    def add_plots(self, statistics_manager, page_size=(17,11), plot_pdf=None):
        self.figure.set_size_inches(page_size)
        subplot_tuples = reversed(sorted(self.content.keys()))
        subplot_cols = self.grid_shape[1]
        subplot_rows = self.grid_shape[0]

        for subplot_tuple in subplot_tuples:
            feature = self.content[subplot_tuple]
            subplot_num = (subplot_tuple[0])*subplot_cols + subplot_tuple[1] + 1 

            subplot_fontsize = int(round(10 - 0.2*(subplot_rows*subplot_cols)))
            ax = self.axes[subplot_tuple]

            if type(feature) is type('string'):
                ax.set_title(feature)
            elif type(feature) is type((0,0)):
                try:
                    for subfeature in feature:
                        statistics_manager[subfeature].plot(ax)
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(subplot_fontsize)
                except AttributeError:
                    ax.set_title(str(feature) + ' Not Available')
            elif type(feature) is type(1):
                try:
                    statistics_manager[feature].plot(ax)
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(subplot_fontsize)
                    if DEBUG: 
                        ax.set_title(str(feature) + ' ' + str(feature_spec['feature_field'][feature]))
                except AttributeError:
                    ax.set_title(str(feature) + '\nNot Available')
                    if DEBUG:
                        ax.set_title(str(feature) + ' ' + str(feature_spec['feature_field'][feature])  + '\nNot Available')

        self.figure.subplots_adjust(wspace=0.4, hspace=0.4)
         
        if plot_pdf:
            plot_pdf.savefig(self.figure)
            #plt.close()
        else:
            plt.show()

    def add_text(self, page_size=(17,11), plot_pdf=None):
        self.figure.set_size_inches(page_size)

        self.figure.suptitle(self.title + '\n' + self.content)

        if plot_pdf:
            plot_pdf.savefig(self.figure)
            #plt.close()
        else:
            plt.show()

class PlotDocument(object):
    def __init__(self, pages, statistics_manager, page_size=(17,11), pdf_filename=None):
        if pdf_filename:
            self.plot_pdf = PdfPages(pdf_filename)
        else:
            self.plot_pdf = None

        self.pages = pages
        self.page_size = page_size
        self.statistics_manager = statistics_manager

    def make_pages(self):
        for page in self.pages:
            if page:
                if page.type == 'GridPlot':
                    page.add_plots(self.statistics_manager, self.page_size, self.plot_pdf)
                else:
                    page.add_text(self.page_size, self.plot_pdf)
        if self.plot_pdf:
            plt.close()
            self.plot_pdf.close()

class ShafferPlotDocument(PlotDocument):
    def __init__(self, pdf_filename=None):
        base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
        root_path = os.path.join(base_path, '30m_wait')

        exp_histogram_manager, ctl_histogram_manager = \
            self.obtain_histograms(root_path, "pickled_histograms.dat")

        print("Done with Histogram generation.  Now let's calculate statistics.")

        statistics_manager = \
            mv.StatisticsManager(exp_histogram_manager, ctl_histogram_manager)

        print("Comparison p and q values are %.2f and %.2f, respectively." %
              (statistics_manager.min_p_wilcoxon, 
               statistics_manager.min_q_wilcoxon))

        PlotDocument.__init__(self, pages=self.shaffer_pages(), statistics_manager=statistics_manager, pdf_filename=pdf_filename)

    def shaffer_pages(self):
        # Assigned numbers are the 'sub-extended feature ID'
        # any features that have feature_type = 'movement' use the standard
        # 6-figure movement type plot.

        # PAGE 0: First an introductory page
        # Table of contents, description of experiment and control, 
        # Heatmap of phenotype
        # a heatmap of available features for each of the worms
        # 
        pages = [None]*94
        pages[0] = PlotPage(content='Table of Contents.')

        # maps (row,col) to sub-extended feature ID
        pages[1] = PlotPage(content={(0,0): 'legend', (0,1):  7, (0,2): 12, (0,3): 14, (0,4): 16,
                   (1,0):  5, (1,1):  6, (1,2): 17, (1,3): 19, (1,4): 21,
                   (2,0): 26, (2,1):  8, (2,2): 50, (2,3): 51, (2,4): 52,
                   (3,0): 22, (3,1): 24, (3,2): 53, (3,3): 54, (3,4): 55}, grid_shape=(4,5))

        pages[2] = PlotPage(content={(0,0): 48, (0,1): 49, (0,2): 39, (0,3): 40, (0,4): 41,
                   (1,0): 32, (1,1): 33, (1,2): 44, (1,3): 45, (1,4): 46,
                   (2,0): 34, (2,1): 35, (2,2): 63, (2,3): 64, (2,4): 65,
                   (3,0): 36, (3,1): 37, (3,2): 68, (3,3): 69, (3,4): 70}, grid_shape=(4,5))

        pages[3] = PlotPage(content={(0,0): 56, (0,1): 28, (0,2): 58, (0,3): 59, (0,4): 60,
                   (1,0):  2, (1,1): 73, (1,2): 74, (1,3): 75, (1,4): 76,
                   (2,0):  3, (2,1): 80, (2,2): 81, (2,3): 82, (2,4): 83,
                   (3,0):  4, (3,1): 87, (3,2): 88, (3,3): 89, (3,4): 90}, grid_shape=(4,5))

        # THE SIX-FIGURE SECTIOIN    
        
        # pages[4] = {(0,0):5,(0,1):5,(1,0):5,(1,1):5,(1,2):5, 'gridshape':(2,3)} # length
        # pages[5] = {(0,0):6,(0,1):6,(1,0):6,(1,1):6,(1,2):6, 'gridshape':(2,3)} # head width
        # pages[6] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[7] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[8] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[9] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[10] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[11] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[12] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[13] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[14] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[15] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[16] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[17] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[18] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[19] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[20] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[21] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[23] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[24] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[25] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width
        # pages[26] = {(0,0):7,(0,1):7,(1,0):7,(1,1):7,(1,2):7, 'gridshape':(2,3)} # midbody width


        # pages[27] = 'bend count'   #(features[58:63])
        # pages[28] = 'coil time'  #(features[58:63])

        # for i in range(29,38):
        #     # More movement features
        #     pages[i] = i

        # pages[38] = 'locomotion.motion_events.forward'
        # pages[39] = 'locomotion.motion_events.paused'
        # pages[40] = 'locomotion.motion_events.backward'

        # for i in range(41,52):
        #     # More movement features
        #     pages[i] = i - 3

        # pages[52] = 50 # Crawling amplitude
        # pages[53] = 51
        # pages[54] = 52

        # pages[55] = 49 # Foraging speed

        # pages[56] = 53 # Crawling frequency
        # pages[57] = 54
        # pages[58] = 55

        # pages[60] = {(0,0):63,(0,1):64,(0,2):(63,64),
        #              (1,0):65, 'gridshape':(2,3)}
        # pages[61] = {(0,0):68,(0,1):69,(0,2):(68,69),
        #              (1,0):70, 'gridshape':(2,2)}
        # pages[62] = 56   # ANOTHER 6-figure plot (Path range)
        # pages[63] = {(0,0):1,(0,1):2,(1,0):3,(1,1):4}  # Worm dwelling four-grid; worm, head, midbody, tail.
        # pages[64] = 67   # ANOTHER 6-figure plot (Path curvature)

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

    def obtain_histograms(self, root_path, pickle_file_path):
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
    plt.ioff()



    if len(sys.argv) > 1:
        document = ShafferPlotDocument(sys.argv[1])
        document.make_pages()
    else:
        document = ShafferPlotDocument('test_pdf.pdf')
        document.make_pages()

