# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:47:16 2015

@author: mcurrie
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pickle

sys.path.append('..')
import open_worm_analysis_toolbox as mv

# Use pandas to load the features specification
feature_spec_path = os.path.join('..', 'documentation', 'database schema',
                                 'Features Specifications.xlsx')

feature_spec = pd.ExcelFile(feature_spec_path).parse('FeatureSpecifications')

DEBUG = False
FEATURESPECS = {}


class PlotPage(object):

    def __init__(
            self,
            content,
            page_title='',
            grid_shape=None,
            page_legend=None,
            footer=None):
        self.content = content
        self.title = page_title

        if grid_shape:
            self.type = 'GridPlot'
            self.grid_shape = grid_shape
            self.figure, self.axes = plt.subplots(
                self.grid_shape[0], self.grid_shape[1])
            self.content = content
            self.title = page_title
            self.page_legend = page_legend
        else:
            self.type = 'Text'
            self.figure = plt.figure()

        if footer:
            self.footer = footer
            self.figure.text(1, 0, self.footer)

        if page_legend:
            self.figure.subplots_adjust(rspace=.3)

    def add_plots(self, statistics_manager, page_size=(17, 11), plot_pdf=None):
        self.figure.set_size_inches(page_size)
        subplot_tuples = reversed(sorted(self.content.keys()))
        subplot_cols = self.grid_shape[1]
        subplot_rows = self.grid_shape[0]

        for subplot_tuple in subplot_tuples:
            feature = self.content[subplot_tuple]
            subplot_num = (subplot_tuple[0]) * \
                subplot_cols + subplot_tuple[1] + 1

            subplot_fontsize = int(
                round(10 - 0.2 * (subplot_rows * subplot_cols)))
            ax = self.axes[subplot_tuple]

            if isinstance(feature, type('string')):
                ax.set_title(feature)
            elif isinstance(feature, type((0, 0))):
                try:
                    for subfeature in feature:
                        statistics_manager[subfeature].plot(ax)
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(subplot_fontsize)
                except AttributeError:
                    ax.set_title(str(feature) + ' Not Available')
            elif isinstance(feature, type(1)):
                try:
                    statistics_manager[feature].plot(ax)
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(subplot_fontsize)
                    if DEBUG:
                        ax.set_title(str(feature) + ' ' +
                                     str(feature_spec['feature_field'][feature]))
                except AttributeError:
                    ax.set_title(str(feature) + '\nNot Available')
                    if DEBUG:
                        ax.set_title(str(feature) +
                                     ' ' +
                                     str(feature_spec['feature_field'][feature]) +
                                     '\nNot Available')

        self.figure.subplots_adjust(wspace=0.4, hspace=0.4)

        if plot_pdf:
            plot_pdf.savefig(self.figure)
            plt.close(self.figure)
        else:
            plt.show()

    def add_text(self, page_size=(17, 11), plot_pdf=None):
        self.figure.set_size_inches(page_size)

        self.figure.suptitle(self.title + '\n' + self.content)

        if plot_pdf:
            plot_pdf.savefig(self.figure)
            plt.close(self.figure)
        else:
            plt.show()


class PlotDocument(object):

    def __init__(
            self,
            pages,
            statistics_manager,
            page_size=(
                17,
                11),
            pdf_filename=None):
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
                    page.add_plots(
                        self.statistics_manager,
                        self.page_size,
                        self.plot_pdf)
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

        PlotDocument.__init__(
            self,
            pages=self.shaffer_pages(),
            statistics_manager=statistics_manager,
            pdf_filename=pdf_filename)

        # print 'Feature,Name,MotionType,DataType'
        # for i in range(len(exp_histogram_manager)):
        #     try:
        #         feature = statistics_manager[i]
        #         FEATURESPECS[i] = {'name':feature.specs.name, 'motion_type':feature.motion_type, 'data_type':feature.data_type}
        #         print str(i) + ',' + FEATURESPECS[i]['name'] + ',' + FEATURESPECS[i]['motion_type'] + ',' + FEATURESPECS[i]['data_type']
        #     except AttributeError:
        #         print str(i) + ',NA,NA,NA'
        #         FEATURESPECS[i] = ',NA,NA,NA'

    def shaffer_pages(self):
        # Assigned numbers are the 'sub-extended feature ID'
        # any features that have feature_type = 'movement' use the standard
        # 6-figure movement type plot.

        # PAGE 0: First an introductory page
        # Table of contents, description of experiment and control,
        # Heatmap of phenotype
        # a heatmap of available features for each of the worms
        #
        pages = [None] * 94
        pages[0] = PlotPage(content='Table of Contents.')

        # maps (row,col) to sub-extended feature ID
        pages[1] = PlotPage(content={(0, 0): 'legend', (0, 1): 8, (0, 2): 28, (0, 3): 60, (0, 4): 92,
                                     (1, 0): 0, (1, 1): 4, (1, 2): 108, (1, 3): 140, (1, 4): 172,
                                     (2, 0): 204, (2, 1): 12, (2, 2): 552, (2, 3): 568, (2, 4): 584,
                                     (3, 0): 188, (3, 1): 196, (3, 2): 600, (3, 3): 616, (3, 4): 632}, grid_shape=(4, 5), footer='1 Summary')

        pages[2] = PlotPage(content={(0,
                                      0): 520,
                                     (0,
                                      1): 536,
                                     (0,
                                      2): 376,
                                     (0,
                                      3): 392,
                                     (0,
                                      4): 408,
                                     (1,
                                      0): 264,
                                     (1,
                                      1): 280,
                                     (1,
                                      2): 456,
                                     (1,
                                      3): 472,
                                     (1,
                                      4): 488,
                                     (2,
                                      0): 296,
                                     (2,
                                      1): 312,
                                     (2,
                                      2): 'Omega Turn Time',
                                     (2,
                                      3): 'Inter Omega Time',
                                     (2,
                                      4): 'Inter Omega Distance',
                                     (3,
                                      0): 328,
                                     (3,
                                      1): 344,
                                     (3,
                                      2): 'Upsilon Turn Time',
                                     (3,
                                      3): 'Inter Upsilon Time',
                                     (3,
                                      4): 'Inter Upsilon Distance'},
                            grid_shape=(4,
                                        5),
                            footer='2 Summary')

        pages[3] = PlotPage(content={(0,
                                      0): 648,
                                     (0,
                                      1): 212,
                                     (0,
                                      2): 672,
                                     (0,
                                      3): 'Inter Coil Time',
                                     (0,
                                      4): 'Inter Coil Distance',
                                     (1,
                                      0): 669,
                                     (1,
                                      1): 'Forward Time',
                                     (1,
                                      2): 'Forward Distance',
                                     (1,
                                      3): 'Inter Forward Time',
                                     (1,
                                      4): 'Inter Forward Distance',
                                     (2,
                                      0): 670,
                                     (2,
                                      1): 712,
                                     (2,
                                      2): 714,
                                     (2,
                                      3): 714,
                                     (2,
                                      4): 715,
                                     (3,
                                      0): 671,
                                     (3,
                                      1): 'Backward Time',
                                     (3,
                                      2): 'Backward Distance',
                                     (3,
                                      3): 'Inter Backward Time',
                                     (3,
                                      4): 'Inter Backward Distance'},
                            grid_shape=(4,
                                        5),
                            footer='3 Summary')

        # THE SIX-FIGURE SECTIION

        # Morphology
        pages[4] = PlotPage(content={(0, 0): 0, (0, 1): 'N2: Length', (1, 0): 1, (
            1, 1): 2, (1, 2): 3}, grid_shape=(2, 3), footer='4 Morphology: Length')
        pages[5] = PlotPage(content={(0, 0): 4, (0, 1): 'N2: Head Width', (1, 0): 5, (
            1, 1): 6, (1, 2): 7}, grid_shape=(2, 3), footer='5 Morphology: Head Width')
        pages[6] = PlotPage(content={(0, 0): 8, (0, 1): 'N2: Midbody Width', (1, 0): 9, (
            1, 1): 10, (1, 2): 11}, grid_shape=(2, 3), footer='6 Morphology: Midbody Width')
        pages[7] = PlotPage(content={(0, 0): 12, (0, 1): 'N2: Tail Width', (1, 0): 13, (
            1, 1): 14, (1, 2): 15}, grid_shape=(2, 3), footer='7 Morphology: Tail Width')
        pages[8] = PlotPage(content={(0, 0): 16, (0, 1): 'N2: Area', (1, 0): 17, (
            1, 1): 18, (1, 2): 19}, grid_shape=(2, 3), footer='8 Morphology: Area')
        pages[9] = PlotPage(content={(0,
                                      0): 20,
                                     (0,
                                      1): 'N2: Area/Length',
                                     (1,
                                      0): 21,
                                     (1,
                                      1): 22,
                                     (1,
                                      2): 23},
                            grid_shape=(2,
                                        3),
                            footer='10 Morphology: Area/Length')
        pages[10] = PlotPage(content={(0,
                                       0): 24,
                                      (0,
                                       1): 'N2: Width/Length',
                                      (1,
                                       0): 25,
                                      (1,
                                       1): 26,
                                      (1,
                                       2): 27},
                             grid_shape=(2,
                                         3),
                             footer='11 Morphology: Width/Length')

        # Posture
        pages[11] = PlotPage(content={(0, 0): 28, (0, 1): 'N2: Head Bend Mean', (1, 0): 32, (
            1, 1): 36, (1, 2): 40}, grid_shape=(2, 3), footer='11 Posture: Head Bend Mean')
        pages[12] = PlotPage(content={(0, 0): 44, (0, 1): 'N2: Neck Bend Mean', (1, 0): 48, (
            1, 1): 52, (1, 2): 56}, grid_shape=(2, 3), footer='12 Posture: Neck Bend Mean')
        pages[13] = PlotPage(content={(0, 0): 60, (0, 1): 'N2: Midbody Bend Mean', (1, 0): 64, (
            1, 1): 68, (1, 2): 72}, grid_shape=(2, 3), footer='13 Posture: Midbody Bend Mean')
        pages[14] = PlotPage(content={(0, 0): 76, (0, 1): 'N2: Hips Bend Mean', (1, 0): 80, (
            1, 1): 84, (1, 2): 88}, grid_shape=(2, 3), footer='14 Posture: Hips Bend Mean')
        pages[15] = PlotPage(content={(0, 0): 92, (0, 1): 'N2: Tail Bend Mean', (1, 0): 96, (
            1, 1): 100, (1, 2): 104}, grid_shape=(2, 3), footer='15 Posture: Tail Bend Mean')

        pages[16] = PlotPage(content={(0, 0): 108, (0, 1): 'N2: Head Bend S.D.', (1, 0): 112, (
            1, 1): 116, (1, 2): 120}, grid_shape=(2, 3), footer='16 Posture: Head Bend S.D.')
        pages[17] = PlotPage(content={(0, 0): 124, (0, 1): 'N2: Neck Bend S.D.', (1, 0): 128, (
            1, 1): 132, (1, 2): 136}, grid_shape=(2, 3), footer='17 Posture: Neck Bend S.D.')
        pages[18] = PlotPage(content={(0, 0): 140, (0, 1): 'N2: Midbody Bend S.D.', (1, 0): 144, (
            1, 1): 148, (1, 2): 152}, grid_shape=(2, 3), footer='18 Posture: Midbody Bend S.D.')
        pages[19] = PlotPage(content={(0, 0): 156, (0, 1): 'N2: Hips Bend S.D', (1, 0): 160, (
            1, 1): 164, (1, 2): 168}, grid_shape=(2, 3), footer='19 Posture: Hips Bend S.D.')
        pages[20] = PlotPage(content={(0, 0): 172, (0, 1): 'N2: Tail Bend S.D', (1, 0): 176, (
            1, 1): 180, (1, 2): 184}, grid_shape=(2, 3), footer='20 Posture: Tail Bend S.D.')

        pages[21] = PlotPage(content={(0, 0): 188, (0, 1): 'N2: Max Amplitude', (1, 0): 189, (
            1, 1): 190, (1, 2): 191}, grid_shape=(2, 3), footer='21 Posture: Max Amplitude')
        pages[22] = PlotPage(content={(0, 0): 192, (0, 1): 'N2: Amplitude Ratio', (1, 0): 193, (
            1, 1): 194, (1, 2): 195}, grid_shape=(2, 3), footer='22 Posture: Amplitude Ratio')
        pages[23] = PlotPage(content={(0, 0): 196, (0, 1): 'N2: Primary Wavelength', (1, 0): 197, (
            1, 1): 198, (1, 2): 199}, grid_shape=(2, 3), footer='23 Posture: Primary Wavelength')
        pages[24] = PlotPage(content={(0,
                                       0): 'Secondary Wavelength',
                                      (0,
                                       1): 'N2: Secondary Wavelength',
                                      (1,
                                       0): 'Forward',
                                      (1,
                                       1): 'Paused',
                                      (1,
                                       2): 'Backward'},
                             grid_shape=(2,
                                         3),
                             footer='24 Posture: Secondary Wavelength')

        pages[25] = PlotPage(content={(0, 0): 204, (0, 1): 'N2: Track Length', (1, 0): 205, (
            1, 1): 206, (1, 2): 207}, grid_shape=(2, 3), footer='25 Posture: Track Length')
        pages[26] = PlotPage(content={(0, 0): 208, (0, 1): 'N2: Eccentricity', (1, 0): 209, (
            1, 1): 210, (1, 2): 211}, grid_shape=(2, 3), footer='26 Posture: Eccentricity')
        pages[27] = PlotPage(content={(0, 0): 212, (0, 1): 'N2: Bend Count', (1, 0): 213, (
            1, 1): 214, (1, 2): 215}, grid_shape=(2, 3), footer='27 Posture: Bend Count')
        pages[28] = PlotPage(content={(0,
                                       0): 672,
                                      (0,
                                       1): 'Inter Coil Time',
                                      (0,
                                       2): 'Inter vs Coil Time',
                                      (1,
                                       0): 'Inter Coil Distance'},
                             grid_shape=(2,
                                         3),
                             footer='28 Posture: Coiling Events')

        pages[29] = PlotPage(content={(0, 0): 216, (0, 1): 'N2: Tail to Head Orientation', (1, 0): 220, (
            1, 1): 224, (1, 2): 228}, grid_shape=(2, 3), footer='29 Posture: Tail to Head Orientation')
        pages[30] = PlotPage(content={(0, 0): 232, (0, 1): 'N2: Head Orientation', (1, 0): 236, (
            1, 1): 240, (1, 2): 244}, grid_shape=(2, 3), footer='30 Posture: Head Orientation')
        pages[31] = PlotPage(content={(0, 0): 248, (0, 1): 'N2: Tail Orientation', (1, 0): 252, (
            1, 1): 256, (1, 2): 260}, grid_shape=(2, 3), footer='31 Posture: Tail Orientation')

        pages[32] = PlotPage(content={(0, 0): 264, (0, 1): 'N2: Eigen Projection 1', (1, 0): 268, (
            1, 1): 272, (1, 2): 276}, grid_shape=(2, 3), footer='32 Posture: Eigen Projection 1')
        pages[33] = PlotPage(content={(0, 0): 280, (0, 1): 'N2: Eigen Projection 2', (1, 0): 284, (
            1, 1): 288, (1, 2): 292}, grid_shape=(2, 3), footer='33 Posture: Eigen Projection 2')
        pages[34] = PlotPage(content={(0, 0): 296, (0, 1): 'N2: Eigen Projection 3', (1, 0): 300, (
            1, 1): 304, (1, 2): 308}, grid_shape=(2, 3), footer='34 Posture: Eigen Projection 3')
        pages[35] = PlotPage(content={(0, 0): 312, (0, 1): 'N2: Eigen Projection 4', (1, 0): 316, (
            1, 1): 320, (1, 2): 324}, grid_shape=(2, 3), footer='35 Posture: Eigen Projection 4')
        pages[36] = PlotPage(content={(0, 0): 328, (0, 1): 'N2: Eigen Projection 5', (1, 0): 332, (
            1, 1): 336, (1, 2): 340}, grid_shape=(2, 3), footer='36 Posture: Eigen Projection 5')
        pages[37] = PlotPage(content={(0, 0): 344, (0, 1): 'N2: Eigen Projection 6', (1, 0): 348, (
            1, 1): 352, (1, 2): 356}, grid_shape=(2, 3), footer='37 Posture: Eigen Projection 6')

        # Motion
        pages[38] = PlotPage(content={(0,
                                       0): 'Forward Time',
                                      (0,
                                       1): 'Inter Forward Time',
                                      (0,
                                       2): 'Inter vs. Forward Time',
                                      (1,
                                       0): 'Forward Distance',
                                      (1,
                                       1): 'Inter Forward Distance',
                                      (1,
                                       2): 'Inter vs. Forward Distance'},
                             grid_shape=(2,
                                         3),
                             footer='38 Motion: Forward Motion')
        pages[39] = PlotPage(content={(0,
                                       0): 'Paused Time',
                                      (0,
                                       1): 714,
                                      (0,
                                       2): 'Inter vs. Paused Time',
                                      (1,
                                       0): 'Paused Distance',
                                      (1,
                                       1): 715,
                                      (1,
                                       2): 'Inter vs. Paused Distance'},
                             grid_shape=(2,
                                         3),
                             footer='39 Motion: Paused Motion')
        pages[40] = PlotPage(content={(0,
                                       0): 'Backward Time',
                                      (0,
                                       1): 'Inter Backward Time',
                                      (0,
                                       2): 'Inter vs. Backward Time',
                                      (1,
                                       0): 'Backward Distance',
                                      (1,
                                       1): 'Inter Backward Distance',
                                      (1,
                                       2): 'Inter vs. Backward Distance'},
                             grid_shape=(2,
                                         3),
                             footer='40 Motion: Backward Motion')

        pages[41] = PlotPage(content={(0, 0): 360, (0, 1): 'N2: Head Tip Speed', (1, 0): 364, (
            1, 1): 368, (1, 2): 372}, grid_shape=(2, 3), footer='41 Motion: Head Tip Speed')
        pages[42] = PlotPage(content={(0, 0): 376, (0, 1): 'N2: Head Speed', (1, 0): 380, (
            1, 1): 384, (1, 2): 388}, grid_shape=(2, 3), footer='42 Motion: Head Speed')
        pages[43] = PlotPage(content={(0, 0): 392, (0, 1): 'N2: Midbody Speed', (1, 0): 396, (
            1, 1): 400, (1, 2): 404}, grid_shape=(2, 3), footer='43 Motion: Midbody Speed')
        pages[44] = PlotPage(content={(0, 0): 408, (0, 1): 'N2: Tail Speed', (1, 0): 412, (
            1, 1): 416, (1, 2): 420}, grid_shape=(2, 3), footer='44 Motion: Tail Speed')
        pages[45] = PlotPage(content={(0, 0): 424, (0, 1): 'N2: Tail Tip Speed', (1, 0): 428, (
            1, 1): 432, (1, 2): 436}, grid_shape=(2, 3), footer='45 Motion: Tail Tip Speed')

        pages[46] = PlotPage(content={(0, 0): 440, (0, 1): 'N2: Head Tip Motion Direction', (1, 0): 444, (
            1, 1): 448, (1, 2): 452}, grid_shape=(2, 3), footer='46 Motion: Head Tip Motion Direction')
        pages[47] = PlotPage(content={(0,
                                       0): 456,
                                      (0,
                                       1): 'N2: Head Motion Direction',
                                      (1,
                                       0): 460,
                                      (1,
                                       1): 464,
                                      (1,
                                       2): 468},
                             grid_shape=(2,
                                         3),
                             footer='47 Motion: Head Motion Direction')
        pages[48] = PlotPage(content={(0, 0): 472, (0, 1): 'N2: Midbody Motion Direction', (1, 0): 476, (
            1, 1): 480, (1, 2): 484}, grid_shape=(2, 3), footer='48 Motion: Midbody Motion Direction')
        pages[49] = PlotPage(content={(0,
                                       0): 488,
                                      (0,
                                       1): 'N2: Tail Motion Direction',
                                      (1,
                                       0): 492,
                                      (1,
                                       1): 496,
                                      (1,
                                       2): 500},
                             grid_shape=(2,
                                         3),
                             footer='49 Motion: Tail Motion Direction')
        pages[50] = PlotPage(content={(0, 0): 504, (0, 1): 'N2: Tail Tip Motion Direction', (1, 0): 508, (
            1, 1): 512, (1, 2): 516}, grid_shape=(2, 3), footer='50 Motion: Tail Tip Motion Direction')

        pages[51] = PlotPage(content={(0, 0): 520, (0, 1): 'N2: Foraging Amplitude', (1, 0): 524, (
            1, 1): 528, (1, 2): 532}, grid_shape=(2, 3), footer='51 Motion: Foraging Amplitude')
        pages[52] = PlotPage(content={(0,
                                       0): 552,
                                      (0,
                                       1): 'N2: Head Crawling Amplitude',
                                      (1,
                                       0): 'Forward',
                                      (1,
                                       1): 'Paused',
                                      (1,
                                       2): 564},
                             grid_shape=(2,
                                         3),
                             footer='52 Motion: Head Crawling Amplitude')
        pages[53] = PlotPage(
            content={
                (
                    0,
                    0): 568,
                (0,
                 1): 'N2: Midbody Crawling Amplitude',
                (1,
                 0): 572,
                (1,
                 1): 'Paused',
                (1,
                 2): 'Backward'},
            grid_shape=(
                2,
                3),
            footer='53 Motion: Midbody Crawling Amplitude')
        pages[54] = PlotPage(content={(0,
                                       0): 584,
                                      (0,
                                       1): 'N2: Tail Crawling Amplitude',
                                      (1,
                                       0): 'Forward',
                                      (1,
                                       1): 'Paused',
                                      (1,
                                       2): 'Backward'},
                             grid_shape=(2,
                                         3),
                             footer='54 Motion: Tail Crawling Amplitude')

        pages[55] = PlotPage(content={(0, 0): 536, (0, 1): 'N2: Foraging Speed', (1, 0): 540, (
            1, 1): 544, (1, 2): 548}, grid_shape=(2, 3), footer='55 Motion: Foraging Speed')
        pages[56] = PlotPage(content={(0,
                                       0): 600,
                                      (0,
                                       1): 'N2: Head Crawling Frequency',
                                      (1,
                                       0): 'Forward',
                                      (1,
                                       1): 'Paused',
                                      (1,
                                       2): 612},
                             grid_shape=(2,
                                         3),
                             footer='56 Motion: Head Crawling Frequency')
        pages[57] = PlotPage(
            content={
                (
                    0,
                    0): 616,
                (0,
                 1): 'N2: Midbody Crawling Frequency',
                (1,
                 0): 620,
                (1,
                 1): 'Paused',
                (1,
                 2): 'Backward'},
            grid_shape=(
                2,
                3),
            footer='57 Motion: Midbody Crawling Frequency')
        pages[58] = PlotPage(content={(0,
                                       0): 632,
                                      (0,
                                       1): 'N2: Tail Crawling Frequency',
                                      (1,
                                       0): 'Forward',
                                      (1,
                                       1): 'Paused',
                                      (1,
                                       2): 'Backward'},
                             grid_shape=(2,
                                         3),
                             footer='58 Motion: Tail Crawling Frequency')

        pages[59] = PlotPage(content={(0,
                                       0): 'Omega Turn Time',
                                      (0,
                                       1): 'Inter Omega Turn Time',
                                      (0,
                                       2): 'Inter vs. Omega Turn Time',
                                      (1,
                                       0): 'Omega Turn Distance',
                                      (1,
                                       1): 'Inter Omega Turn Distance',
                                      (1,
                                       2): 'Inter vs. Omega Turn Distance'},
                             grid_shape=(2,
                                         3),
                             footer='59 Motion: Omega Turn Motion')
        pages[60] = PlotPage(content={(0,
                                       0): 'Upsilon Turn Time',
                                      (0,
                                       1): 'Inter Upsilon Turn Time',
                                      (0,
                                       2): 'Inter vs. Upsilon Turn Time',
                                      (1,
                                       0): 'Upsilon Turn Distance',
                                      (1,
                                       1): 'Inter Upsilon Turn Distance',
                                      (1,
                                       2): 'Inter vs. Upsilon Turn Distance'},
                             grid_shape=(2,
                                         3),
                             footer='60 Motion: Upsilon Turn Motion')

        # Path
        pages[61] = PlotPage(content={(0, 0): 648, (0, 1): 'N2: Range', (1, 0): 649, (
            1, 1): 650, (1, 2): 651}, grid_shape=(2, 3), footer='26 Path: Range')
        pages[62] = PlotPage(
            content={
                (
                    0, 0): 668, (0, 1): 669, (1, 0): 670, (1, 1): 671}, grid_shape=(
                2, 2), footer='26 Path: Dwelling')
        pages[63] = PlotPage(content={(0, 0): 652, (0, 1): 'N2: Curvature', (1, 0): 656, (
            1, 1): 660, (1, 2): 664}, grid_shape=(2, 3), footer='26 Path: Curvature')

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

            # Compute pathplots on our files
            exp_pathplot_manager = mv.PathPlotManager(experiment_files[:10])
            ctl_pathplot_manager = mv.PathPlotManager(control_files[:10])

            # Store a pickle file in the same folder as this script
            # (i.e. open-worm-analysis-toolbox/examples/)
            with open(pickle_file_path, "wb") as pickle_file:
                pickle.dump(exp_histogram_manager, pickle_file)
                pickle.dump(ctl_histogram_manager, pickle_file)

        print("Experiment has a total of " +
              str(len(exp_histogram_manager.merged_histograms)) +
              " histograms")

        return exp_histogram_manager, ctl_histogram_manager, exp_pathplot_manager, ctl_pathplot_manager

if __name__ == '__main__':
    plt.ioff()

    if len(sys.argv) > 1:
        document = ShafferPlotDocument(sys.argv[1])
        document.make_pages()
    else:
        document = ShafferPlotDocument('test_pdf.pdf')
        document.make_pages()
