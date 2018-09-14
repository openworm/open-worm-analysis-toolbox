# -*- coding: utf-8 -*-
"""

Entry Point
-----------
mv.HistogramManager(feature_path_or_object_list)

The current processing approach is to take a set of features from an
experiment and to summarize each of these features as a binned data set
(i.e. a histogram) where for each bin of a given width the # of values
occupying each bin is logged.

The histogram manager holds histograms for all computed features. Besides
helping to instantiate these histograms, it also holds any information
that is common to all of the histograms (e.g. experiment sources). By holding
the histograms it serves as a nice entry point for the histograms, rather
than just having a list of histograms.

Formerly SegwormMatlabClasses/+seg_worm/+stats/@hist/manager.m

"""
import h5py
import numpy as np
import six  # For compatibility with Python 2.x
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .. import utils
from ..features.worm_features import WormFeatures

from .histogram import Histogram, MergedHistogram

# This is where I'd like to go with things ...
# Names need some work
#===================================================


#JAH 2018-08 - I think this was going to include code to clean
#up this code but we never got to it ....
class HistogramManagerDos(object):

    def __init__(self, feature_sets):
        pass


class HistogramSet(object):
    """
    Each histogram should
    """
    pass
#===================================================


#%%
class HistogramManager(object):
    """
    Histograms calculated on all features for a collection of feature files
    or WormFeatures objects.

    Attributes
    ----------
    hist_matrix : [n_features x n_videos] open_worm_analysis_toolbox.statistics.histogram.Histogram
        The individual histograms.
    row_names : feature names for each row
    merged_histograms: numpy array of MergedHistogram objects
        This can be accessed via the overloaded [] operator

    Notes
    -------------
    Translated from the seg_worm.stats.hist.manager class

    """
    #%%

    def __init__(self, feature_path_or_object_list, verbose=False):
        """
        Parameters
        ----------
        feature_path :
            
        object_list : TODO: What type????
        feature_path_or_object_list: list of strings or feature objects
            Full paths to all feature files making up this histogram, or
            their in-memory object equivalents.

        Outline:
        -------
        1) Create individual histograms for the features on a per video basis
        2) Merge histograms across all video

        """
        
        if len(feature_path_or_object_list) == 0:
            raise Exception('Empty input to histogram manager not supported')
        
        #TODO: We need better type checking ...
        
        if verbose:
            print("Number of feature files passed into the histogram manager:",
                  len(feature_path_or_object_list))

        # This will have shape (len(feature_path_or_object_list), 726)
        temp_hist_array = []
        
        all_hist_names = [];

        n_videos = len(feature_path_or_object_list)

        # Loop over all feature files and get histogram objects for each
        for feature_path_or_object in feature_path_or_object_list:
            worm_features = None

            if isinstance(feature_path_or_object, six.string_types):
                # If we have a string, it's a filepath to an HDF5 feature file
                file_path = feature_path_or_object
                worm_features = WormFeatures.from_disk(file_path)
            else:
                # Otherwise the worm features have been passed directly
                # as an instance of WormFeatures (we hope)
                worm_features = feature_path_or_object

            # TODO: Need to add on info to properties
            # worm_features.info -> obj.info

            new_histogram_set = self.init_histograms(worm_features)
            
            #Note that names from features are always valid, unlike
            #the histogram
            hist_names = np.array([x.spec.name for x in worm_features])
            
            temp_hist_array.append(new_histogram_set)

            all_hist_names.extend(hist_names)


        unique_names = np.unique(all_hist_names)
        
        n_features = len(unique_names)
        
        #JAH: I rewrote this code to ensure that we had a matrix shaped
        #group of histograms, with None as the default value for missing
        hist_matrix = np.full([n_features,n_videos],None,object)
        
        #TODO: This could be sped up a bit ...
        for i in range(n_videos):
            vid_hists = temp_hist_array[i]
            for hist in vid_hists:
                if hist is not None:
                    hist_name = hist.name
                    for k, name in enumerate(unique_names):
                        if name == hist_name:
                            hist_matrix[k,i] = hist
        
        self.row_names = unique_names
        self.hist_matrix = hist_matrix
                    
        #Merge histogram objects across all videos ...
        self.merged_histograms = \
            HistogramManager.merge_histograms(self.hist_matrix)

    def __getitem__(self, index):
        return self.merged_histograms[index]

    def __len__(self):
        return len(self.merged_histograms)

    @property
    def valid_histograms_mask(self):
        return np.array([h is not None for h in self.merged_histograms])

    @property
    def valid_histograms_array(self):
        return self.merged_histograms[self.valid_histograms_mask]

    @property
    def valid_means_array(self):
        return np.array([hist.mean for hist in self.valid_histograms_array])

    @property
    def num_videos(self):
        return self.hist_matrix.shape[1]
    
    @property
    def num_features(self):
        return self.hist_matrix.shape[0]

    @property
    def valid_2d_mask(self):
        """
        Return a mask showing which elements of hist_cell_array are null.

        """
        valid_mean_detector = lambda h: True if h is not None else False
        return np.vectorize(valid_mean_detector)(self.hist_matrix)

    @property
    def means_2d_dataframe(self):
        """
        Returns
        -----------
        Pandas dataframe
            Shape (10,726)
            
        #??? Where does this come from????

        """
        
        valid_mean_detector = lambda h: h.mean if h is not None else np.NaN
        means_array = np.vectorize(valid_mean_detector)(self.hist_matrix)

        df = pd.DataFrame(data=means_array)
        
        # Give a more human-readable column name
        df.columns = ['Video %d mean' % i for i in range(self.num_videos)]
        
        df2 = pd.DataFrame(data = self.row_names)
        df2.columns = ['Feature name']
        
        return df2.join(df)

        #Old Code
        #-------------------------------
        #feature_spec
        #set_index('sub-extended feature ID',
        #          'motion_type', 'data_type')
        #feature_spec = WormFeatures.get_feature_spec(extended=True)
        #feature_spec = feature_spec[['feature_field',
        #                             'data_type',
        #                             'motion_type']]
        
        #??? feature_spec index is a tuple of (extended_id, regular_id)
        #df.index is now a RangeIndex(start=0, stop=93, step=1)
        
        #cannot join with no level specified and no overlapping names
        #try:
        #    return feature_spec.join(df)
        #except:
        #    import pdb
        #    pdb.set_trace()

    #%%
    def init_histograms(self, worm_features):
        """

        #TODO: Add documentation

        Parameters
        ------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology,
            path, locomotion, in an h5py group.
            (seg_worm.features or strut) This may truly be a feature
            object or the old structure. Both have the same format.
            -@JimHokanson


        """

        return np.array([Histogram.create_histogram(f) for f in worm_features])

    @staticmethod
    def merge_histograms(hist_matrix, verbose=False):
        """
        The goal of this function is to go from n collections of 708
        histogram summaries of features each, to one set of 708 histogram
        summaries, that has n elements, one for each video

        i.e. from something like:
        {a.b a.b a.b a.b} where .b may have size [1 x m]

        to:

        a.b, where .b is of size [n x m], in this example above [4 x m]

        Parameters
        -------------------------
        hist_cell_array: a numpy array of Histogram objects, of shape (n,f),
            Where f is the number of features, and
                  n is the number of histograms per feature.

        Returns
        -------------------------
        A list of merged histograms.

        Notes
        -------------------------
        Currently each histogram object to be merged should only contain
        a single "video", i.e. a single set of data, and thus a single mean.
        Perhaps in the future this could be changed so we can merge
        MergedHistogram objects, but for now we can only merge an array
        of (not merged) Histogram objects.

        Formerly objs = seg_worm.stats.hist.mergeObjects(hist_cell_array)
        
        
        JAH: I'm not sure if this needs to be a static method. Perhaps 2 methods
        would be better, 1 for creating from self, and another from merging two
        objects

        """
        
        #TODO: add print verbose support
        #print("In HistogramManager.merge_histograms... # of "
        #      "histograms to merge:", len(hist_matrix))
        
        #TODO: readd video check ... (old code)
        #        num_videos_per_histogram = np.array([hist.num_videos for hist
        #                                     in flat_array
        #                                     if hist is not None])
        
        n_features = hist_matrix.shape[0]
        merged_histograms = np.full(n_features,None)
        
        for i, row in enumerate(hist_matrix):
            merged_histograms[i] = \
                MergedHistogram.merged_histogram_factory(row)
                            
        return merged_histograms

    def plot_information(self):
        """
        Plot diagnostic information about what histograms are available.

        """
        valid_2d_mask = self.valid_2d_mask

        # Cumulative chart of false entries (line chart)
        plt.figure()
        plt.plot(np.cumsum(np.sum(~valid_2d_mask, axis=1)))
        plt.xlabel('Feature #')
        plt.ylabel('Number of invalid histograms')
        plt.show()

        # False entries by video (bar chart)
        plt.figure()
        plt.bar(x=np.arange(valid_2d_mask.shape[1]),
                height=np.sum(~valid_2d_mask, axis=0))
        plt.xlabel('Video #')
        plt.ylabel('Number of unavailable histograms')
        plt.show()

        #JAH: This information was not being used so I commented it out
        # List of features with no histograms
        #blank_feature_list = np.flatnonzero(np.all(~valid_2d_mask, axis=1))
        #valid_feature_list = np.flatnonzero(~np.all(~valid_2d_mask, axis=1))

        
        #This can never be correct, assumes a feature manipulation
        #feature_spec = WormFeatures.get_feature_spec(extended=True)
        
        #TODO: This needs to be done differently
        #print('Features that had no histograms for any video:')
        #print(feature_spec.ix[blank_feature_list][['feature_field',
        #                                           'data_type',
        #                                           'motion_type']])
        #

        # Pie chart of features that are:
        # - totally good
        # - partially bad
        # - all bad
        all_bad = len(np.flatnonzero(np.all(~valid_2d_mask, axis=1)))
        all_good = len(np.flatnonzero(np.all(valid_2d_mask, axis=1)))
        partially_bad = valid_2d_mask.shape[0] - all_bad - all_good

        print("all bad: %d, all good: %d, partially bad: %d" % (all_bad, all_good, partially_bad))
        plt.figure()
        plt.pie([all_good, partially_bad, all_bad], labels=['All Good',
                                                            'Partially bad',
                                                            'All Bad'])
        plt.show()
    

        # CREATE A PANDAS DATAFRAME OF MEANS FOR EACH HISTOGRAM!!
        print("Means of each histogram:")
        print(self.means_2d_dataframe)

        #TODO: This is hard to 
        # Set up the matplotlib figure
        plt.figure()
        #fig, ax = plt.subplots(figsize=(12, 9))
        # Draw the heatmap using seaborn
        # Heatmap (hard to read)
        sns.heatmap(valid_2d_mask, square=True)

        # ax.legend().set_visible(False)  # this doesn't seem to work
