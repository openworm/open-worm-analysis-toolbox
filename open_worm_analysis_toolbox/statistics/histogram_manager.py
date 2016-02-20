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
    hist_cell_array:
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
        feature_path_or_object_list: list of strings or feature objects
            Full paths to all feature files making up this histogram, or
            their in-memory object equivalents.

        """
        if verbose:
            print("Number of feature files passed into the histogram manager:",
                  len(feature_path_or_object_list))

        # This will have shape (len(feature_path_or_object_list), 726)
        self.hist_cell_array = []

        # Loop over all feature files and get histogram objects for each
        for feature_path_or_object in feature_path_or_object_list:
            worm_features = None

            if isinstance(feature_path_or_object, six.string_types):
                # If we have a string, it's a filepath to an HDF5 feature file
                file_path = feature_path_or_object
                worm_features = WormFeaturesDos.from_disk(file_path)
            else:
                # Otherwise the worm features have been passed directly
                # as an instance of WormFeatures (we hope)
                worm_features = feature_path_or_object

            # TODO: Need to add on info to properties
            # worm_features.info -> obj.info

            new_histogram_set = self.init_histograms(worm_features)
            self.hist_cell_array.append(new_histogram_set)

        # JAH TODO: I'm not sure what this is doing ..., add documentation
        #----------------------------------------------------------------
        # Convert to a numpy array

        # If self.hist_cell_array elements are the same size (e.g we have n of m long arrays)
        # then the result is a single array of (m,n)
        # However if of those n elements, some are not m long, then we get
        # an array of length n, where each contains its elements
        #[(m1),(m2),(m3),(m4),...etc]
        self.hist_cell_array = np.array(self.hist_cell_array)

        # At this point hist_cell_array is a list, with one element for
        # each video.
        # Each element is a numpy array of 700+ Histogram instances.
        # Here we merge them and we assign to self.merged_histograms a
        # numpy array of 700+ Histogram instances, for the merged video
        self.merged_histograms = \
            HistogramManager.merge_histograms(self.hist_cell_array)

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
        return self.hist_cell_array.shape[0]

    @property
    def valid_2d_mask(self):
        """
        Return a mask showing which elements of hist_cell_array are null.

        """
        valid_mean_detector = lambda h: True if h is not None else False
        return np.vectorize(valid_mean_detector)(self.hist_cell_array)

    @property
    def means_2d_dataframe(self):
        """
        Returns
        -----------
        Pandas dataframe
            Shape (10,726)

        """
        valid_mean_detector = lambda h: h.mean if h is not None else np.NaN
        means_array = np.vectorize(valid_mean_detector)(self.hist_cell_array)

        # Change shape to (726,10) since pandas wants the first axis
        # to be the rows of the dataframe
        means_array = np.rollaxis(means_array, axis=1)

        df = pd.DataFrame(data=means_array)
        # Give a more human-readable column name
        df.columns = ['Video %d mean' % i for i in range(self.num_videos)]

        feature_spec = WormFeaturesDos.get_feature_spec(extended=True)
        feature_spec = feature_spec[['feature_field',
                                     'data_type',
                                     'motion_type']]

        return feature_spec.join(df)

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
    def merge_histograms(hist_cell_array):
        """
        The goal of this function is to go from n collections of 708
        histogram summaries of features each, to one set of 708 histogram
        summaries, that has n elements, one for each video

        i.e. from something like:
        {a.b a.b a.b a.b} where .b may have size [1 x m]

        to:

        a.b, where .b is of size [n x m], in this case [4 x m]

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

        """

        # DEBUG
        print("In HistogramManager.merge_histograms... # of "
              "histograms to merge:", len(hist_cell_array))

        # Make sure that the hist_cell_array is a numpy array
        hist_cell_array = np.array(hist_cell_array)

        # Check that we don't have any multiple videos in any histogram,
        # since it's not implemented to merge already-merged histograms
        num_videos_per_histogram = np.array([hist.num_videos for hist
                                             in hist_cell_array.flatten()
                                             if hist is not None])
        if np.count_nonzero(num_videos_per_histogram != 1) > 0:
            raise Exception("Merging already-merged histograms is not yet "
                            "implemented")

        # Let's assign some nicer names to the dimensions of hist_cell_array
        (num_histograms_per_feature, num_features) = hist_cell_array.shape

        # Pre-allocate space for the 700+ Histogram objects
        merged_histograms = np.array([None] * num_features)

        # Go through each feature and create a merged histogram
        for feature_index in range(num_features):
            histograms = hist_cell_array[:, feature_index]

            # This is @MichaelCurrie's kludge to step over features that
            # for some reason didn't get all their histograms populated
            none_hist_found = False
            for hist in histograms:
                if hist is None:
                    none_hist_found = True
            if none_hist_found:
                if num_histograms_per_feature == 1:
                    print("The histogram is None for feature #%d.  Bypassing."
                          % feature_index)
                else:
                    if histograms[0] is not None:
                        long_field = histograms[0].specs.name
                    else:
                        long_field = 'NAME UNAVAILABLE'
                    print("For feature #%d (%s), at least one video is None. "
                          "Bypassing." % (feature_index, long_field))

                continue

            merged_histograms[feature_index] = \
                MergedHistogram.merged_histogram_factory(histograms)

        return merged_histograms

    def plot_information(self):
        """
        Plot diagnostic information about what histograms are available.

        """
        valid_2d_mask = self.valid_2d_mask

        # Cumulative chart of false entries (line chart)
        plt.figure()
        plt.plot(np.cumsum(np.sum(~valid_2d_mask, axis=0)))
        plt.xlabel('Feature #')
        plt.ylabel('Number of invalid histograms')
        plt.show()

        # False entries by video (bar chart)
        plt.figure()
        plt.bar(left=np.arange(valid_2d_mask.shape[0]),
                height=np.sum(~valid_2d_mask, axis=1))
        plt.xlabel('Video #')
        plt.ylabel('Number of unavailable histograms')
        plt.show()

        # List of features with no histograms
        blank_feature_list = np.flatnonzero(np.all(~valid_2d_mask, axis=0))
        valid_feature_list = np.flatnonzero(~np.all(~valid_2d_mask, axis=0))

        feature_spec = WormFeatures.get_feature_spec(extended=True)
        print('Features that had no histograms for any video:')
        print(feature_spec.ix[blank_feature_list][['feature_field',
                                                   'data_type',
                                                   'motion_type']])

        # Pie chart of features that are:
        # - totally good
        # - partially bad
        # - all bad
        all_bad = len(np.flatnonzero(np.all(~valid_2d_mask, axis=0)))
        all_good = len(np.flatnonzero(np.all(valid_2d_mask, axis=0)))
        partially_bad = valid_2d_mask.shape[1] - all_bad - all_good

        print("%d, %d, %d" % (all_bad, all_good, partially_bad))
        plt.figure()
        plt.pie([all_good, partially_bad, all_bad], labels=['All Good',
                                                            'Partially bad',
                                                            'All Bad'])

        # CREATE A PANDAS DATAFRAME OF MEANS FOR EACH HISTOGRAM!!
        print("Means of each histogram:")
        print(self.means_2d_dataframe)

        # Set up the matplotlib figure
        plt.figure()
        #fig, ax = plt.subplots(figsize=(12, 9))
        # Draw the heatmap using seaborn
        # Heatmap (hard to read)
        sns.heatmap(valid_2d_mask, square=True)

        # ax.legend().set_visible(False)  # this doesn't seem to work
