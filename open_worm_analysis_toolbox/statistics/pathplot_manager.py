# -*- coding: utf-8 -*-
"""

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
from .specifications import SimpleSpecs, EventSpecs, MovementSpecs

#%%


class HistogramManager(object):
    """
    Histograms calculated on all features for a collection of feature files
    or WormFeatures objects.

    Attributes
    -------------
    merged_histograms: numpy array of MergedHistogram objects
        This can be accessed via the overloaded [] operator

    Notes
    -------------
    Translated from the seg_worm.stats.hist.manager class

    """

    def __init__(self, feature_path_or_object_list=[]):
        """
        Parameters
        ----------
        feature_path_or_object_list: list of strings or feature objects
            Full paths to all feature files making up this histogram, or
            their in-memory object equivalents.

        """
        print("Number of feature files passed into the histogram manager:",
              len(feature_path_or_object_list))

        # Consider the case that an empty list is passed
        # (useful to initialize the object without having to create histograms)
        if not feature_path_or_object_list:
            return

        # This will have shape (len(feature_path_or_object_list), 726)
        self.hist_cell_array = []

        # Loop over all feature files and get histogram objects for each
        for feature_path_or_object in feature_path_or_object_list:
            worm_features = None

            if isinstance(feature_path_or_object, six.string_types):
                # If we have a string, it's a filepath to an HDF5 feature file
                feature_file = h5py.File(feature_path_or_object, 'r')
                #worm_features = feature_file["worm"]
                worm_features = WormFeatures.from_disk(feature_path_or_object)
                feature_file.close()
            else:
                # Otherwise the worm features have been passed directly
                # as an instance of WormFeatures (we hope)
                worm_features = feature_path_or_object

            # TODO: Need to add on info to properties
            # worm_features.info -> obj.info

            self.hist_cell_array.append(self.init_histograms(worm_features))

        # Convert to a numpy array
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

        feature_spec = WormFeatures.get_feature_spec(extended=True)
        feature_spec = feature_spec[['feature_field',
                                     'data_type',
                                     'motion_type']]

        return feature_spec.join(df)

    #%%
    def init_histograms(self, worm_features):
        """
        For a given set of worm features, prepare a 1D array of Histogram
        instances, consisting of, in order:
        - The movement histograms
        - The "simple" histograms
        - The event histograms

        Parameters
        ------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology,
            path, locomotion, in an h5py group.
            (seg_worm.features or strut) This may truly be a feature
            object or the old structure. Both have the same format.
            -@JimHokanson

        Notes
        ------------------
        Formerly:
        hist_objs = seg_worm.stats.hist.manager.initObjects(feature_file_paths)

        Potential Improvements
        ------------------
        - We could optimize the histogram calculation data for motion data
        - For event data, we should remove partial events (events that
          start at the first frame or end at the last frame)

        """

        # Movement histograms
        m_hists = self.__movement_histograms(worm_features,
                                             MovementSpecs.specs_factory())

        # Simple histograms
        s_hists = self.__simple_histograms(worm_features,
                                           SimpleSpecs.specs_factory())

        # Event histograms

        # :/ HACK: - @JimHokanson
        # Just get the size from the size of one of the pieces of data
        num_samples = len(worm_features.morphology.length)

        e_hists = self.__event_histograms(worm_features,
                                          EventSpecs.specs_factory(),
                                          num_samples)

        # Put all these histograms together into one single-dim numpy array.
        return np.hstack((m_hists, s_hists, e_hists))

    ###########################################################################
    # THREE FUNCTIONS TO CONVERT DATA TO HISTOGRAMS:
    ## __simple_histograms, __movement_histograms, __event_histograms
    ###########################################################################
    #%%
    def __simple_histograms(self, worm_features, specs):
        """
        Compute simple histograms

        Parameters
        ---------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology,
            path, locomotion, in an h5py group.
        specs: A list of SimpleSpecs instances

        Returns
        --------------------
        A list of Histogram instances, one for each of the simple features

        """
        return [Histogram.create_histogram(
                utils.filter_non_numeric(specs[iSpec].get_data(worm_features)),
                specs[iSpec],
                'simple', 'all', 'all')
                for iSpec in range(len(specs))]

    #%%
    def __movement_histograms(self, worm_features, specs):
        """
        For movement features, we compute either 4 or 16 histogram objects,
        depending on whether or not the feature can be signed. If the data is
        not signed, then we compute 4. If it is, we compute 16. We compute
        histograms for when the motion of the midbody is:
            - going forward
            - going backward
            - paused
            - all 3 of the above combined

        If signed, this also gets computed on feature data that is:
            - positive
            - negative
            - absolute value of data
            - both positive and negative

        By combining the motion of the midbody with the sign of the data, we
        get 16 different possible combinations for each feature specification.

        Parameters
        -------------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology,
            path, locomotion, in an h5py group.
        specs: A list of MovementSpecs instances

        Returns
        --------------------
        A list of Histogram instances, one for each of the movement features

        Notes
        -------------------------
        Formerly m_hists = h_computeMHists(worm_features, specs)

        We could significantly reduce the amount of binning done in this
        function - @JimHokanson

        """
        motion_modes = worm_features.locomotion.motion_mode

        num_frames = len(motion_modes)

        indices_use_mask = {}
        indices_use_mask["all"] = np.ones(num_frames, dtype=bool)
        indices_use_mask["forward"] = motion_modes == 1
        indices_use_mask["backward"] = motion_modes == -1
        indices_use_mask["paused"] = motion_modes == 0

        # NOTE: motion types refers to the motion of the worm's midbody
        motion_types = ['all', 'forward', 'paused', 'backward']
        data_types = ['all', 'absolute', 'positive', 'negative']

        movement_histograms = []

        for cur_spec in specs:
            cur_data = cur_spec.get_data(worm_features)

            good_data_mask = ~utils.get_non_numeric_mask(cur_data).flatten()

            # Now let's create 16 histograms, for each element of
            # (motion_types x data_types)

            for cur_motion_type in motion_types:
                if (good_data_mask.size !=
                        indices_use_mask[cur_motion_type].size):
                    # DEBUG
                    #import pdb
                    # pdb.set_trace()

                    assert(good_data_mask.size ==
                           indices_use_mask[cur_motion_type].size)

                cur_mask = indices_use_mask[cur_motion_type] & good_data_mask
                assert(isinstance(cur_data, np.ndarray))
                assert(isinstance(cur_mask, np.ndarray))
                assert(cur_data.size == cur_mask.size)

                temp_data = cur_data[cur_mask]

                # Create the histogram for the case where we consider all
                # numeric data
                all_hist = Histogram.create_histogram(temp_data,
                                                      cur_spec,
                                                      'motion',
                                                      cur_motion_type,
                                                      data_types[0])

                movement_histograms.append(all_hist)

                if cur_spec.is_signed:

                    # Histogram for the data made absolute
                    # TODO: This could be improved by merging results
                    #       from positive and negative - @JimHokanson
                    abs_hist = Histogram.create_histogram(abs(temp_data),
                                                          cur_spec,
                                                          'motion',
                                                          cur_motion_type,
                                                          data_types[1])

                    # TODO: To get a speed-up, we could avoid reliance on
                    # create_histogram.  Instead, we could take the
                    # positive and negative aspects of the object
                    # that included all data. - @JimHokanson
                    # (see the SegWormMatlabClasses version of this to see
                    #  how this could be done)

                    # Histogram for just the positive data
                    pos_hist = Histogram.create_histogram(
                        temp_data[temp_data >= 0],
                        cur_spec,
                        'motion',
                        cur_motion_type,
                        data_types[2])

                    # Histogram for just the negative data
                    neg_hist = Histogram.create_histogram(
                        temp_data[temp_data <= 0],
                        cur_spec,
                        'motion',
                        cur_motion_type,
                        data_types[3])

                    # Append our list with these histograms
                    movement_histograms.append(abs_hist)
                    movement_histograms.append(pos_hist)
                    movement_histograms.append(neg_hist)

        return movement_histograms

    #%%
    def __event_histograms(self, worm_features, specs, num_samples):
        """
        Compute event histograms.  We produce four histograms for each
        specification in specs, for:
        - all data
        - absolute data
        - positive data
        - negative data

        Parameters
        ---------------------
        worm_features : A WormFeatures instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology,
            path, locomotion, in an h5py group.
        specs: a list of EventSpecs instances
        num_samples: int
            number of samples

        Returns
        --------------------
        A list of Histogram instances, one for each of the event features

        """
        temp_hists = []

        for cur_specs in specs:

            cur_data = cur_specs.get_data(worm_features, num_samples)

            # Remove the NaN and Inf entries
            cur_data = utils.filter_non_numeric(cur_data)

            # Calculate the first histogram, on all the data.
            temp_hists.append(
                Histogram.create_histogram(
                    cur_data,
                    cur_specs,
                    'event',
                    'all',
                    'all'))

            # If the data is signed, we calculate three more histograms:
            # - On an absolute version of the data,
            # - On only the positive data, and
            # - On only the negative data.
            if cur_specs.is_signed:
                if cur_data is None:
                    # TODO: This is a bit opaque and should be clarified
                    # The call to create_histograms() just returns None, so
                    # we put together a bunch of None's
                    # in a list and append, rather than calling the
                    # function three times
                    temp_hists = temp_hists + [None, None, None]
                else:
                    temp_hists.append(
                        Histogram.create_histogram(
                            abs(cur_data),
                            cur_specs,
                            'event',
                            'all',
                            'absolute'))

                    positive_histogram = \
                        Histogram.create_histogram(cur_data[cur_data > 0],
                                                   cur_specs, 'event',
                                                   'all', 'positive')
                    negative_histogram = \
                        Histogram.create_histogram(cur_data[cur_data < 0],
                                                   cur_specs, 'event',
                                                   'all', 'negative')

                    temp_hists.append(positive_histogram)
                    temp_hists.append(negative_histogram)

        return temp_hists

    #%%
    ###########################################################################
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
                        long_field = histograms[0].specs.long_field
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
