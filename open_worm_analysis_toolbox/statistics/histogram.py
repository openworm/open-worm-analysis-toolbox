# -*- coding: utf-8 -*-
"""

TODO: Most of the attributes here are not computed initially.
We should add a paragraph or two regarding design decisions made here



This code is generally interfaced with by the HistogramManager.

The Histogram class, and its subclass, MergedHistogram:

MergedHistoram contains a constructor that accepts a list
of Histogram objects

Notes
-------------------------
Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / hist.m
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
%2Bseg_worm/%2Bstats/%40hist/hist.m

"""
import scipy as sp
import numpy as np

from .. import config, utils

#%%


class Histogram(object):
    """
    Encapsulates the notion of a single histogram for a single feature.

    All bins in this histogram have an equal bin width.

    A Histogram object can be created by one of two class methods:
        1. create_histogram factory method, which accepts raw data.
        2. merged_histogram_factory method, which accepts a list of
           histograms and then returns a new, merged, histogram from them.

    Attributes
    -----------------
    data: numpy array
    specs: Specs object
    histogram_type: str
    motion_type: str
    data_type: str

    counts: numpy array of ints
    pdf: numpy array of floats
    mean: float
    num_samples: int
    bin_boundaries: numpy array
    bin_midpoints: numpy array
    first_bin_midpoint: float
    last_bin_midpoint: float

    Notes
    -----------------
    TODO: Missing Features:
        - saving to disk
        - version comparison
        - allow loading from a saved file

    """
    #%%

    def __init__(self, feature):
        """
        Initializer

        Parameters
        ----------
        data: numpy array
            The data to be counted for the histogram
        specs: instance of Specs class
        histogram_type: string
            histogram type  # 'motion', 'simple', 'event
        motion_type: string
             # 'all', 'forward', 'paused', 'backward'
        data_type: string
            # 'all', 'absolute', 'postive', 'negative'
            This is an additional filter on the values of the data.
            Either all values are included or:
              - the absolute values are used
              - only positive values are used
              - only negative values are used

        """
        # The underlying data itself
        self.data = feature.value
        self.specs = feature.spec

        # JAH TODO: Should this be added to the spec
        # in the expansion?
        # TODO: This should also work without expanded features, so this
        # would need to be in the default spec as well (or have defaults in
        # the code)
        #
        # Maybe this could be made generic such as "feature_manipulations"
        # with a list of strings that get concatenated together
        #
        #=======================================
        # "Expanded" features specifications
        #self.histogram_type = histogram_type
        #self.motion_type    = motion_type
        #self.data_type      = data_type

        if self.data is not None:
            # Find a set of bins that will cover the data
            # i.e. populate self.bin_boundaries
            self.compute_covering_bins()

    #%%
    @classmethod
    def create_histogram(cls, feature):
        """
        Factory method to create a Histogram instance.

        The only thing this does beyond the Histogram constructor is
        to check if the data is empty.

        Parameters
        ------------------
        data: numpy array
        specs: instance of Specs class
        histogram_type:
        motion_type:
        data_type:

        Returns
        ------------------
        An instance of the Histogram class, prepared with
        the data provided.

        Notes
        ------------------
        Formerly located in HistogramManager, and called:
        function obj = h__createIndividualObject(self, data, specs,
                                                 histogram_type, motion_type,
                                                 data_type)

        """
        data = feature.value
        if data is None or not isinstance(data, np.ndarray) or data.size == 0:
            return None
        else:
            return cls(feature)
    #%%

    @property
    def num_samples(self):
        try:
            return self._num_samples
        except AttributeError:
            self._num_samples = len(self.data)

            return self._num_samples

    def __repr__(self):
        return utils.print_object(self)

    @property
    def description(self):
        """
        A longer version of the name, suitable for use as the title
        of a histogram plot.

        """
        return (self.specs.long_field + ' ' +
                ', motion_type:' + self.motion_type +
                ', data_type: ' + self.data_type)

    @property
    def first_bin_midpoint(self):
        return self.bin_midpoints[0]

    @property
    def last_bin_midpoint(self):
        return self.bin_midpoints[-1]

    @property
    def num_bins(self):
        """
        An integer; the number of bins.

        """
        return len(self.bin_midpoints)
    #%%

    def compute_covering_bins(self):
        """
        Compute histogram bin boundaries that will be enough to cover
        the given data

        Parameters
        ----------
        None, but we will use member variables:
        self.data: numpy array
            This is the data for which we must have enough bins to cover
        self.bin_width: float
            The width of the bins

        Returns
        -------
        None
            However, self.bin_boundaries, a numpy array, is populated.
            The bin_boundaries are the boundaries of the bins that will
            accomodate the data given.
            All bins are right half-open except the last, which is closed.
            i.e. if the array edges = (a1, a2, ..., a(n+1) was returned,
                 there are n bins and
                 bin #1 = [a1, a2)
                 bin #2 = [a2, a3)
                 ...
                 bin #n = [an, an+1]

        Notes
        -----
        This version may have an extra bin than the previous version but
        this one is MUCH simpler and merging should be much simpler as edges
        should always align ...
        -   min -65.4
        -   max 20.01
        -   bin_width 1
        -   Old:
            -   boundaries -65.5 to 20.5
        -   New:
            -   boundaries -70 to 21

        Formerly:
        function [bins,edges] = h__computeBinInfo(data,bin_width)

        """
        # Compute the data range.  We apply np.ravel because for some reason
        # with posture.bends.head.mean the data was coming in like:
        # >> self.data
        # array([[-33.1726576 ], [-33.8501644 ],[-32.60058523], ...])
        # Applying ravel removes any extraneous array structure so it becomes:
        # array([-33.1726576, -33.8501644, -32.60058523, ...])
        min_data = min(np.ravel(self.data))
        max_data = max(np.ravel(self.data))

        bin_width = self.specs.bin_width

        # Let's "snap the bins to a grid" if you will, so that they will
        # line up when we try to merge multiple histograms later.
        # so if the bin_width = 2 and the min_data = 11, we will
        # start the first bin at 10, since that is a multiple of the
        # bin width.
        min_boundary = np.floor(min_data / bin_width) * bin_width
        max_boundary = np.ceil(max_data / bin_width) * bin_width

        # If we have a singular value, then we will get a singular edge,
        # which isn't good for binning. We always need to make sure that
        # our data is bounded by a high and low end. Given how hist works
        # (it is inclusive on the low end, when we only have one edge we
        # add a second edge by increasing the high end, NOT by decreasing
        # the low end.
        #
        # i.e. In Matlab you can't bound 3 by having edges at 2 & 3, the
        #      edges would need to be at 3 & 4
        if min_boundary == max_boundary:
            max_boundary = min_boundary + bin_width

        num_bins = (max_boundary - min_boundary) / bin_width

        if num_bins > config.MAX_NUMBER_BINS:
            raise Exception("Given the specified resolution of " +
                            str(bin_width) + ", the number of data " +
                            "bins exceeds the maximum, which has been " +
                            "set to MAX_NUMBER_BINS = " +
                            str(config.MAX_NUMBER_BINS))

        self.bin_boundaries = np.arange(min_boundary,
                                        max_boundary + bin_width,
                                        step=bin_width)

        # Because of the nature of floating point figures we can't guarantee
        # that these asserts work without the extra buffer of + self.bin_width
        # (though this bound could probably be greatly improved)
        assert(min_data >= self.bin_boundaries[0] - bin_width)
        assert(max_data <= self.bin_boundaries[-1] + bin_width)

    @property
    def bin_width(self):
        """
        float

        """
        return self.specs.bin_width

    @property
    def bin_midpoints(self):
        """
        Return a numpy array of the midpoints of all the bins.

        """
        try:
            return self._bin_midpoints
        except AttributeError:
            self._bin_midpoints = (self.bin_boundaries[:-1] +
                                   self.specs.bin_width / 2)

            return self._bin_midpoints

    @property
    def counts(self):
        """
        The actual counts for the bins given the data.

        Returns
        ----------------
        numpy array of int
            The values of the histogram

        """
        try:
            return self._counts
        except AttributeError:
            self._counts, _ = np.histogram(self.data,
                                           bins=self.bin_boundaries)

            return self._counts

    @property
    def pdf(self):
        """
        The probability distribution function (PDF).

        A numpy array.

        """
        try:
            return self._pdf
        except AttributeError:
            if sum(self.counts) == 0:
                # Handle the divide-by-zero case
                self._pdf = None
            else:
                self._pdf = self.counts / sum(self.counts)

            return self._pdf

    #%%
    @property
    def mean(self):
        """
        A float.  The mean of the data.

        """
        try:
            return self._mean
        except AttributeError:
            self._mean = np.mean(self.data)

            return self._mean

    @property
    def std(self):
        """
        The standard deviation.

        """
        try:
            return self._std
        except AttributeError:
            num_samples = len(self.data)
            if num_samples == 1:
                self._std = 0
            else:
                # We can optimize standard deviation computation
                # since we've already calculated the mean above
                self._std = np.sqrt \
                    (
                        (1 / (num_samples - 1)) *
                        sum((self.data - self.mean)**2)
                    )

            return self._std

    @property
    def num_videos(self):
        """
        Return 1 since this is a simple Histogram, not one created from
        the merging of multiple underlying Histograms.

        """
        return 1


###############################################################################
#%%
class MergedHistogram(Histogram):
    """
    A Histogram, plus some extra data about the individual histograms
    that make it up.

    Extra attributes:
    --------------------
    mean_per_video: numpy array of floats
        The means of the original constituent histograms making up
        this merged histogram.
    std_per_video: numpy array of floats
        Same as mean_per_video but for standard deviation.
    num_samples_per_video: numpy array of ints
    num_videos: int
    num_valid_videos: int
    all_videos_valid: bool
    no_valid_videos: bool
    p_normal: float

    """

    def __init__(self, specs):
        self.data = None
        self.specs = specs
        #super(MergedHistogram, self).__init__(data, specs)

    #%%
    @classmethod
    def merged_histogram_factory(cls, histograms):
        """
        Given a list of histograms, return a new Histogram instance
        with all the bin counts merged.

        This method assumes the specs and histogram_type, motion_type and
        data_type for all the histograms in histograms are the same.

        This method can merge histograms that are computed using different
        bins. IMPORTANTLY, because of the way that we do the bin
        definitions, bin edges will always match if they are close, i.e.
        we'll NEVER have:

        edges 1: 1,2,3,4,5
        edges 2: 3.5,4.5,5.5,6.5

        Instead we might have:
        edges 2: 3,4,5,6,7,8

        This simplifies the merging process a bit. This is accomplished by
        always setting bin edges at multiples of the bin_width. This was
        not done in the original Schafer Lab code.

        Parameters
        ------------------
        histograms: a list of Histogram objects

        Returns
        ------------------
        A Histogram object

        """
        # Create an output object with same meta properties
        merged_hist = cls(specs=histograms[0].specs)

        # Let's concatenate all the underlying data in case anyone downstream
        # wants to see it.  It's not needed for the bin and count calculation,
        # since we do that efficiently by aligning the bins.
        merged_hist.data = np.concatenate([x.data for x in histograms])

        # Align all bins
        # ---------------------------------------------------------------
        num_bins = [x.num_bins for x in histograms]
        first_bin_midpoints = [x.first_bin_midpoint for x in histograms]
        min_bin_midpoint = min(first_bin_midpoints)
        max_bin_midpoint = max([x.last_bin_midpoint for x in histograms])

        cur_bin_width = merged_hist.specs.bin_width
        new_bin_midpoints = np.arange(min_bin_midpoint,
                                      max_bin_midpoint + cur_bin_width,
                                      step=cur_bin_width)

        # Colon operator was giving warnings about non-integer indices :/
        # - @JimHokanson
        start_indices = ((first_bin_midpoints - min_bin_midpoint) /
                         cur_bin_width)
        start_indices = start_indices.round()
        end_indices = start_indices + num_bins

        num_histograms = len(histograms)
        new_counts = np.zeros((num_histograms, len(new_bin_midpoints)))

        for i in range(num_histograms):
            cur_start = start_indices[i]
            if not np.isnan(cur_start):
                cur_end = end_indices[i]
                new_counts[i, cur_start:cur_end] = histograms[i].counts

        num_samples_array = np.array([x.num_samples for x in histograms])

        # Update final properties
        # Note that each of these is now no longer a scalar as in the
        # single-video case; it is now a numpy array
        # ---------------------------------------------------------------
        merged_hist._bin_midpoints = new_bin_midpoints
        merged_hist._counts = new_counts
        merged_hist._pdf = (sum(merged_hist._counts, 0) /
                            sum(num_samples_array))

        merged_hist.mean_per_video = \
            np.array([x.mean for x in histograms])

        merged_hist.std_per_video = \
            np.array([x.std for x in histograms])

        merged_hist.num_samples_per_video = \
            np.array([x.num_samples for x in histograms])

        merged_hist._num_samples = sum(merged_hist.num_samples_per_video)

        return merged_hist

    @property
    def mean(self):
        try:
            return self._mean
        except AttributeError:
            self._mean = np.mean(self.mean_per_video)

            return self._mean

    @property
    def std(self):
        try:
            return self._std
        except AttributeError:
            # TODO
            # I'm pretty sure this is not very good; we should be calculating
            # standard deviation from a concatenation of the underlying data,
            # not from just the means of the videos. - @MichaelCurrie
            self._std = np.std(self.mean_per_video)

            return self._std

    @property
    def valid_videos_mask(self):
        return ~np.isnan(self.mean_per_video)

    @property
    def valid_mean_per_video(self):
        return self.mean_per_video[self.valid_videos_mask]

    @property
    def num_videos(self):
        """
        The number of videos that this instance contains.

        """
        return len(self.mean_per_video)

    @property
    def num_valid_videos(self):
        """

        Returns
        -----------
        int
            The number of non-NaN means across all stored video means.

        Notes
        -----------
        Also known as num_samples, or n_samples.

        """
        return sum(~np.isnan(self.mean_per_video))

    @property
    def all_videos_valid(self):
        """
        Returns
        ------------------
        boolean
            True if there are no NaN means
            False if there is even one NaN mean
        """
        return all(~np.isnan(self.mean_per_video))

    @property
    def no_valid_videos(self):
        """
        Returns
        -----------
        bool
            True if there was at least one video that wasn't NaN for this
            Histogram.
        """
        return self.num_valid_videos == 0

    @property
    def p_normal(self):
        """
        Shapiro-Wilk normality test:

        Estimate of the probability that the video means were drawn from
        a normal distribution.

        Returns
        -----------
        float

        Notes
        -----------
        Formerly:
        seg_worm.fex.swtest(data(i).dataMeans, 0.05, 0)

        """
        # The try-except structure allows for lazy evaluation: we only
        # compute the value the first time it's asked for, and then never
        # again.
        try:
            return self._p_normal
        except AttributeError:
            if self.num_valid_measurements < 3:
                self._p_normal = np.NaN
            else:
                # Shapiro-Wilk parametric hypothsis test of composite normality.
                # i.e. test the null hypothesis that the data was drawn from
                # a normal distribution.
                # Note: this is a two-sided test.
                # The previous method was to use swtest(x, 0.05, 0) from
                # Matlab Central: http://www.mathworks.com/matlabcentral/
                # fileexchange/46548-hockey-stick-and-climate-change/
                # content/Codes_data_publish/Codes/swtest.m
                _, self._p_normal = sp.stats.shapiro(self.mean_per_video)

            return self._p_normal
