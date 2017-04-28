# -*- coding: utf-8 -*-
"""
Calculate the "Bends" locomotion feature

JAH: 2014-10-29 - this documentation is out of date

Contains two classes:
  LocomotionCrawlingBends, which yields properties:
    .head
      .amplitude
      .frequency
    .mid
      .amplitude
      .frequency
    .tail
      .amplitude
      .frequency

  LocomotionForagingBends, which yields properties:
    .amplitude
    .angleSpeed

"""

import numpy as np
import scipy.ndimage.filters as filters

import warnings

from . import generic_features
from .generic_features import Feature
from .. import utils

class BendHelper(object):
    def h__getBendData(self, avg_bend_angles, bound_info, options, fps):
        """
        Compute the bend amplitude and frequency.

        Parameters
        ----------
        avg_bend_angles: numpy.array
            - [1 x n_frames]
        bound_info:
        options: open-worm-analysis-toolbox.features.feature_processing_options.LocomotionCrawlingBends
        fps: float
            Frames Per Second


        Returns
        -------

        """

        # Compute the short-time Fourier transforms (STFT).
        #--------------------------------------------------
        # Unpack options ...
        
        #make sure the frames per seconds is an integer
        fps = int(fps)

        max_freq = options.max_frequency(fps)
        min_freq = options.min_frequency
        fft_n_samples = options.fft_n_samples
        max_amp_pct_bandwidth = options.max_amplitude_pct_bandwidth
        peak_energy_threshold = options.peak_energy_threshold

        # Maximum index to keep for frequency analysis:
        fft_max_I = int(fft_n_samples / 2)
        # This gets multiplied by an index to compute the frequency at that
        # index
        freq_scalar = (fps / 2) * 1 / (fft_max_I - 1)

        n_frames = len(avg_bend_angles)
        amps = np.empty(n_frames) * np.NaN
        freqs = np.empty(n_frames) * np.NaN

        left_bounds = bound_info.left_bounds
        right_bounds = bound_info.right_bounds
        is_bad_mask = bound_info.is_bad_mask

        # This is a processing optimization that in general will speed
        # things up
        max_freq_I = max_freq / freq_scalar
        INIT_MAX_I_FOR_BANDWIDTH = \
            round(options.initial_max_I_pct * max_freq_I)

        # Convert each element from float to int
        right_bounds = right_bounds.astype(int)
        left_bounds = left_bounds.astype(int)

        for iFrame in np.flatnonzero(~is_bad_mask):
            windowed_data = avg_bend_angles[
                left_bounds[iFrame]:right_bounds[iFrame]]
            data_win_length = len(windowed_data)

            #
            # fft frequency and bandwidth
            #
            # Compute the real part of the STFT.
            # These two steps take a lot of time ...

            # New code:
            fft_data = abs(np.fft.rfft(windowed_data, fft_n_samples))

            # Find the peak frequency.
            maxPeakI = np.argmax(fft_data)
            maxPeak = fft_data[maxPeakI]

            # NOTE: If this is true, we'll never bound the peak on the left.
            # We are looking for a hump with a peak, not just a decaying
            # signal.
            if maxPeakI == 0:
                continue

            unsigned_freq = freq_scalar * maxPeakI

            if not (min_freq <= unsigned_freq <= max_freq):
                continue

            peakStartI, peakEndI = \
                self.h__getBandwidth(data_win_length,
                                     fft_data,
                                     maxPeakI,
                                     INIT_MAX_I_FOR_BANDWIDTH)
            
            if np.isnan(peakStartI) or np.isnan(peakEndI):
                #wrong indexes, next loop
                continue

            # Store data
            #------------------------------------------------------------------
            fenergy = fft_data**2
            tot_energy = np.sum(fenergy)
            peak_energy = np.sum(fenergy[peakStartI:peakEndI])

            peak_amplitud_treshold = (max_amp_pct_bandwidth * maxPeak)
            if not (# The minima can't be too big:
                    fft_data[peakStartI] > peak_amplitud_treshold or
                    fft_data[peakEndI] > peak_amplitud_treshold or
                    # Needs to have enough energy:
                    (peak_energy < (peak_energy_threshold * tot_energy))
                    ):

                # Convert the peak to a time frequency.
                dataSign = np.sign(np.nanmean(windowed_data))  # sign the data
                amps[iFrame] = (2 * fft_data[maxPeakI] /
                                data_win_length) * dataSign
                freqs[iFrame] = unsigned_freq * dataSign

        return amps, freqs


    def h__getBandwidth(self, data_win_length, fft_data,
                        max_peak_I, INIT_MAX_I_FOR_BANDWIDTH):
        """
        The goal is to find minimum 'peaks' that border the maximal frequency
        response.

        Since this is a time-intensive process, we try and start with a small
        range of frequencies, as execution time is proportional to the length
        of the input data.  If this fails we use the full data set.

        Called by: h__getBendData

        Parameters
        ----------
        data_win_length
          Length of real data (ignoring zero padding) that
          went into computing the FFT

        fft_data
          Output of the fft function

        max_peak_I
          Location (index) of the maximum of fft_data

        INIT_MAX_I_FOR_BANDWIDTH
          See code


        Returns
        -------
        peak_start_I: scalar

        peak_end_I: scalar


        Notes
        ---------------------------------------
        Formerly [peak_start_I,peak_end_I] = \
               h__getBandwidth(data_win_length, fft_data,
                               max_peak_I, INIT_MAX_I_FOR_BANDWIDTH)

        See also, formerly: seg_worm.util.maxPeaksDist

        """

        peakWinSize = round(np.sqrt(data_win_length))

        # Find the peak bandwidth.
        if max_peak_I < INIT_MAX_I_FOR_BANDWIDTH:
            # NOTE: It is incorrect to filter by the maximum here, as we want to
            # allow matching a peak that will later be judged invalid. If we
            # filter here we may find another smaller peak which will not be
            # judged invalid later on.
            min_peaks, min_peaks_I = utils.separated_peaks(
                fft_data[:INIT_MAX_I_FOR_BANDWIDTH],
                peakWinSize,
                use_max=False,
                value_cutoff=np.inf)

            del min_peaks   # this part of max_peaks_dist's return is unused

            # TODO: This is wrong, replace add find to utils ...
            peak_start_I = min_peaks_I[utils.find(min_peaks_I < max_peak_I, 1)]
            peak_end_I = min_peaks_I[utils.find(min_peaks_I > max_peak_I, 1)]
        else:
            peak_start_I = np.array([])
            peak_end_I = np.array([])

        # NOTE: Besides checking for an empty value, we also need to ensure that
        # the minimum didn't come too close to the data border, as more data
        # could invalidate the result we have.
        #
        # NOTE: In order to save time we only look at a subset of the FFT data.
        if (peak_end_I.size == 0) | \
                (peak_end_I + peakWinSize >= INIT_MAX_I_FOR_BANDWIDTH):
            # If true, then rerun on the full set of data
            [min_peaks, min_peaks_I] = utils.separated_peaks(
                fft_data, peakWinSize, use_max=False, value_cutoff=np.inf)

            del(min_peaks)  # This part of max_peaks_dist's return is unused

            peak_start_I = min_peaks_I[utils.find(min_peaks_I < max_peak_I, 1)]
            peak_end_I = min_peaks_I[utils.find(min_peaks_I > max_peak_I, 1)]

        assert peak_start_I.size <= 1
        assert peak_end_I.size <= 1
        
        #return an array it is problematic, and can give rise to deprecation errors. Let's return a tuple instead.
        peak_start_Int = int(peak_start_I[0]) if peak_start_I.size == 1 else np.nan
        peak_end_Int = int(peak_end_I[0]) if peak_end_I.size == 1 else np.nan

        # TODO: Why is this not a tuple - tuple would be more consistent
        return (peak_start_Int, peak_end_Int)

class LocomotionBend(object):
    """
    Element for LocomotionCrawlingBends

    """

    def __init__(self, amplitude, frequency, name):
        self.amplitude = amplitude
        self.frequency = frequency
        self.name = name

    @classmethod
    def from_disk(cls, bend_ref, name):

        self = cls.__new__(cls)

        self.amplitude = utils._extract_time_from_disk(bend_ref, 'amplitude')
        self.frequency = utils._extract_time_from_disk(bend_ref, 'frequency')
        self.name = name

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):

        # merge_nans=True - utils.separated_peaks works slightly differently
        # so merge_nans is true here, for now. The utils function works correctly
        # and the old version works incorrectly but was convoluted enough that
        # it was hard to replicate

        return utils.correlation(
            self.amplitude,
            other.amplitude,
            'locomotion.bends.' +
            self.name +
            '.amplitude',
            merge_nans=True) and utils.correlation(
            self.frequency,
            other.frequency,
            'locomotion.bends.' +
            self.name +
            '.frequency',
            merge_nans=True)


class LocomotionCrawlingBends(BendHelper):
    """
    Locomotion Crawling Bends Feature.

    Attributes
    ----------
    head : LocomotionBend
    midbody : LocomotionBend
    tail : LocomotionBend

    Notes
    ---------------------------------------
    Formerly +segworm/+features/@locomotion/getLocomotionBends

    Originally, part of wormBends.m


    Note from Ev Yemini on Setup Options
    ---------------------------------------
    Empirically I've found the values below achieve good signal.

    Furthermore:
    The body bend frequency is much easier to see (than foraging). The N2
    signal is clearly centered around 1/3Hz in both the literature and
    through visual inspection.

    I chose a high-frequency threshold of 4 frames. With 4 frames a 3-frame
    tick, resulting from segmentation noise, will be diluted by the
    additional frame.


    Nature Methods Description
    ---------------------------------------

    Worm crawling is expressed as both an amplitude and frequency
    (Supplementary Fig. 4e). We measure these features instantaneously at
    the head, midbody, and tail. The amplitude and frequency are signed
    negatively whenever the worm’s ventral side is contained within the
    concave portion of its instantaneous bend.

    Crawling is only measured during forward and backward motion states.
    The worm bend mean angles (described in the section on “Posture”) show
    a roughly periodic signal as the crawling wave travels along the worm’s
    body. This wave can be asymmetric due to differences in dorsal-ventral
    flexibility or simply because the worm is executing a turn. Moreover
    the wave dynamics can change abruptly to speed up or slow down.
    Therefore, the signal is only roughly periodic and we measure its
    instantaneous properties.

    Worm bends are linearly interpolated across unsegmented frames. The
    motion states criteria (described earlier in this section) guarantee
    that interpolation is no more than 1/4 of a second long. For each
    frame, we search both backwards and forwards for a zero crossing in the
    bend angle mean – the location where the measured body part (head,
    midbody, or tail) must have hit a flat posture (a supplementary bend
    angle of 0°). This guarantees that we are observing half a cycle for
    the waveform. Crawling is bounded between 1/30Hz (a very slow wave that
    would not resemble crawling) and 1Hz (an impossibly fast wave on agar).

    If the window between zero crossings is too small, the nearest zero
    crossing is assumed to be noise and we search for the next available
    zero crossing in its respective direction. If the window is too big,
    crawling is marked undefined at the frame.

    Once an appropriate window has been found, the window is extended in
    order to center the frame and measure instantaneous crawling by
    ensuring that the distance on either side to respective zero crossings
    is identical. If the distances are not identical, the distance of the
    larger side is used in place of the zero-crossing distance of the
    smaller side in order to expand the small side and achieve a symmetric
    window, centered at the frame of interest.

    We use a Fourier transform to measure the amplitude and frequency
    within the window described above. The largest peak within the
    transform is chosen for the crawling amplitude and frequency. If the
    troughs on either side of the peak exceed 1/2 its height, the peak is
    rejected for being unclear and crawling is marked as undefined at the
    frame. Similarly, if the integral between the troughs is less than half
    the total integral, the peak is rejected for being weak.


    """

    bend_names = ['head', 'midbody', 'tail']

    def __init__(
            self,
            features_ref,
            bend_angles,
            is_paused,
            is_segmented_mask):
        """
        Compute the temporal bending frequency at the head, midbody, and tail.

        Parameters:
        -----------
        features_ref :
        bend_angles : numpy.array
            - [49 x n_frames]
        is_paused : numpy.array
            - [1 x n_frames]
            Whether or not the worm is considered to be paused during the frame
        is_segmented_mask : [1 x n_frames]

        """

        options = features_ref.options.locomotion.crawling_bends

        if not features_ref.options.should_compute_feature(
                'locomotion.crawling_bends', features_ref):
            self.head = None
            self.midbody = None
            self.tail = None
            return

        timer = features_ref.timer
        timer.tic()

        fps = features_ref.video_info.fps

        # Special Case: No worm data.
        #------------------------------------
        if ~np.any(is_segmented_mask):
            nan_data = np.empty(len(is_segmented_mask)) * np.NaN
            bend_dict = {'frequency': nan_data.copy(),
                         'amplitude': nan_data.copy()}

            raise Exception('This is no longer impelemented properly')
            self.head = bend_dict.copy()
            self.midbody = bend_dict.copy()
            self.tail = bend_dict.copy()
            return

        for cur_partition_name in self.bend_names:
            # Find the mean bend angle for the current partition, across all
            # frames

            s = slice(*options.bends_partitions[cur_partition_name])

            # Suppress RuntimeWarning: Mean of empty slice
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                avg_bend_angles = np.nanmean(bend_angles[s, :], axis=0)

            # Ensure there are both data and gaps if we are going to
            # interpolate - i.e.:
            # - that not everything is segmented (missing) - i.e. data
            # - that something is segmented - i.e. gaps
            if not(np.all(is_segmented_mask)) and np.any(is_segmented_mask):
                avg_bend_angles = utils.interpolate_with_threshold(
                    avg_bend_angles)

            bound_info = CrawlingBendsBoundInfo(
                avg_bend_angles, is_paused, options, fps)

            [amplitude, frequency] = self.h__getBendData(avg_bend_angles,
                                                         bound_info,
                                                         options,
                                                         fps)

            setattr(
                self,
                cur_partition_name,
                LocomotionBend(
                    amplitude,
                    frequency,
                    cur_partition_name))

        timer.toc('locomotion.crawling_bends')


    @classmethod
    def from_disk(cls, bend_ref):

        self = cls.__new__(cls)

        self.head = LocomotionBend.from_disk(bend_ref['head'], 'head')
        self.midbody = LocomotionBend.from_disk(bend_ref['midbody'], 'midbody')
        self.tail = LocomotionBend.from_disk(bend_ref['tail'], 'tail')

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        return self.head == other.head and \
            self.midbody == other.midbody and \
            self.tail == other.tail

#%%


class LocomotionForagingBends(object):

    """
    Locomotion Foraging Bends Feature.

    Attributes
    ----------
    amplitude
    angleSpeed


    Methods
    ---------------------------------------
    __init__
    h__computeNoseBends
    h__computeAvgAngles
    h__interpData
    h__getNoseInterpolationIndices
    h__foragingData
    h__getAmps

    Notes
    ---------------------------------------
    Formerly +segworm/+features/@locomotion/getForaging

    Originally, part of wormBends.m

    """

    def __init__(self, features_ref, is_segmented_mask, ventral_mode):
        """
        Initialize an instance of LocomotionForagingBends

        Parameters
        ----------
        nw: NormalizedWorm instance
        is_segmented_mask: boolean numpy array [1 x n_frames]
        ventral_mode: int
            0, 1, or 2 depending on the orientation of the worm.

        """

        options = features_ref.options.locomotion.foraging_bends

        if not features_ref.options.should_compute_feature(
                'locomotion.foraging_bends', features_ref):
            self.amplitude = None
            self.angle_speed = None
            return

        timer = features_ref.timer
        timer.tic()

        # self.amplitude  = None  # DEBUG
        # self.angleSpeed = None # DEBUG

        fps = features_ref.video_info.fps

        nose_x, nose_y = \
            features_ref.nw.get_partition('head_tip',
                                          data_key='skeleton',
                                          split_spatial_dimensions=True)

        neck_x, neck_y = \
            features_ref.nw.get_partition('head_base',
                                          data_key='skeleton',
                                          split_spatial_dimensions=True)

        # TODO: Add "reversed" and "interpolated" options to the get_partition
        # function, to replace the below blocks of code!
        #----------------------------------------------------------------------

        # We need to flip the orientation (i.e. reverse the entries along the
        # first, or skeleton index, axis) for angles and consistency with old
        # code:
        nose_x = nose_x[::-1, :]
        nose_y = nose_y[::-1, :]
        neck_x = neck_x[::-1, :]
        neck_y = neck_y[::-1, :]

        # Step 1: Interpolation of skeleton indices
        #---------------------------------------
        # TODO: ensure that we are excluding the points at the beginning
        # and ending of the second dimension (the frames list) of nose_x, etc.
        # from being interpolated.  (this was a step in
        # h__getNoseInterpolationIndices, that we no longer have since I've
        # put the interpolation code into
        # utils.interpolate_with_threshold_2D instead.  But we
        # might be okay, since the beginning and end are going to be left alone
        # since I've set left=np.NaN and right=np.NaN in the underlying
        # utils.interpolate_with_threshold code.
        interp = utils.interpolate_with_threshold_2D

        max_samples_interp = options.max_samples_interp_nose(fps)

        nose_xi = interp(nose_x, threshold=max_samples_interp)
        nose_yi = interp(nose_y, threshold=max_samples_interp)
        neck_xi = interp(neck_x, threshold=max_samples_interp)
        neck_yi = interp(neck_y, threshold=max_samples_interp)
        #----------------------------------------------------------------------

        # Step 2: Calculation of the bend angles
        #---------------------------------------
        nose_bends = self.h__computeNoseBends(
            nose_xi, nose_yi, neck_xi, neck_yi)

        # Step 3:
        #---------------------------------------
        [nose_amps, nose_freqs] = \
            self.h__foragingData(fps, nose_bends,
                                 options.min_nose_window_samples(fps))

        if ventral_mode > 1:
            nose_amps = -nose_amps
            nose_freqs = -nose_freqs

        self.amplitude = nose_amps
        self.angle_speed = nose_freqs

        timer.toc('locomotion.foraging_bends')

    def h__computeNoseBends(self, nose_x, nose_y, neck_x, neck_y):
        """
        Compute the difference in angles between the nose and neck (really the
        head tip and head base).

        Parameters
        ----------
        nose_x: [4 x n_frames]
        nose_y: [4 x n_frames]
        neck_x: [4 x n_frames]
        neck_y: [4 x n_frames]

        Returns
        -------
        nose_bends_d

        Notes
        ---------------------------------------
        Formerly nose_bends_d = h__computeNoseBends(nose_x,nose_y,neck_x,neck_y)

        """

        nose_angles = self.h__computeAvgAngles(nose_x, nose_y)
        neck_angles = self.h__computeAvgAngles(neck_x, neck_y)

        # TODO: These three should be a method, calculating the difference
        # in angles and ensuring all results are within +/- 180
        nose_bends_d = (nose_angles - neck_angles) * (180 / np.pi)

        # Suppress warnings so we can compare a numpy array that may contain NaNs
        # without triggering a Runtime Warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nose_bends_d[nose_bends_d > 180] -= 360
            nose_bends_d[nose_bends_d < -180] += 360

        return nose_bends_d

    def h__computeAvgAngles(self, x, y):
        """
        Take average difference between successive x and y skeleton points,
        then compute the arc tangent from those averages.

        Parameters
        ---------------------------------------
        x : m x n float numpy array
          m is the number of skeleton points
          n is the number of frames
        y : m x n float numpy array
          (Same as x)

        Returns
        ---------------------------------------
        1-d float numpy array of length n
          The angles

        Notes
        ---------------------------------------
        Simple helper for h__computeNoseBends

        """
        # Suppress RuntimeWarning: Mean of empty slice
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            avg_diff_x = np.nanmean(np.diff(x, n=1, axis=0), axis=0)
            avg_diff_y = np.nanmean(np.diff(y, n=1, axis=0), axis=0)

        angles = np.arctan2(avg_diff_y, avg_diff_x)

        return angles

    def h__foragingData(self, fps, nose_bend_angle_d, min_win_size):
        """
        Compute the foraging amplitude and angular speed.

        Parameters
        ----------
        fps :
        nose_bend_angle_d : [n_frames x 1]
        min_win_size : (scalar)

        Returns
        ---------------------------------------
        amplitudes : [1 x n_frames]
        speeds : [1 x n_frames]

        Notes
        ---------------------------------------
        Formerly [amps,speeds] = h__foragingData(nose_bend_angle_d,
                                                 min_win_size, fps)

        """
        if min_win_size > 0:
            # Clean up the signal with a gaussian filter.
            gauss_filter = utils.gausswin(2 * min_win_size + 1) / min_win_size
            nose_bend_angle_d = filters.convolve1d(nose_bend_angle_d,
                                                   gauss_filter,
                                                   cval=0,
                                                   mode='constant')

            # Remove partial data frames ...
            nose_bend_angle_d[:min_win_size] = np.NaN
            nose_bend_angle_d[-min_win_size:] = np.NaN

        # Calculate amplitudes
        amplitudes = self.h__getAmplitudes(nose_bend_angle_d)
        assert(np.shape(nose_bend_angle_d) == np.shape(amplitudes))

        # Calculate angular speed
        # Compute the speed centered between the back and front foraging movements.
        #
        # TODO: fix the below comments to conform to 0-based indexing
        # I believe I've fixed the code already.  - @MichaelCurrie
        #  1     2    3
        # d1    d2     d1 = 2 - 1,   d2 = 3 - 2
        #     x        assign to x, avg of d1 and d2

        #???? - why multiply and not divide by fps????

        d_data = np.diff(nose_bend_angle_d) * fps
        speeds = np.empty(amplitudes.size) * np.NaN
        # This will leave the first and last frame's speed as NaN:
        speeds[1:-1] = (d_data[:-1] + d_data[1:]) / 2

        # Propagate NaN for speeds to amplitudes
        amplitudes[np.isnan(speeds)] = np.NaN

        return amplitudes, speeds

    def h__getAmplitudes(self, nose_bend_angle_d):
        """
        In between all sign changes, get the maximum or minimum value and
        apply to all indices that have the same sign within the stretch

        Parameters
        ---------------------------------------
        nose_bend_angle_d : 1-d numpy array of length n_frames

        Returns
        ---------------------------------------
        1-d numpy array of length n_frames

        Notes
        ---------------------------------------
        Formerly amps = h__getAmps(nose_bend_angle_d):

        NOTE: This code is very similar to wormKinks

        Example
        ---------------------------------------
        >>> h__getAmps(np.array[1, 2, 3, 2, 1, -1, -2, -1, 1, 2, 2, 5])
                          array[3, 3, 3, 3, 3, -2, -2, -2, 5, 5, 5, 5]
        (indentation is used here to line up the returned array for clarity)

        """
        n_frames = len(nose_bend_angle_d)

        # Suppress warnings related to finding the sign of a numpy array that
        # may contain NaN values.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data_sign = np.sign(nose_bend_angle_d)
        sign_change_I = np.flatnonzero(data_sign[1:] != data_sign[:-1])

        start_I = np.concatenate([[0], sign_change_I + 1])
        stop_I = np.concatenate([sign_change_I, [n_frames - 1]])

        # All NaN values are considered sign changes,
        # but we don't want them considered that way.
        # So create a mask of items to be removed:
        mask = np.isnan(nose_bend_angle_d[start_I])
        # Keep only those items NOT in the mask:
        start_I = start_I[np.flatnonzero(~mask)]
        stop_I = stop_I[np.flatnonzero(~mask)]

        # Python's array index notation requires that we specify one PAST the
        # index of the last entry in the "run"
        end_I = stop_I + 1

        amps = np.empty(n_frames) * np.NaN
        # For each chunk, get max or min, depending on whether the data is positive
        # or negative ...
        for i_chunk in range(len(start_I)):
            cur_start = start_I[i_chunk]
            cur_end = end_I[i_chunk]

            if nose_bend_angle_d[cur_start] > 0:
                amps[cur_start:cur_end] = max(
                    nose_bend_angle_d[cur_start:cur_end])
            else:
                amps[cur_start:cur_end] = min(
                    nose_bend_angle_d[cur_start:cur_end])

        return amps

    @classmethod
    def from_disk(cls, foraging_ref):

        self = cls.__new__(cls)

        self.amplitude = utils._extract_time_from_disk(
            foraging_ref, 'amplitude')
        self.angle_speed = utils._extract_time_from_disk(
            foraging_ref, 'angleSpeed')

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        return utils.correlation(
            self.amplitude,
            other.amplitude,
            'locomotion.foraging.amplitude') and utils.correlation(
            self.angle_speed,
            other.angle_speed,
            'locomotion.foraging.angle_speed')

#%%


#==============================================================================
#                       New Feature Organization
#==============================================================================
# self.crawling_bends = locomotion_bends.LocomotionCrawlingBends(
#                                            features_ref,
#                                            nw.angles,
#                                            self.motion_events.is_paused,
#                                            video_info.is_segmented)
#
# self.foraging_bends = locomotion_bends.LocomotionForagingBends(
#                                            features_ref,
#                                            video_info.is_segmented,
#                                            video_info.ventral_mode)

# locomotion.foraging_bends.amplitude
# locomotion.foraging_bends.angle_speed
# locomotion.crawling_bends.head.amplitude
# locomotion.crawling_bends.midbody.amplitude
# locomotion.crawling_bends.tail.amplitude
# locomotion.crawling_bends.head.frequency
# locomotion.crawling_bends.midbody.frequency
# locomotion.crawling_bends.tail.frequency

class ForagingBends(Feature):

    """
    temporary feature: locomotion.foraging_bends

    Attributes
    ----------
    amplitude
    angle_speed


    Methods
    -------
    __init__
    h__computeNoseBends
    h__computeAvgAngles
    h__interpData
    h__getNoseInterpolationIndices
    h__foragingData
    h__getAmps

    Notes
    ---------------------------------------
    Formerly +segworm/+features/@locomotion/getForaging

    Originally, part of wormBends.m

    """

    def __init__(self, wf, feature_name):
        """
        Initialize an instance of LocomotionForagingBends

        Parameters
        ----------
        nw: NormalizedWorm instance
        is_segmented_mask: boolean numpy array [1 x n_frames]
        ventral_mode: int
            0, 1, or 2 depending on the orientation of the worm.

        """

        #features_ref, is_segmented_mask, ventral_mode

        self.name = feature_name

        options = wf.options.locomotion.foraging_bends
        video_info = wf.video_info
        fps = video_info.fps
        nw = wf.nw
        ventral_mode = video_info.ventral_mode

        # TODO: Why don't we use this anymore?????
        is_segmented_mask = video_info.is_segmented

        timer = wf.timer
        timer.tic()

        # self.amplitude  = None  # DEBUG
        # self.angleSpeed = None # DEBUG

        nose_x, nose_y = \
            nw.get_partition('head_tip',
                             data_key='skeleton',
                             split_spatial_dimensions=True)

        neck_x, neck_y = \
            nw.get_partition('head_base',
                             data_key='skeleton',
                             split_spatial_dimensions=True)

        # TODO: Add "reversed" and "interpolated" options to the get_partition
        # function, to replace the below blocks of code!
        #----------------------------------------------------------------------

        # We need to flip the orientation (i.e. reverse the entries along the
        # first, or skeleton index, axis) for angles and consistency with old
        # code:
        nose_x = nose_x[::-1, :]
        nose_y = nose_y[::-1, :]
        neck_x = neck_x[::-1, :]
        neck_y = neck_y[::-1, :]

        # Step 1: Interpolation of skeleton indices
        #---------------------------------------
        # TODO: ensure that we are excluding the points at the beginning
        # and ending of the second dimension (the frames list) of nose_x, etc.
        # from being interpolated.  (this was a step in
        # h__getNoseInterpolationIndices, that we no longer have since I've
        # put the interpolation code into
        # utils.interpolate_with_threshold_2D instead.  But we
        # might be okay, since the beginning and end are going to be left alone
        # since I've set left=np.NaN and right=np.NaN in the underlying
        # utils.interpolate_with_threshold code.
        interp = utils.interpolate_with_threshold_2D

        max_samples_interp = options.max_samples_interp_nose(fps)

        nose_xi = interp(nose_x, threshold=max_samples_interp)
        nose_yi = interp(nose_y, threshold=max_samples_interp)
        neck_xi = interp(neck_x, threshold=max_samples_interp)
        neck_yi = interp(neck_y, threshold=max_samples_interp)
        #----------------------------------------------------------------------

        # Step 2: Calculation of the bend angles
        #---------------------------------------
        nose_bends = self.h__computeNoseBends(
            nose_xi, nose_yi, neck_xi, neck_yi)

        # Step 3:
        #---------------------------------------
        [nose_amps, nose_freqs] = \
            self.h__foragingData(fps, nose_bends,
                                 options.min_nose_window_samples(fps))

        if ventral_mode == 2:
            nose_amps = -nose_amps
            nose_freqs = -nose_freqs

        self.amplitude = nose_amps
        self.angle_speed = nose_freqs

        timer.toc('locomotion.foraging_bends')

    def h__computeNoseBends(self, nose_x, nose_y, neck_x, neck_y):
        """
        Compute the difference in angles between the nose and neck (really the
        head tip and head base).

        Parameters
        ----------
        nose_x: [4 x n_frames]
        nose_y: [4 x n_frames]
        neck_x: [4 x n_frames]
        neck_y: [4 x n_frames]

        Returns
        -------
        nose_bends_d

        Notes
        ---------------------------------------
        Formerly nose_bends_d = h__computeNoseBends(nose_x,nose_y,neck_x,neck_y)

        """

        nose_angles = self.h__computeAvgAngles(nose_x, nose_y)
        neck_angles = self.h__computeAvgAngles(neck_x, neck_y)

        # TODO: These three should be a method, calculating the difference
        # in angles and ensuring all results are within +/- 180
        nose_bends_d = (nose_angles - neck_angles) * (180 / np.pi)

        # Suppress warnings so we can compare a numpy array that may contain NaNs
        # without triggering a Runtime Warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nose_bends_d[nose_bends_d > 180] -= 360
            nose_bends_d[nose_bends_d < -180] += 360

        return nose_bends_d

    def h__computeAvgAngles(self, x, y):
        """
        Take average difference between successive x and y skeleton points,
        then compute the arc tangent from those averages.

        Parameters
        ----------
        x : m x n float numpy array
          m is the number of skeleton points
          n is the number of frames
        y : m x n float numpy array
          (Same as x)

        Returns
        -------
        1-d float numpy array of length n
          The angles

        Notes
        ---------------------------------------
        Simple helper for h__computeNoseBends

        """
        # Suppress RuntimeWarning: Mean of empty slice
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            avg_diff_x = np.nanmean(np.diff(x, n=1, axis=0), axis=0)
            avg_diff_y = np.nanmean(np.diff(y, n=1, axis=0), axis=0)

        angles = np.arctan2(avg_diff_y, avg_diff_x)

        return angles

    def h__foragingData(self, fps, nose_bend_angle_d, min_win_size):
        """
        Compute the foraging amplitude and angular speed.

        Parameters
        ----------
        fps :
        nose_bend_angle_d : [n_frames x 1]
        min_win_size : (scalar)

        Returns
        ---------------------------------------
        amplitudes : [1 x n_frames]
        speeds : [1 x n_frames]

        Notes
        ---------------------------------------
        Formerly [amps,speeds] = h__foragingData(nose_bend_angle_d,
                                                 min_win_size, fps)

        """
        if min_win_size > 0:
            # Clean up the signal with a gaussian filter.
            gauss_filter = utils.gausswin(2 * min_win_size + 1) / min_win_size
            nose_bend_angle_d = filters.convolve1d(nose_bend_angle_d,
                                                   gauss_filter,
                                                   cval=0,
                                                   mode='constant')

            # Remove partial data frames ...
            nose_bend_angle_d[:min_win_size] = np.NaN
            nose_bend_angle_d[-min_win_size:] = np.NaN

        # Calculate amplitudes
        amplitudes = self.h__getAmplitudes(nose_bend_angle_d)
        assert(np.shape(nose_bend_angle_d) == np.shape(amplitudes))

        # Calculate angular speed
        # Compute the speed centered between the back and front foraging movements.
        #
        #  0     1    2
        #    d1    d2     d1 = 1 - 0,   d2 = 2 - 1
        #       x        assign to x, avg of d1 and d2

        #???? - why multiply and not divide by fps????

        d_data = np.diff(nose_bend_angle_d) * fps
        speeds = np.empty(amplitudes.size) * np.NaN
        # This will leave the first and last frame's speed as NaN:
        speeds[1:-1] = (d_data[:-1] + d_data[1:]) / 2

        # Propagate NaN for speeds to amplitudes
        amplitudes[np.isnan(speeds)] = np.NaN

        return amplitudes, speeds

    def h__getAmplitudes(self, nose_bend_angle_d):
        """
        In between all sign changes, get the maximum or minimum value and
        apply to all indices that have the same sign within the stretch

        Parameters
        ---------------------------------------
        nose_bend_angle_d : 1-d numpy array of length n_frames

        Returns
        ---------------------------------------
        1-d numpy array of length n_frames

        Notes
        ---------------------------------------
        Formerly amps = h__getAmps(nose_bend_angle_d):

        NOTE: This code is very similar to wormKinks

        Example
        ---------------------------------------
        >>> h__getAmps(np.array[1, 2, 3, 2, 1, -1, -2, -1, 1, 2, 2, 5])
                          array[3, 3, 3, 3, 3, -2, -2, -2, 5, 5, 5, 5]
        (indentation is used here to line up the returned array for clarity)

        """
        n_frames = len(nose_bend_angle_d)

        # Suppress warnings related to finding the sign of a numpy array that
        # may contain NaN values.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data_sign = np.sign(nose_bend_angle_d)
        sign_change_I = np.flatnonzero(data_sign[1:] != data_sign[:-1])

        start_I = np.concatenate([[0], sign_change_I + 1])
        stop_I = np.concatenate([sign_change_I, [n_frames - 1]])

        # All NaN values are considered sign changes,
        # but we don't want them considered that way.
        # So create a mask of items to be removed:
        mask = np.isnan(nose_bend_angle_d[start_I])
        # Keep only those items NOT in the mask:
        start_I = start_I[np.flatnonzero(~mask)]
        stop_I = stop_I[np.flatnonzero(~mask)]

        # Python's array index notation requires that we specify one PAST the
        # index of the last entry in the "run"
        end_I = stop_I + 1

        amps = np.empty(n_frames) * np.NaN
        # For each chunk, get max or min, depending on whether the data is positive
        # or negative ...
        for i_chunk in range(len(start_I)):
            cur_start = start_I[i_chunk]
            cur_end = end_I[i_chunk]

            if nose_bend_angle_d[cur_start] > 0:
                amps[cur_start:cur_end] = max(
                    nose_bend_angle_d[cur_start:cur_end])
            else:
                amps[cur_start:cur_end] = min(
                    nose_bend_angle_d[cur_start:cur_end])

        return amps

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.amplitude = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'bends', 'foraging', 'amplitude'])
        self.angle_speed = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'bends', 'foraging', 'angleSpeed'])

        return self

#    def __eq__(self, other):
#        return utils.correlation(self.amplitude, other.amplitude, 'locomotion.foraging.amplitude') and \
#             utils.correlation(self.angle_speed, other.angle_speed, 'locomotion.foraging.angle_speed')


class ForagingAmplitude(Feature):

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'locomotion.foraging_bends').amplitude

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class ForagingAngleSpeed(Feature):

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'locomotion.foraging_bends').angle_speed

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class CrawlingBend(Feature, BendHelper):

    """
    Locomotion Crawling Bends Feature.

    Notes
    ---------------------------------------
    Formerly +segworm/+features/@locomotion/getLocomotionBends

    Originally, part of wormBends.m


    Note from Ev Yemini on Setup Options
    ---------------------------------------
    Empirically I've found the values below achieve good signal.

    Furthermore:
    The body bend frequency is much easier to see (than foraging). The N2
    signal is clearly centered around 1/3Hz in both the literature and
    through visual inspection.

    I chose a high-frequency threshold of 4 frames. With 4 frames a 3-frame
    tick, resulting from segmentation noise, will be diluted by the
    additional frame.

    TODO: Move this to a different location and reference in the code
    Nature Methods Description
    ---------------------------------------

    Worm crawling is expressed as both an amplitude and frequency
    (Supplementary Fig. 4e). We measure these features instantaneously at
    the head, midbody, and tail. The amplitude and frequency are signed
    negatively whenever the worm’s ventral side is contained within the
    concave portion of its instantaneous bend.

    Crawling is only measured during forward and backward motion states.
    The worm bend mean angles (described in the section on “Posture”) show
    a roughly periodic signal as the crawling wave travels along the worm’s
    body. This wave can be asymmetric due to differences in dorsal-ventral
    flexibility or simply because the worm is executing a turn. Moreover
    the wave dynamics can change abruptly to speed up or slow down.
    Therefore, the signal is only roughly periodic and we measure its
    instantaneous properties.

    Worm bends are linearly interpolated across unsegmented frames. The
    motion states criteria (described earlier in this section) guarantee
    that interpolation is no more than 1/4 of a second long. For each
    frame, we search both backwards and forwards for a zero crossing in the
    bend angle mean – the location where the measured body part (head,
    midbody, or tail) must have hit a flat posture (a supplementary bend
    angle of 0°). This guarantees that we are observing half a cycle for
    the waveform. Crawling is bounded between 1/30Hz (a very slow wave that
    would not resemble crawling) and 1Hz (an impossibly fast wave on agar).

    If the window between zero crossings is too small, the nearest zero
    crossing is assumed to be noise and we search for the next available
    zero crossing in its respective direction. If the window is too big,
    crawling is marked undefined at the frame.

    Once an appropriate window has been found, the window is extended in
    order to center the frame and measure instantaneous crawling by
    ensuring that the distance on either side to respective zero crossings
    is identical. If the distances are not identical, the distance of the
    larger side is used in place of the zero-crossing distance of the
    smaller side in order to expand the small side and achieve a symmetric
    window, centered at the frame of interest.

    We use a Fourier transform to measure the amplitude and frequency
    within the window described above. The largest peak within the
    transform is chosen for the crawling amplitude and frequency. If the
    troughs on either side of the peak exceed 1/2 its height, the peak is
    rejected for being unclear and crawling is marked as undefined at the
    frame. Similarly, if the integral between the troughs is less than half
    the total integral, the peak is rejected for being weak.


    """

    #bend_names = ['head', 'midbody', 'tail']

    def __init__(self, wf, feature_name, bend_name):
        """
        Compute the temporal bending frequency at the head, midbody, and tail.

        Parameters:
        -----------
        features_ref :
        bend_angles : numpy.array
            - [49 x n_frames]
        is_paused : numpy.array
            - [1 x n_frames]
            Whether or not the worm is considered to be paused during the frame
        is_segmented_mask : [1 x n_frames]

        """

        #features_ref, bend_angles, is_paused, is_segmented_mask

        self.name = feature_name

        options = wf.options.locomotion.crawling_bends
        video_info = wf.video_info
        fps = video_info.fps
        is_segmented_mask = video_info.is_segmented
        is_paused = self.get_feature(
            wf, 'locomotion.motion_events.is_paused').value
        bend_angles = wf.nw.angles
        
        timer = wf.timer
        timer.tic()

        # Special Case: No worm data.
        #------------------------------------
        if ~np.any(is_segmented_mask):
            self.amplitude = None
            self.frequency = None
            return

        # Find the mean bend angle for the current partition, across all frames
        s = slice(*options.bends_partitions[bend_name])

        # Suppress RuntimeWarning: Mean of empty slice
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            avg_bend_angles = np.nanmean(bend_angles[s, :], axis=0)

        # interpolate if:
        # not all are segmented - if all present we don't need to interpolate
        # some are segmented - segmented data provides basis for interpolation
        if not(np.all(is_segmented_mask)) and np.any(is_segmented_mask):
            avg_bend_angles = utils.interpolate_with_threshold(avg_bend_angles)

        bound_info = CrawlingBendsBoundInfo(
            avg_bend_angles, is_paused, options, fps)

        [amplitude, frequency] = self.h__getBendData(avg_bend_angles,
                                                     bound_info,
                                                     options,
                                                     fps)

        self.amplitude = amplitude
        self.frequency = frequency
        # setattr(self,cur_partition_name,LocomotionBend(amplitude,frequency,cur_partition_name))

        timer.toc('locomotion.crawling_bends')

#    def __eq__(self, other):
#
#        #merge_nans=True - utils.separated_peaks works slightly differently
#        #so merge_nans is true here, for now. The utils function works correctly
#        #and the old version works incorrectly but was convoluted enough that
#        #it was hard to replicate
#
#        return utils.correlation(self.value, other.v,
#                                  'locomotion.bends.' + self.name + '.amplitude',
#                                  merge_nans=True) and \
#             utils.correlation(self.frequency, other.frequency,
#                                'locomotion.bends.' + self.name + '.frequency',
#                                merge_nans=True)


    @classmethod
    def from_schafer_file(cls, wf, feature_name, bend_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.amplitude = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'bends', bend_name, 'amplitude'])
        self.frequency = utils.get_nested_h5_field(
            wf.h, ['locomotion', 'bends', bend_name, 'frequency'])

        return self


class CrawlingBendsBoundInfo(object):

    """
    This class is used by LocomotionCrawlingBends.

    Attributes
    ----------
    back_zeros_I :
    front_zeros_I :
    left_bounds :
    right_bounds :
    half_distances :


    """

    def __init__(self, avg_bend_angles, is_paused, options, fps):

        # TODO: This needs to be cleaned up ...  - @JimHokanson
        min_number_frames_for_bend = round(options.min_time_for_bend * fps)
        max_number_frames_for_bend = round(options.max_time_for_bend * fps)

        [back_zeros_I, front_zeros_I] = \
            self.h__getBoundingZeroIndices(avg_bend_angles,
                                           min_number_frames_for_bend)

        n_frames = len(avg_bend_angles)

        # Go left and right, and get the
        left_distances = np.array(range(n_frames)) - back_zeros_I
        right_distances = front_zeros_I - np.array(range(n_frames))
        half_distances = np.maximum(left_distances, right_distances)

        left_bounds = np.array(range(n_frames)) - half_distances

        #+1 for slicing to be inclusive of the right bound
        right_bounds = np.array(range(n_frames)) + half_distances + 1

        self.back_zeros_I = back_zeros_I
        self.front_zeros_I = front_zeros_I
        self.left_bounds = left_bounds
        self.right_bounds = right_bounds
        self.half_distances = half_distances

        # Compute conditions by which we will ignore frames:
        # -------------------------------------------------
        # 1) frame is not bounded on both sides by a sign change
        #- avg_bend_angles is NaN, this will only happen on the edges because we
        #    interpolate over the other frames ... (we just don't extrapolate)
        #- the sign change region is too large
        #- the bounds we settle on exceed the data region
        #- mode segmentation determined the frame was a paused frame
        #
        #
        #??? - what about large NaN regions, are those paused regions???

        # MRC code placed restriction on half distance, not on the full distance
        # This is still left in place below
        # Should be 2*half_distances > max_number_frames_for_bend

        self.is_bad_mask  = \
            (back_zeros_I  == -1) | \
            (front_zeros_I == -1) | \
            np.isnan(avg_bend_angles) | \
            (half_distances > max_number_frames_for_bend) | \
            (left_bounds < 0) | \
            (right_bounds > n_frames) | \
            is_paused

    def h__getBoundingZeroIndices(
            self,
            avg_bend_angles,
            min_number_frames_for_bend):
        """
        The goal of this function is to bound each index of avg_bend_angles by
        sign changes.


        Parameters:
        -----------
        avg_bend_angles : [1 x n_frames]
        min_number_frames_for_bend    : int
          The minimum size of the data window

        Returns
        ----------------------
        back_zeros_I    : [1 x n_frames]
          For each frame, this specifies a preceding frame in which a
          change in the bend angle occurs. Invalid entries are
          indicated by -1.
        front_zeros_I   : [1 x n_frames]

        Notes
        ----------------------
        Formerly [back_zeros_I,front_zeros_I] = \
                h__getBoundingZeroIndices(avg_bend_angles,min_number_frames_for_bend)

        """

        # Getting sign change indices ...
        # ---------------------------------------
        # The old code found sign changes for every frame, even though
        # the sign changes never changed. Instead we find all sign changes,
        # and then for each frame know which frame to the left and right
        # have sign changes. We do this in such a way so that if we need to
        # look further to the left or right, it is really easy to get the
        # next answer. In other words, if we are bounded by the 3rd and 4th sign
        # change, and we are going towards the 3rd sign change, then if the
        # 3rd sign change doesn't work, we can go to the 2nd sign change index,
        # not by searching the data array, but by getting the 2nd element of
        # the sign change index array.

        with np.errstate(invalid='ignore'):
            sign_change_mask = np.sign(avg_bend_angles[:-1]) != \
                np.sign(avg_bend_angles[1:])

        sign_change_I = np.flatnonzero(sign_change_mask)
        n_sign_changes = len(sign_change_I)
        n_frames = len(avg_bend_angles)

        if n_sign_changes == 0:
            # no changes of sign return two zeros arrays
            return [np.zeros(n_frames), np.zeros(n_frames)]

        """
        To get the correct frame numbers, we need to do the following
        depending on whether or not the bound is the left (backward)
        bound or the right (forward) bound.

        Note from @JimHokanson: I haven't really thought through why
        this is, but it mimics the old code.

        for left bounds   - at sign changes - don't subtract or add
        for right bounds  - we need to add 1


        Let's say we have sign changes at indices 3  6  9
        What we need ...
                1 2 3 4 5 6 7  9  10  Indices
        Left  = 0 0 0 3 3 3 6  6  6   - at 4, the left sign change is at 3
        Right = 4 4 4 7 7 7 10 10 0   - at 4, the right sign change is at 7

        NOTE: The values above are the final indices or values, but instead we
        want to work with the indices, so we need:

                1 2 3 4 5 6 7  9  10  Indices
        Left  = 0 0 0 1 1 1 2  2  2 - left_sign_change_I
        Right = 1 1 1 2 2 2 3  3  3 - right_sign_change_I

        we also need:
        left_values  = [3 6 9]  #the sign change indices
        right_values = [4 7 10] #+1

        So this says:
        left_sign_change_I(7) => 2
        left_values(2) => 6, our sign change is at 6

        Let's say we need to expand further to the left, then we take
        left_sign_change_I(7) - 1 => 1
        left_values(1) => 3, our new sign change is at 3

        Further:
        left_sign_change_I(7) - 2 => 0
        We've gone too far, nothing at index 0, set to invalid
        """

        # For each element, determine the indices to the left and right of the
        # element at which a sign change occurs.

        BAD_INDEX_VALUE = -1

        # For each element in the array, these values indicate which
        # sign change index to use ...
        left_sign_change_I = np.zeros(n_frames)

        left_sign_change_I[sign_change_I + 1] = 1
        # We increment at values to the right of the sign changes
        left_sign_change_I = left_sign_change_I.cumsum() - 1

        # NOTE: We need to do this after the cumsum :/
        left_sign_change_I[:sign_change_I[0]] = BAD_INDEX_VALUE
        # The previous line is a little Matlab trick in which
        # something like:
        # 0  1 0 1 0 0 1 0 0 <= sign change indices
        # 0  1 2 3 4 5 6 7 8 <= indices
        # becomes:
        # -1 0 0 1 1 1 2 2 2 <= -1 is off limits, values are inclusive
        #
        # so now at each frame, we get the index of the value that
        # is to the left.
        #
        # From above; sign_change_I = [0 3 6]
        #
        # So at index 5, the next sign change is at sign_change_I[left_change_I[5]]
        # or sign_change_I[1] => 3

        # This does:
        # 0  1 0 1 0 0 1  0  0 <= sign change indices
        # 0  1 2 3 4 5 6  7  8 <= indices
        # 0  0 1 1 2 2 2 -1 -1 <= indices of sign change to right
        right_sign_change_I = np.zeros(n_frames)
        right_sign_change_I[sign_change_I[:-1] + 1] = 1
        right_sign_change_I[0] = 1
        right_sign_change_I = right_sign_change_I.cumsum() - 1
        # We must have nothing to the right of the last change:
        right_sign_change_I[sign_change_I[-1] + 1:] = BAD_INDEX_VALUE

        # Indices that each left_sign_change_I or right_sign_change_I points to
        left_values = sign_change_I
        right_values = sign_change_I + 1  # By definition
        #----------------------------------------------------------------

        back_zeros_I = np.zeros(n_frames)
        back_zeros_I[:] = BAD_INDEX_VALUE
        front_zeros_I = np.zeros(n_frames)
        back_zeros_I[:] = BAD_INDEX_VALUE

        for iFrame in range(n_frames):
            cur_left_index = left_sign_change_I[iFrame]
            cur_right_index = right_sign_change_I[iFrame]

            if cur_left_index == BAD_INDEX_VALUE or cur_right_index == BAD_INDEX_VALUE:
                continue

            # Convert from float to int
            cur_left_index = int(cur_left_index)
            cur_right_index = int(cur_right_index)

            back_zero_I = left_values[cur_left_index]
            front_zero_I = right_values[cur_right_index]

            use_values = True

            # Expand the zero-crossing window.
            #----------------------------------
            # Note from @JimHokanson:
            #
            # TODO: Fix and move this code to old config
            #
            # General problem, we specify a minimum acceptable window size,
            # and the old code needlessly expands the window past this point
            # by doing the following comparison:
            #
            # - distance from right to left > min_window_size?
            #
            #   The following code centers on 2x the larger of the following gaps:
            #
            #   - distance from left to center
            #   - distance from right to center
            #
            #   So we should check if either of these is half ot the
            #   required width.
            #
            # half-window sizes:
            # left_window_size  = iFrame - back_zero_I
            # right_window_size = front_zero_I - iFrame
            #
            # so in reality we should use:
            #
            # front_zero_I - iFrame < min_number_frames_for_bend/2 and
            # iFrame - back_zero_I < min_number_frames_for_bend/2
            #
            # By not doing this, we overshoot the minimum window size that
            # we need to use. Consider window sizes that are in terms of
            # the minimum window size.
            #
            # i.e. 0.5w means the left or right window is half min_number_frames_for_bend
            #
            # Consider we have:
            # 0.5w left
            # 0.3w right
            #
            #   total 0.8w => not at 1w, thus old code should expand
            #
            #   But in reality, if we stopped now we would be at twice 0.5w

            while (
                    front_zero_I -
                    back_zero_I +
                    1) < min_number_frames_for_bend:
                # Expand the smaller of the two windows
                # -------------------------------------
                #  left_window_size       right_window_size
                if (iFrame - back_zero_I) < (front_zero_I - iFrame):
                    # Expand to the left:
                    cur_left_index = cur_left_index - 1
                    if cur_left_index == BAD_INDEX_VALUE:
                        use_values = False
                        break
                    back_zero_I = left_values[cur_left_index]
                else:
                    # Expand to the right:
                    cur_right_index = cur_right_index + 1
                    if cur_right_index >= n_sign_changes:
                        use_values = False
                        break
                    front_zero_I = right_values[cur_right_index]

            if use_values:
                back_zeros_I[iFrame] = back_zero_I
                front_zeros_I[iFrame] = front_zero_I

        return [back_zeros_I, front_zeros_I]

    def __repr__(self):
        return utils.print_object(self)


class BendAmplitude(Feature):
    """
    Feature: locomotion.crawling_bends.[bend_name].amplitude
    """

    def __init__(self, wf, feature_name, bend_name):
        parent_name = generic_features.get_parent_feature_name(feature_name)
        self.name = feature_name
        self.value = self.get_feature(wf, parent_name).amplitude

    @classmethod
    def from_schafer_file(cls, wf, feature_name, bend_name):
        return cls(wf, feature_name, bend_name)

    def __eq__(self, other):

        return utils.correlation(self.value, other.value,
                                 self.name,
                                 merge_nans=True)


class BendFrequency(Feature):
    """
    Feature: locomotion.crawling_bends.[bend_name].frequency
    """

    def __init__(self, wf, feature_name, bend_name):
        parent_name = generic_features.get_parent_feature_name(feature_name)
        self.name = feature_name
        self.value = self.get_feature(wf, parent_name).frequency

    @classmethod
    def from_schafer_file(cls, wf, feature_name, bend_name):
        return cls(wf, feature_name, bend_name)

    def __eq__(self, other):

        return utils.correlation(self.value, other.value,
                                 self.name,
                                 merge_nans=True)

