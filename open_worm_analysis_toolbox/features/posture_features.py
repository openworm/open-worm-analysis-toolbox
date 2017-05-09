# -*- coding: utf-8 -*-
"""
Posture features

"""


from __future__ import division

import scipy.ndimage.filters as filters
import numpy as np
import warnings
import os
import h5py

import cv2

from . import generic_features
from .generic_features import Feature
from .. import config, utils
from . import events


class Bends(object):
    """
    Posture Bends

    Attributes
    ----------
    head : BendSection
    midbody : BendSection
    tail : BendSection
    hips : BendSection
    neck : BendSection

    """

    def __init__(self, features_ref):
        nw = features_ref.nw

        p = nw.get_partition_subset('normal')

        self.posture_bend_keys = p.keys()

        for partition_key in self.posture_bend_keys:
            # Retrieve the part of the worm we are currently looking at:
            bend_angles = nw.get_partition(partition_key, 'angles')

            # shape = (n):

            # Suppress RuntimeWarning: Mean of empty slice for those frames
            # that are ALL NaN.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)

                # Throws warning on taking the mean of an empty slice
                temp_mean = np.nanmean(a=bend_angles, axis=0)

                # Throws warning when degrees of freedom <= 0 for slice
                temp_std = np.nanstd(a=bend_angles, axis=0)

                # Sign the standard deviation (to provide the bend's
                # dorsal/ventral orientation)
                temp_std[temp_mean < 0] *= -1





            setattr(self, partition_key,
                    BendSection(temp_mean, temp_std, partition_key))

    @classmethod
    def create(self, features_ref):
        options = features_ref.options

        # TODO: this should be populated by calling
        # WormPartition.get_partition_subset('normal'), and
        # get_partition_subset should be an @classmethod.
        self.posture_bend_keys = ['head', 'midbody', 'tail', 'hips', 'neck']

        if options.should_compute_feature('locomotion.bends', features_ref):
            return Bends(features_ref)
        else:
            return None

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        is_equal = True
        for partition_key in self.posture_bend_keys:
            is_equal = is_equal and (getattr(self, partition_key) ==
                                     getattr(other, partition_key))

        return is_equal

    @classmethod
    def from_disk(cls, saved_bend_data):

        self = cls.__new__(cls)

        for partition_key in saved_bend_data.keys():
            setattr(self, partition_key,
                    BendSection.from_disk(saved_bend_data[partition_key],
                                          partition_key))

        return self


class BendSection(object):

    """
    Attributes
    ----------

    See Also
    --------
    Bends

    """

    def __init__(self, mean, std_dev, name):
        self.mean = mean
        self.std_dev = std_dev
        self.name = name

    @classmethod
    def from_disk(cls, saved_bend_data, name):

        self = cls.__new__(cls)

        self.mean = utils._extract_time_from_disk(saved_bend_data, 'mean')

        try:
            self.std_dev = \
                utils._extract_time_from_disk(saved_bend_data, 'std_dev')
        except KeyError:
            self.std_dev = \
                utils._extract_time_from_disk(saved_bend_data, 'stdDev')

        self.name = name

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        # TODO: Why is the head.std_dev so low???
        # Are we not mimicing some old error properly???
        return utils.correlation(self.mean, other.mean,
                                 'posture.bends.' + self.name + '.mean',
                                 high_corr_value=0.95) and \
            utils.correlation(self.std_dev, other.std_dev,
                              'posture.bends.' + self.name + '.std_dev',
                              high_corr_value=0.60)



def get_worm_kinks(features_ref):
    """
    Parameters
    ----------
    features_ref : open-worm-analysis-toolbox.features.worm_features.WormFeatures

    Returns
    -------
    numpy.array

    """

    nw = features_ref.nw
    timer = features_ref.timer
    timer.tic()

    options = features_ref.options

    KINK_LENGTH_THRESHOLD_PCT = options.posture.kink_length_threshold_pct

    bend_angles = nw.angles

    # Determine the bend segment length threshold.
    n_angles = bend_angles.shape[0]
    length_threshold = np.round(n_angles * KINK_LENGTH_THRESHOLD_PCT)

    # Compute a gaussian filter for the angles.
    #--------------------------------------------------------------------------
    # JAH NOTE: This is a nice way of getting the appropriate odd value
    # unlike the other code with so many if statements ...
    #- see window code which tries to get an odd value ...
    #- I'd like to go back and fix that code ...
    half_length_thr = np.round(length_threshold / 2)
    gauss_filter = utils.gausswin(half_length_thr * 2 + 1) / half_length_thr

    # Compute the kinks for the worms.
    n_frames = bend_angles.shape[1]
    n_kinks_all = np.zeros(n_frames, dtype=float)
    n_kinks_all[:] = np.NaN

    #(np.any(np.logical_or(mask_pos,mask_neg),axis=0)).nonzero()[0]

    nan_mask = np.isnan(bend_angles)

    for iFrame in (~np.all(nan_mask, axis=0)).nonzero()[0]:
        smoothed_bend_angles = filters.convolve1d(
            bend_angles[:, iFrame], gauss_filter, cval=0, mode='constant')

        # This code is nearly identical in getForaging
        #-------------------------------------------------------
        n_frames = smoothed_bend_angles.shape[0]

        with np.errstate(invalid='ignore'):
            dataSign = np.sign(smoothed_bend_angles)

        if np.any(np.equal(dataSign, 0)):
            # I don't expect that we'll ever actually reach 0
            # The code for zero was a bit weird, it keeps counting if no sign
            # change i.e. + + + 0 + + + => all +
            #
            # but it counts for both if sign change
            # + + 0 - - - => 3 +s and 4 -s

            # I had to change this to a warning and returning NaNs
            # to get my corner case unit tests working, i.e. the case
            # of a perfectly straight worm.  - @MichaelCurrie
            n_kinks_all[:] = np.NaN
            #raise Warning("Unhandled code case")
            return n_kinks_all

        sign_change_I = (
            np.not_equal(dataSign[1:], dataSign[0:-1])).nonzero()[0]

        end_I = np.concatenate(
            (sign_change_I,
             n_frames * np.ones(1, dtype=np.result_type(sign_change_I))))

        wtf1 = np.zeros(1, dtype=np.result_type(sign_change_I))
        wtf2 = sign_change_I + 1
        start_I = np.concatenate((wtf1, wtf2))  # +2? due to inclusion rules???

        # All NaN values are considered sign changes, remove these ...
        keep_mask = np.logical_not(np.isnan(smoothed_bend_angles[start_I]))

        start_I = start_I[keep_mask]
        end_I = end_I[keep_mask]

        # The old code had a provision for having NaN values in the middle
        # of the worm. I have not translated that feature to the newer code. I
        # don't think it will ever happen though for a valid frame, only on the
        # edges should you have NaN values.
        if start_I.size != 0 and \
           np.any(np.isnan(smoothed_bend_angles[start_I[0]:end_I[-1]])):
            raise Exception("Unhandled code case")

        #-------------------------------------------------------
        # End of identical code ...

        lengths = end_I - start_I + 1

        # Adjust lengths for first and last:
        # Basically we allow NaN values to count towards the length for the
        # first and last stretches
        if lengths.size != 0:
            if start_I[0] != 0:  # Due to leading NaNs
                lengths[0] = end_I[0] + 1
            if end_I[-1] != n_frames:  # Due to trailing NaNs
                lengths[-1] = n_frames - start_I[-1]

        n_kinks_all[iFrame] = np.sum(lengths >= length_threshold)

    timer.toc('posture.kinks')

    return n_kinks_all


def get_worm_coils(features_ref, midbody_distance):
    """
    Get the worm's posture.coils.

    Parameters
    ----------
    features_ref : open-worm-analysis-toolbox.features.worm_features.WormFeatures

    This function is currently very reliant on the MRC processor.

    Translated From:
    https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
    %2Bseg_worm/%2Bfeatures/%40posture/getCoils.m

    """
    options = features_ref.options
    posture_options = options.posture
    timer = features_ref.timer

    fps = features_ref.video_info.fps

    timer.tic()

    frame_code = features_ref.video_info.frame_code

    COIL_FRAME_THRESHOLD = posture_options.coiling_frame_threshold(fps)

    # These are values that are specific to the MRC processor
    COIL_START_CODES = [105, 106]
    # Code that indicates a frame was successfully segmented
    FRAME_SEGMENTED = 1

    # Algorithm: Whenever a new start is found, find the
    # first segmented frame; that's the end.

    # Add on a frame to allow closing a coil at the end ...
    coil_start_mask = (frame_code == COIL_START_CODES[0]) | (
        frame_code == COIL_START_CODES[1])
    np_false = np.zeros((1,), dtype=bool)
    coil_start_mask = np.concatenate((coil_start_mask, np_false))

    # NOTE: These are not guaranteed ends, just possible ends ...
    end_coil_mask = frame_code == FRAME_SEGMENTED
    np_true = ~np_false
    end_coil_mask = np.concatenate((end_coil_mask, np_true))

    in_coil = False
    coil_frame_start = -1
    n_coils = 0
    n_frames_plus1 = len(frame_code) + 1

    starts = []
    ends = []

    for iFrame in range(n_frames_plus1):
        if in_coil:
            if end_coil_mask[iFrame]:
                n_coil_frames = iFrame - coil_frame_start
                if n_coil_frames >= COIL_FRAME_THRESHOLD:
                    n_coils += 1

                    starts.append(coil_frame_start)
                    ends.append(iFrame - 1)

                in_coil = False
        elif coil_start_mask[iFrame]:
            in_coil = True
            coil_frame_start = iFrame

    if options.mimic_old_behaviour:
        if (len(starts) > 0) and (ends[-1] == len(frame_code) - 1):
            ends[-1] += -1
            starts[-1] += -1

    temp = events.EventList(np.transpose(np.vstack((starts, ends))))

    timer.toc('posture.coils')

    return events.EventListWithFeatures(fps, temp, midbody_distance)

#=====================================================================
#                           New Features
#=====================================================================


class EccentricityAndOrientationProcessor(Feature):

    """
    Temporary Feature: posture.eccentricity_and_orientation

    Attributes
    ----------
    eccentricity
    orientation
    """

    def __init__(self, wf, feature_name):
        """
        Get the eccentricity and orientation of a contour using the moments

        http://en.wikipedia.org/wiki/Image_moment

        Calculated by opencv moments():
        http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        
        
        This code might not work if there are redundant points in the contour (green approximation fails if the).
        
        If there are not contours the code will use the minimal rectangular area.
        http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=minarearect#minarearect
        The function moments only work on a close non overlaping contour.
        The box width and length are used instead of the ellipse minor and major axis
        to get an estimate of the eccentricity.
        """
        
        def _cnt_momentum(cnt):
            moments = cv2.moments(cnt)
            return moments['mu11'], moments['mu20'], moments['mu02']
            
        def _skel_momentum(skel):
            mat_cov = np.cov(skel.T)
            mu20 = mat_cov[0,0]
            mu02 = mat_cov[1,1]
            mu11 = mat_cov[0,1]
            return mu11, mu20, mu02
            
            
        def _momentum_eccentricty_orientation(mu11, mu20, mu02):
            a1 = (mu20 + mu02) / 2
            a2 = np.sqrt(4 * mu11**2 +
                         (mu20 - mu02)**2) / 2

            minor_axis = a1 - a2
            major_axis = a1 + a2
            
            ecc = np.sqrt(1 - minor_axis / major_axis)
            ang = np.arctan2(2 * mu11, (mu20 - mu02)) / 2
            ang *= 180 / np.pi
            return ecc, ang

        def _box_eccentricity_orientation(skel):
            (CMx, CMy), (L, W), angle = cv2.minAreaRect(skel)
            if W > L:
                L, W = W, L  # switch if width is larger than length
                angle += 90 # this means that the angle is shifted too
            quirkiness = np.sqrt(1 - W**2 / L**2)
            return quirkiness, angle

        self.name = feature_name

        wf.timer.tic()

        #Try to use the contour, otherwise use the skeleton
        try:
            points = wf.nw.contour_without_redundant_points
            _get_momentum = _cnt_momentum
            
        except:
            points = wf.nw.skeleton
            _get_momentum = _skel_momentum
        
        
        
        # OpenCV does not like float64, this actually make sense for image
        # data where we do not require a large precition in the decimal part.
        # This could save quite a lot of space
        points = points.astype(np.float32)
        
        tot = points.shape[-1]

        eccentricity = np.full(tot, np.nan)
        orientation = np.full(tot, np.nan)
        for ii in range(tot):
            frame_points = points[:, :, ii]
            
            if ~np.any(np.isnan(frame_points)):
                mu11, mu20, mu02 = _get_momentum(frame_points)
                eccentricity[ii], orientation[ii] = \
                _momentum_eccentricty_orientation(mu11, mu20, mu02)
                
        wf.timer.toc(self.name)

        self.eccentricity = eccentricity
        self.orientation = orientation

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.eccentricity = utils.get_nested_h5_field(
            wf.h, ['posture', 'eccentricity'])

        # This isn't saved to disk
        self.orientation = None
        return self


class Eccentricity(Feature):
    """
    Feature: 'posture.eccentricity'
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'posture.eccentricity_and_orientation').eccentricity

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class AmplitudeAndWavelengthProcessor(Feature):

    """
    Temporary Feature: posture.amplitude_wavelength_processor

    Attributes
    ----------
    amplitude_max
    amplitude_ratio
    primary_wavelength
    secondary_wavelength
    track_length
    """

    def __init__(self, wf, feature_name):
        """
        Calculates amplitude of rotated worm (relies on orientation
        aka theta_d)

        Parameters
        ----------
        theta_d
        sx
        sy
        worm_lengths

        """

        self.name = feature_name
        theta_d = self.get_feature(
            wf, 'posture.eccentricity_and_orientation').orientation

        timer = wf.timer
        timer.tic()

        options = wf.options

        nw = wf.nw
        sx = nw.skeleton_x
        sy = nw.skeleton_y
        worm_lengths = nw.length

        # TODO: Move these into posture options

        wave_options = wf.options.posture.wavelength

        # https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/
        # %2Bseg_worm/%2Bfeatures/%40posture/getAmplitudeAndWavelength.m
        N_POINTS_FFT = wave_options.n_points_fft
        HALF_N_FFT = int(N_POINTS_FFT / 2)
        MIN_DIST_PEAKS = wave_options.min_dist_peaks
        WAVELENGTH_PCT_MAX_CUTOFF = wave_options.pct_max_cutoff
        WAVELENGTH_PCT_CUTOFF = wave_options.pct_cutoff

        assert sx.shape[0] <= N_POINTS_FFT # of points used in the FFT
        # must be more than the number of points in the skeleton

        # Rotate the worm so that it lies primarily along a single axis
        #-------------------------------------------------------------
        theta_r = theta_d * (np.pi / 180)
        wwx = sx * np.cos(theta_r) + sy * np.sin(theta_r)
        wwy = sx * -np.sin(theta_r) + sy * np.cos(theta_r)

        # Subtract mean
        #-----------------------------------------------------------------
        #??? - Why isn't this done before the rotation?
        wwx = wwx - np.mean(wwx, axis=0)
        wwy = wwy - np.mean(wwy, axis=0)

        # Calculate track amplitude
        #-----------------------------------------------------------------
        amp1 = np.amax(wwy, axis=0)
        amp2 = np.amin(wwy, axis=0)
        amplitude_max = amp1 - amp2
        amp2 = np.abs(amp2)

        # Ignore NaN division warnings
        with np.errstate(invalid='ignore'):
            amplitude_ratio = np.divide(np.minimum(amp1, amp2),
                                        np.maximum(amp1, amp2))

        # Calculate track length
        #-----------------------------------------------------------------
        # This is the x distance after rotation, and is different from the
        # worm length which follows the skeleton. This will always be smaller
        # than the worm length. If the worm were perfectly straight these
        # values would be the same.
        track_length = np.amax(wwx, axis=0) - np.amin(wwx, axis=0)

        # Wavelength calculation
        #-----------------------------------------------------------------
        dwwx = np.diff(wwx, 1, axis=0)
        
        # Does the sign change? This is a check to make sure that the
        # change in x is always going one way or the other. Is sign of all
        # differences the same as the sign of the first, or rather, are any
        # of the signs not the same as the first sign, indicating a "bad
        # worm orientation".
        #
        # NOT: This means that within a frame, if the worm x direction
        #      changes, then it is considered a bad worm and is not
        #      evaluated for wavelength
        #

        with np.errstate(invalid='ignore'):
            bad_worm_orientation = np.any(
                np.not_equal(np.sign(dwwx), np.sign(dwwx[0, :])), axis=0)

        n_frames = bad_worm_orientation.size
        primary_wavelength = np.full(n_frames, np.nan)
        secondary_wavelength = np.full(n_frames, np.nan)

        # NOTE: Right now this varies from worm to worm which means the
        # spectral resolution varies as well from worm to worm
        spatial_sampling_frequency = (wwx.shape[0] - 1) / track_length

        ds = 1 / spatial_sampling_frequency

        frames_to_calculate = \
            (np.logical_not(bad_worm_orientation)).nonzero()[0]

        for cur_frame in frames_to_calculate:
            # Create an evenly sampled x-axis, note that ds varies
            xx = wwx[:, cur_frame]
            yy = wwy[:, cur_frame]
            if xx[0] > xx[-1]: #switch we want to have monotonically inceasing values
                xx = xx[::-1]
                yy = yy[::-1]
            
            iwwx = utils.colon(xx[0], ds[cur_frame], xx[-1])
            iwwy = np.interp(iwwx, xx, yy)
            iwwy = iwwy[::-1]
            
            temp = np.fft.fft(iwwy, N_POINTS_FFT)


            if options.mimic_old_behaviour:
                iY = temp[0:HALF_N_FFT]
                iY = iY * np.conjugate(iY) / N_POINTS_FFT
            else:
                iY = np.abs(temp[0:HALF_N_FFT])
            
            
            # Find peaks that are greater than the cutoff
            peaks, indx = utils.separated_peaks(iY,
                                                MIN_DIST_PEAKS,
                                                True,
                                                (WAVELENGTH_PCT_MAX_CUTOFF *
                                                 np.amax(iY)))
            
            # This is what the supplemental says, not what was done in
            # the previous code. I'm not sure what was done for the actual
            # paper, but I would guess they used power.
            #
            # This gets used when determining the secondary wavelength, as
            # it must be greater than half the maximum to be considered a
            # secondary wavelength.

            # NOTE: True Amplitude = 2*abs(fft)/
            #                    (length_real_data i.e. 48 or 49, not 512)
            #
            # i.e. for a sinusoid of a given amplitude, the above formula
            # would give you the amplitude of the sinusoid

            # We sort the peaks so that the largest is at the first index
            # and will be primary, this was not done in the previous
            # version of the code
            I = np.argsort(-1 * peaks)
            indx = indx[I]

            frequency_values = (indx - 1) / N_POINTS_FFT * \
                spatial_sampling_frequency[cur_frame]

            all_wavelengths = 1 / frequency_values

            p_temp = all_wavelengths[0]

            if indx.size > 1:
                s_temp = all_wavelengths[1]
            else:
                s_temp = np.NaN

            worm_wavelength_max = (WAVELENGTH_PCT_CUTOFF *
                                   worm_lengths[cur_frame])

            # Cap wavelengths ...
            if p_temp > worm_wavelength_max:
                p_temp = worm_wavelength_max

            # ??? Do we really want to keep this as well if p_temp == worm_2x?
            # i.e., should the secondary wavelength be valid if the primary is
            # also limited in this way ?????
            if s_temp > worm_wavelength_max:
                s_temp = worm_wavelength_max

            primary_wavelength[cur_frame] = p_temp
            secondary_wavelength[cur_frame] = s_temp

        if options.mimic_old_behaviour:
            # In the old code, the first peak (i.e. larger wavelength,
            # lower frequency) was always the primary wavelength, where as
            # the new definition is based on the amplitude of the peaks,
            # not their position along the frequency axis
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mask = secondary_wavelength > primary_wavelength

            temp = secondary_wavelength[mask]
            secondary_wavelength[mask] = primary_wavelength[mask]
            primary_wavelength[mask] = temp

        self.amplitude_max = amplitude_max
        self.amplitude_ratio = amplitude_ratio
        self.primary_wavelength = primary_wavelength
        self.secondary_wavelength = secondary_wavelength
        self.track_length = track_length

        timer.toc('posture.amplitude_and_wavelength')

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.amplitude_max = utils.get_nested_h5_field(
            wf.h, ['posture', 'amplitude', 'max'])
        self.amplitude_ratio = utils.get_nested_h5_field(
            wf.h, ['posture', 'amplitude', 'ratio'])
        self.primary_wavelength = utils.get_nested_h5_field(
            wf.h, ['posture', 'wavelength', 'primary'])
        self.secondary_wavelength = utils.get_nested_h5_field(
            wf.h, ['posture', 'wavelength', 'secondary'])
        self.track_length = utils.get_nested_h5_field(
            wf.h, ['posture', 'tracklength'])

        return self


class AmplitudeMax(Feature):
    """
    Feature: posture.amplitude_max
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'posture.amplitude_wavelength_processor').amplitude_max

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class AmplitudeRatio(Feature):
    """
    Feature: posture.amplitude_ratio
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'posture.amplitude_wavelength_processor').amplitude_ratio

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class PrimaryWavelength(Feature):
    """
    Feature: posture.primary_wavelength
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'posture.amplitude_wavelength_processor').primary_wavelength

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)

    def __eq__(self, other):

        return utils.correlation(self.value, other.value,
                                 self.name, high_corr_value=0.98,
                                 merge_nans=True)


class SecondaryWavelength(Feature):
    """
    Feature: posture.secondary_wavelength
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'posture.amplitude_wavelength_processor').secondary_wavelength

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)

    def __eq__(self, other):

        return utils.correlation(self.value, other.value,
                                 self.name, high_corr_value=0.98,
                                 merge_nans=True)


class TrackLength(Feature):
    """
    Feature: posture.track_length
    """

    def __init__(self, wf, feature_name):
        self.name = feature_name
        self.value = self.get_feature(
            wf, 'posture.amplitude_wavelength_processor').track_length

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class Coils(Feature):

    def __init__(self, wf, feature_name):
        """
        Feature Name: posture.coils

        Get the worm's posture.coils.


        """

        self.name = feature_name
        midbody_distance = self.get_feature(
            wf, 'locomotion.velocity.mibdody.distance').value

        options = wf.options
        posture_options = options.posture
        timer = wf.timer

        fps = wf.video_info.fps

        timer.tic()

        frame_code = wf.video_info.frame_code

        COIL_FRAME_THRESHOLD = posture_options.coiling_frame_threshold(fps)

        # These are values that are specific to the MRC processor
        COIL_START_CODES = [105, 106]
        # Code that indicates a frame was successfully segmented
        FRAME_SEGMENTED = 1

        # Algorithm: Whenever a new start is found, find the
        # first segmented frame; that's the end.

        # Add on a frame to allow closing a coil at the end ...
        coil_start_mask = (frame_code == COIL_START_CODES[0]) | (
            frame_code == COIL_START_CODES[1])
        np_false = np.zeros((1,), dtype=bool)
        coil_start_mask = np.concatenate((coil_start_mask, np_false))

        # NOTE: These are not guaranteed ends, just possible ends ...
        end_coil_mask = frame_code == FRAME_SEGMENTED
        np_true = ~np_false
        end_coil_mask = np.concatenate((end_coil_mask, np_true))

        in_coil = False
        coil_frame_start = -1
        n_coils = 0
        n_frames_plus1 = len(frame_code) + 1

        starts = []
        ends = []

        for iFrame in range(n_frames_plus1):
            if in_coil:
                if end_coil_mask[iFrame]:
                    n_coil_frames = iFrame - coil_frame_start
                    if n_coil_frames >= COIL_FRAME_THRESHOLD:
                        n_coils += 1

                        starts.append(coil_frame_start)
                        ends.append(iFrame - 1)

                    in_coil = False
            elif coil_start_mask[iFrame]:
                in_coil = True
                coil_frame_start = iFrame

        if options.mimic_old_behaviour:
            if (len(starts) > 0) and (ends[-1] == len(frame_code) - 1):
                ends[-1] += -1
                starts[-1] += -1

        temp = events.EventList(np.transpose(np.vstack((starts, ends))))

        timer.toc('posture.coils')

        self.value = events.EventListWithFeatures(fps, temp, midbody_distance)

        self.no_events = self.value.is_null

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        ref = utils.get_nested_h5_field(
            wf.h, ['posture', 'coils'], resolve_value=False)
        self.value = events.EventListWithFeatures.from_disk(ref, 'MRC')
        self.no_events = self.value.is_null
        return self


class Kinks(Feature):

    def __init__(self, wf, feature_name):
        """
        Feature Name: posture.kinks

        Parameters
        ----------
        features_ref : open-worm-analysis-toolbox.features.worm_features.WormFeatures

        Returns
        -------
        numpy.array

        """

        self.name = feature_name

        nw = wf.nw
        timer = wf.timer
        timer.tic()

        options = wf.options

        KINK_LENGTH_THRESHOLD_PCT = options.posture.kink_length_threshold_pct

        bend_angles = nw.angles

        # Determine the bend segment length threshold.
        n_angles = bend_angles.shape[0]
        length_threshold = np.round(n_angles * KINK_LENGTH_THRESHOLD_PCT)

        # Compute a gaussian filter for the angles.
        #----------------------------------------------------------------------
        # JAH NOTE: This is a nice way of getting the appropriate odd value
        # unlike the other code with so many if statements ...
        #- see window code which tries to get an odd value ...
        #- I'd like to go back and fix that code ...
        half_length_thr = np.round(length_threshold / 2)
        gauss_filter = utils.gausswin(
            half_length_thr * 2 + 1) / half_length_thr

        # Compute the kinks for the worms.
        n_frames = bend_angles.shape[1]
        n_kinks_all = np.full(n_frames, np.nan, dtype=float)

        #(np.any(np.logical_or(mask_pos,mask_neg),axis=0)).nonzero()[0]

        nan_mask = np.isnan(bend_angles)

        for iFrame in (~np.all(nan_mask, axis=0)).nonzero()[0]:
            smoothed_bend_angles = filters.convolve1d(
                bend_angles[:, iFrame], gauss_filter, cval=0, mode='constant')

            # This code is nearly identical in getForaging
            #-------------------------------------------------------
            n_frames = smoothed_bend_angles.shape[0]

            with np.errstate(invalid='ignore'):
                dataSign = np.sign(smoothed_bend_angles)

            if np.any(np.equal(dataSign, 0)):
                # I don't expect that we'll ever actually reach 0
                # The code for zero was a bit weird, it keeps counting if no sign
                # change i.e. + + + 0 + + + => all +
                #
                # but it counts for both if sign change
                # + + 0 - - - => 3 +s and 4 -s
                
                #this case does happend. I will continue (default nan) instead of risign an error (AEJ)
                #raise Exception("Unhandled code case")
                continue

            sign_change_I = (
                np.not_equal(dataSign[1:], dataSign[0:-1])).nonzero()[0]

            end_I = np.concatenate(
                (sign_change_I,
                 n_frames * np.ones(1, dtype=np.result_type(sign_change_I))))

            wtf1 = np.zeros(1, dtype=np.result_type(sign_change_I))
            wtf2 = sign_change_I + 1
            # +2? due to inclusion rules???
            start_I = np.concatenate((wtf1, wtf2))

            # All NaN values are considered sign changes, remove these ...
            keep_mask = np.logical_not(np.isnan(smoothed_bend_angles[start_I]))

            start_I = start_I[keep_mask]
            end_I = end_I[keep_mask]

            # The old code had a provision for having NaN values in the middle
            # of the worm. I have not translated that feature to the newer code. I
            # don't think it will ever happen though for a valid frame, only on the
            # edges should you have NaN values.
            if start_I.size != 0 and \
               np.any(np.isnan(smoothed_bend_angles[start_I[0]:end_I[-1]])):
                #this case does happend. I will continue (default nan) instead of risign an error (AEJ)
                #raise Exception("Unhandled code case")
                continue

            #-------------------------------------------------------
            # End of identical code ...

            lengths = end_I - start_I + 1

            # Adjust lengths for first and last:
            # Basically we allow NaN values to count towards the length for the
            # first and last stretches
            if lengths.size != 0:
                if start_I[0] != 0:  # Due to leading NaNs
                    lengths[0] = end_I[0] + 1
                if end_I[-1] != n_frames:  # Due to trailing NaNs
                    lengths[-1] = n_frames - start_I[-1]

            n_kinks_all[iFrame] = np.sum(lengths >= length_threshold)

        timer.toc('posture.kinks')

        self.value = n_kinks_all

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.value = utils.get_nested_h5_field(wf.h, ['posture', 'kinks'])
        return self


def load_eigen_worms():
    """
    Load the eigen_worms, which are stored in a Matlab data file

    The eigenworms were computed by the Schafer lab based on N2 worms

    Returns
    ----------
    eigen_worms: [7 x 48]

    From http://stackoverflow.com/questions/50499/

    """

    eigen_worm_file_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    config.EIGENWORM_FILE)

    h = h5py.File(eigen_worm_file_path, 'r')
    eigen_worms = h['eigenWorms'].value

    return np.transpose(eigen_worms)


class EigenProjectionProcessor(Feature):

    def __init__(self, wf, feature_name):
        """
        Feature: 'posture.all_eigenprojections'

        Parameters
        ----------
        features_ref : open-worm-analysis-toolbox.features.worm_features.WormFeatures

        Returns
        -------
        eigen_projections: [N_EIGENWORMS_USE, n_frames]

        """

        self.name = feature_name

        posture_options = wf.options.posture
        N_EIGENWORMS_USE = posture_options.n_eigenworms_use
        timer = wf.timer
        timer.toc
        # eigen_worms: [7,48]
        eigen_worms = load_eigen_worms()

        

        sx = wf.nw.skeleton_x
        sy = wf.nw.skeleton_y
        #nw.angles calculation is inconsistent with this one...
        # I think bends angles should be between -180 to 180, while for the eigenworms they must be continous.
        angles = np.arctan2(np.diff(sy, n=1, axis=0), np.diff(sx, n=1, axis=0))
        if wf.nw.video_info.ventral_mode == 2:
            #switch in the angle sign in case of the contour orientation is anticlockwise
            angles = -angles

        n_frames = angles.shape[1]

        # need to deal with cases where angle changes discontinuously from -pi
        # to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
        # respectively to all remaining points.  This effectively extends the
        # range outside the -pi to pi range.  Everything is re-centred later
        # when we subtract off the mean.
        false_row = np.zeros((1, n_frames), dtype=bool)

        # NOTE: By adding the row of falses, we shift the trues
        # to the next value, which allows indices to match. Otherwise after every
        # find statement we would need to add 1, I think this is a bit faster
        # ...

        with np.errstate(invalid='ignore'):
            mask_pos = np.concatenate(
                (false_row, np.diff(angles, n=1, axis=0) > np.pi), axis=0)
            mask_neg = np.concatenate(
                (false_row, np.diff(angles, n=1, axis=0) < -np.pi), axis=0)

        # Only fix the frames we need to, in which there is a jump in going
        # from one segment to the next ...
        fix_frames_I = (
            np.any(np.logical_or(mask_pos, mask_neg), axis=0)).nonzero()[0]

        for cur_frame in fix_frames_I:

            positive_jump_I = (mask_pos[:, cur_frame]).nonzero()[0]
            negative_jump_I = (mask_neg[:, cur_frame]).nonzero()[0]

            # subtract 2pi from remainging data after positive jumps
            # Note that the jumps impact all subsequent frames
            for cur_pos_jump in positive_jump_I:
                angles[cur_pos_jump:, cur_frame] -= 2 * np.pi

            # add 2pi to remaining data after negative jumps
            for cur_neg_jump in negative_jump_I:
                angles[cur_neg_jump:, cur_frame] += 2 * np.pi

        angles = angles - np.mean(angles, axis=0)
        
        
        eigen_projections = np.dot(eigen_worms[0:N_EIGENWORMS_USE, :], angles)
        
        #change signs for anticlockwise
        #if nw.video_info.ventral_mode == 2:
        #    eigen_projections = -eigen_projections


        timer.toc('posture.eigenworms')

        self.value = eigen_projections

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.value = np.transpose(utils.get_nested_h5_field(
            wf.h, ['posture', 'eigenProjection'], is_matrix=True))
        return self


class EigenProjection(Feature):

    def __init__(self, wf, feature_name):
        self.name = feature_name
        projection_matrix = self.get_feature(
            wf, 'posture.all_eigenprojections').value
        index = int(feature_name[-1])
        self.value = projection_matrix[index, :]

    @classmethod
    def from_schafer_file(cls, wf, feature_name):
        return cls(wf, feature_name)


class Bend(Feature):
    """

    Old MRC code used very different indices for this part:
    #s are in Matlab format, 1 based and inclusive
    Indices Mismatch
    %            OLD                        NEW
    %---------------------------------------------
    %head     : 1:9                         1:8
    %neck     : 9:17                        9:16
    %midbody  : 17:32 (mean) 17:31 (std)    17:33
    %hip      : 31:39                       34:41
    %tail     : 39:48                       42:49
    """

    def __init__(self, wf, feature_name, bend_name):

        self.name = 'posture.bends.' + bend_name

        nw = wf.nw

        # Retrieve the part of the worm we are currently looking at:
        bend_angles = nw.get_partition(bend_name, 'angles')

        # shape = (n):

        # Suppress RuntimeWarning: Mean of empty slice for those frames
        # that are ALL NaN.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            # Throws warning on taking the mean of an empty slice
            temp_mean = np.nanmean(a=bend_angles, axis=0)

            # Throws warning when degrees of freedom <= 0 for slice
            temp_std = np.nanstd(a=bend_angles, axis=0)

            # Sign the standard deviation (to provide the bend's
            # dorsal/ventral orientation)
            temp_std[temp_mean < 0] *= -1

        self.mean = temp_mean
        self.std_dev = temp_std

    @classmethod
    def from_schafer_file(cls, wf, feature_name, bend_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.mean = utils.get_nested_h5_field(
            wf.h, ['posture', 'bends', bend_name, 'mean'])
        self.std_dev = utils.get_nested_h5_field(
            wf.h, ['posture', 'bends', bend_name, 'stdDev'])

        return self


class BendMean(Feature):

    def __init__(self, wf, feature_name, bend_name):

        parent_name = generic_features.get_parent_feature_name(feature_name)
        self.name = feature_name
        self.value = self.get_feature(wf, parent_name).mean

    @classmethod
    def from_schafer_file(cls, wf, feature_name, bend_name):
        return cls(wf, feature_name, bend_name)

    def __eq__(self, other):
        return utils.correlation(self.value, other.value,
                                 self.name, high_corr_value=0.95)


class BendStdDev(Feature):

    def __init__(self, wf, feature_name, bend_name):

        parent_name = generic_features.get_parent_feature_name(feature_name)
        self.name = feature_name
        self.value = self.get_feature(wf, parent_name).std_dev

    @classmethod
    def from_schafer_file(cls, wf, feature_name, bend_name):
        return cls(wf, feature_name, bend_name)

    def __eq__(self, other):
        return utils.correlation(self.value, other.value,
                                 self.name, high_corr_value=0.60)


class Skeleton(Feature):

    """
    Feature: posture.skeleton

    This just holds onto the skeleton x & y coordinates from normalized worm.
    We don't do anything with these coordinates as far as feature processing.
    """

    def __init__(self, wf, feature_name):

        nw = wf.nw

        self.name = feature_name
        self.x = nw.skeleton_x
        self.y = nw.skeleton_y

    @classmethod
    def from_schafer_file(cls, wf, feature_name, bend_name):
        self = cls.__new__(cls)
        self.name = feature_name
        self.x = utils.get_nested_h5_field(
            wf.h, ['posture', 'skeleton', 'x'], is_matrix=True)
        self.y = utils.get_nested_h5_field(
            wf.h, ['posture', 'skeleton', 'y'], is_matrix=True)
        return self

    @classmethod
    def from_disk(cls, skeleton_ref):
        self = cls.__new__(cls)

        x_temp = utils._extract_time_from_disk(skeleton_ref, 'x',
                                               is_matrix=True)
        y_temp = utils._extract_time_from_disk(skeleton_ref, 'y',
                                               is_matrix=True)
        self.x = x_temp.transpose()
        self.y = y_temp.transpose()

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        eq_skeleton_x = utils.correlation(np.ravel(self.x),
                                          np.ravel(other.x),
                                          'posture.skeleton.x')
        eq_skeleton_y = utils.correlation(np.ravel(self.y),
                                          np.ravel(other.y),
                                          'posture.skeleton.y')

        return eq_skeleton_x and eq_skeleton_y


class Direction(Feature):

    """

    Implements Features:
    --------------------
    posture.directions.tail2head
    posture.directions.tail
    posture.directions.head

    """

    def __init__(self, wf, feature_name, key_name):
        """
        Feature: posture.directions.[key_name]

        Parameters
        ----------
        wf :

        """

        nw = wf.nw

        sx = nw.skeleton_x
        sy = nw.skeleton_y
        wp = nw.worm_partitions

        if key_name == 'tail2head':
            tip_I = wp['head']  # I - "indices" - really a tuple of start,stop
            # tail is referencing a vector tail, not the worm's tail
            tail_I = wp['tail']
        elif key_name == 'head':
            tip_I = wp['head_tip']
            tail_I = wp['head_base']
        else:
            tip_I = wp['tail_tip']
            tail_I = wp['tail_base']

        tip_slice = slice(*tip_I)
        tail_slice = slice(*tail_I)

        # Compute the centroids of the tip and tail
        # then compute a direction vector between them (tip - tail)

        tip_x = np.mean(sx[tip_slice, :], axis=0)
        tip_y = np.mean(sy[tip_slice, :], axis=0)
        tail_x = np.mean(sx[tail_slice, :], axis=0)
        tail_y = np.mean(sy[tail_slice, :], axis=0)


        #attempt to match segworm behaviour. This should shift the angles by 180.
        # dy = (tip_y - tail_y)
        # dx = (tip_x - tail_x)
        # dir_value = 180 / np.pi * (-np.arctan2(dy,-dx))
        dir_value = 180 / np.pi * np.arctan2(tip_y - tail_y, tip_x - tail_x)
        self.value = dir_value

    @classmethod
    def from_schafer_file(cls, wf, feature_name, key_name):
        self = cls.__new__(cls)
        self.value = utils.get_nested_h5_field(
            wf.h, ['posture', 'directions', key_name])
        return self

