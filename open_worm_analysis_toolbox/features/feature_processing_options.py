# -*- coding: utf-8 -*-
"""
This module holds a class that is referenced when
processing features.

I'd like to move things from "config" into here ...
- @JimHokanson

"""

from __future__ import division

from .. import utils

# Can't do this, would be circular
#from .worm_features import WormFeatures


class FeatureProcessingOptions(object):
    """
    """

    def __init__(self):

        # The idea with this attribute is that functions will check if
        # they are in this list. If they are then they can display some
        # sort of popup that clarifies how they are working.
        #
        # No functions actually use this yet. It is just a placeholder.
        #
        # An example of this might be:
        #    'morphology.length'
        #    s
        self.functions_to_explain = []

        # This indicates that, where possible, code should attempt to
        # replicate the errors and inconsistencies present in the way that
        # the Schafer lab computed features. This can be useful for ensuring
        # that we are able to compute features in the same way that they did.
        #
        # NOTE: There are a few instances where this is not supported such
        # that the behavior will not match even if this value is set to True.
        self.mimic_old_behaviour = True

        self.locomotion = LocomotionOptions()
        self.posture = PostureOptions()

        # TODO: Implement this.
        # This is not yet implemented. The idea is to support not
        # computing certain features. We might also allow disabling
        # certain groups of features.
        self.features_to_ignore = []

    def should_compute_feature(self, feature_name, worm_features):
        """

        """
        # TODO
        return True

    def disable_contour_features(self):
        """
        Contour features:

        """
        # see self.features_to_ignore
        contour_dependent_features = [
            'morphology.width',
            'morphology.area',
            'morphology.area_per_length',
            'morphology.width_per_length',
            'posture.eccentricity']

        self.features_to_ignore = list(set(self.features_to_ignore +
                                           contour_dependent_features))

    def disable_feature_sections(self, section_names):
        """

        This can be used to disable processing of features by section
        (see the options available below)

        Modifies 'features_to_ignore'

        Parameters
        ----------
        section_names : list[str]
            Options are:
            - morphology
            - locomotion
            - posture
            - path

        Examples
        --------
        fpo.disable_feature_sections(['morphology'])

        fpo.disable_feature_sections(['morphology','locomotion'])

        """
        new_ignores = []
        f = IgnorableFeatures()
        for section in section_names:
            new_ignores.extend(getattr(f, section))

        self.features_to_ignore = list(set(self.features_to_ignore +
                                           new_ignores))

    def __repr__(self):
        return utils.print_object(self)


class PostureOptions(object):

    def __init__(self):
        # Grid size for estimating eccentricity, this is the
        # max # of points that will fill the wide dimension.
        # (scalar) The # of points to place in the long dimension. More points
        # gives a more accurate estimate of the ellipse but increases
        # the calculation time.
        #
        # Used by: posture_features.get_eccentricity_and_orientation
        self.n_eccentricity_grid_points = 50

        # The maximum # of available values is 7 although technically there
        # are generally 48 eigenvectors avaiable, we've just only precomputed
        # 7 to use for the projections
        #
        # Used by: posture.eigenprojections
        self.n_eigenworms_use = 6

        # This the fraction of the worm length that a bend must be
        # in order to be counted. The # of worm points
        # (this_value*worm_length_in_samples) is rounded to an integer
        # value. The threshold value is inclusive.
        #
        # Used by: posture_features.get_worm_kinks
        self.kink_length_threshold_pct = 1 / 12

        self.wavelength = PostureWavelengthOptions()

    def coiling_frame_threshold(self, fps):
        # This is the # of
        # frames that an epoch must exceed in order for it to be truly
        # considered a coiling event
        # Current value translation: 1/5 of a second
        #
        # Used by: posture_features.get_worm_coils
        return int(round(1 / 5 * fps))


class PostureWavelengthOptions(object):
    """
    These options are all used in:
    get_amplitude_and_wavelength

    """

    def __init__(self):

        self.n_points_fft = 512

        # This value is in samples, not a
        # spatial frequency. The spatial frequency sampling also
        # varies by the worm length, so this resolution varies on a
        # frame by frame basis.
        self.min_dist_peaks = 5

        self.pct_max_cutoff = 0.5
        self.pct_cutoff = 2


class LocomotionOptions(object):

    def __init__(self):
        # locomotion_features.LocomotionVelocity
        #-------------------------------------
        # Units: seconds
        # NOTE: We could get the defaults from the class ...
        self.velocity_tip_diff = 0.25
        self.velocity_body_diff = 0.5

        # locomotion_features.MotionEvents
        #--------------------------------------
        # Interpolate only this length of NaN run; anything longer is
        # probably an omega turn.
        # If set to "None", interpolate all lengths (i.e. infinity)
        # TODO - Inf would be a better specification
        self.motion_codes_longest_nan_run_to_interpolate = None
        # These are a percentage of the worm's length
        self.motion_codes_speed_threshold_pct = 0.05
        self.motion_codes_distance_threshold_pct = 0.05
        self.motion_codes_pause_threshold_pct = 0.025

        #   These are times (s)
        self.motion_codes_min_frames_threshold = 0.5
        self.motion_codes_max_interframes_threshold = 0.25

        # locomotion_bends.LocomotionCrawlingBends
        self.crawling_bends = LocomotionCrawlingBends()
        self.foraging_bends = LocomotionForagingBends()
        self.locomotion_turns = LocomotionTurns()

    def __repr__(self):
        return utils.print_object(self)


class LocomotionTurns(object):

    def __init__(self):
        self.max_interpolation_gap_allowed = 9  # frames

    def min_omega_event_length(self, fps):
        return int(round(fps / 4))

        # TODO: There is still a lot to put into here


class LocomotionForagingBends(object):

    def __init__(self):
        # NOTE: The nose & neck can also be thought of as the head tip
        # and head neck
        pass

    def min_nose_window_samples(self, fps):
        return int(round(0.1 * fps))

    def max_samples_interp_nose(self, fps):
        return 2 * self.min_nose_window_samples(fps) - 1


class LocomotionCrawlingBends(object):

    def __init__(self):
        self.fft_n_samples = 2 ** 14

        self.bends_partitions = \
            {'head': (5, 10),
             'midbody': (22, 27),
             'tail': (39, 44)}

        self.peak_energy_threshold = 0.5

        # max_amplitude_pct_bandwidth - when determining the bandwidth,
        # the minimums that are found can't exceed this percentage of the
        # maximum.  Doing so invalidates the result.
        self.max_amplitude_pct_bandwidth = 0.5

        self.min_time_for_bend = 0.5
        self.max_time_for_bend = 15

        # TODO: What are the units on these things ????
        # This is a spatial frequency

        # The comment that went with this in the original code was:
        #"require at least 50% of the wave"
        self.min_frequency = 1 / (4 * self.max_time_for_bend)

        # This is wrong ...
        #self.min_frequency = 0.25 * self.max_time_for_bend

        # This is a processing optimization.
        # How far into the maximum peaks should we look ...
        # If this value is low, an expensive computation could go faster.
        # If it is too low, then we end up rerunning the calculation the
        # whole dataset and we end up losing time.
        self.initial_max_I_pct = 0.5

    def max_frequency(self, fps):
        # What is the technical max???? 0.5 fps????
        return 0.25 * fps

    def __repr__(self):
        return utils.print_object(self)


class IgnorableFeatures:
    """
    I'm not thrilled with where this is placed, but placing it in
    WormFeatures creates a circular dependency

    """

    def __init__(self):
        temp = [
            'length',
            'width',
            'area',
            'area_per_length',
            'width_per_length']
        self.morphology = ['morphology.' + s for s in temp]
        # None of these are implemented ...

        temp = ['velocity', 'motion_events', 'motion_mode',
                'crawling_bends', 'foraging_bends', 'turns']
        self.locomotion = ['locomotion.' + s for s in temp]
        # locomotion
        # crawling_bends: Done
        # turns: Done

        temp = ['bends', 'eccentricity', 'amplitude_and_wavelength',
                'kinks', 'coils', 'directions', 'eigen_projection']
        self.posture = ['posture.' + s for s in temp]
        # None of these are implemented ...
