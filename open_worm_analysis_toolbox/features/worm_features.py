# -*- coding: utf-8 -*-
"""
WormFeatures module

Contains the classes needed for users to calculate the features
of a worm from a NormalizedWorm instance.

Classes
---------------------------------------
WormMorphology
WormLocomotion
WormPosture
WormPath

WormFeatures
FeatureProcessingSpec


A translation of Matlab code written by Jim Hokanson, in the
SegwormMatlabClasses GitHub repo.

Original code path:
SegwormMatlabClasses/+seg_worm/@feature_calculator/features.m

"""

import copy
import csv
import os
import warnings
import h5py  # For loading from disk
import numpy as np
import collections  # For namedtuple, OrderedDict
import pandas as pd

from .. import utils

from . import feature_processing_options as fpo
from . import events
from . import generic_features
from . import path_features
from . import posture_features
from . import locomotion_features
from . import locomotion_bends
from . import locomotion_turns
from . import morphology_features


FEATURE_SPEC_CSV_PATH = os.path.join(
    os.path.dirname(
        os.path.realpath(__file__)),
    'feature_metadata',
    'features_list.csv')

"""
===============================================================================
===============================================================================
"""


class WormMorphology(object):
    """
    The worm's morphology features class.

    Nature Methods Description
    ---------------------------------------
    1. Length. Worm length is computed from the segmented skeleton by
    converting the chain-code pixel length to microns.

    2. Widths. Worm width is computed from the segmented skeleton. The
    head, midbody, and tail widths are measured as the mean of the widths
    associated with the skeleton points covering their respective sections.
    These widths are converted to microns.

    3. Area. The worm area is computed from the number of pixels within the
    segmented contour. The sum of the pixels is converted to microns2.

    4. Area/Length.

    5. Midbody Width/Length.


    Notes
    ---------------------------------------
    Formerly SegwormMatlabClasses /
    +seg_worm / @feature_calculator / getMorphologyFeatures.m

    Old files that served as a reference:
      morphology_process.m
      schaferFeatures_process.m

    """

    def __init__(self, features_ref):
        """

        Parameters:
        -----------
        features_ref : WormFeatures

        """
        print('Calculating Morphology Features')

        nw = features_ref.nw

        self.length = nw.length

        self.width = morphology_features.Widths(features_ref)

        self.area = nw.area

        self.area_per_length = self.area / self.length
        self.width_per_length = self.width.midbody / self.length

    @classmethod
    def from_disk(cls, m_var):

        self = cls.__new__(cls)

        self.length = utils._extract_time_from_disk(m_var, 'length')
        self.width = morphology_features.Widths.from_disk(m_var['width'])
        self.area = utils._extract_time_from_disk(m_var, 'area')
        self.area_per_length = utils._extract_time_from_disk(
            m_var, 'areaPerLength')
        self.width_per_length = utils._extract_time_from_disk(
            m_var, 'widthPerLength')

        return self

    def __eq__(self, other):

        return utils.correlation(
            self.length,
            other.length,
            'morph.length') and self.width == other.width and utils.correlation(
            self.area,
            other.area,
            'morph.area') and utils.correlation(
            self.area_per_length,
            other.area_per_length,
            'morph.area_per_length') and utils.correlation(
                self.width_per_length,
                other.width_per_length,
            'morph.width_per_length')

    def __repr__(self):
        return utils.print_object(self)

    def save_for_gepetto(self):
        # See
        # https://github.com/openworm/org.geppetto.recording/blob/master/org/geppetto/recording/CreateTestGeppettoRecording.py
        pass


"""
===============================================================================
===============================================================================
"""


class WormLocomotion(object):

    """
    The worm's locomotion features class.

    Attributes
    ----------
    velocity :
    motion_events :
    motion_mode :
    crawling_bends :
    foraging_bends :
    turns :

    """

    def __init__(self, features_ref):
        """
        Initialization method for WormLocomotion

        Parameters
        ----------
        features_ref : WormFeatures

        """
        print('Calculating Locomotion Features')

        nw = features_ref.nw
        video_info = features_ref.video_info

        self.velocity = locomotion_features.LocomotionVelocity(features_ref)

        self.motion_events = \
            locomotion_features.MotionEvents(features_ref,
                                             self.velocity.midbody.speed,
                                             nw.length)

        self.motion_mode = self.motion_events.get_motion_mode()

        self.crawling_bends = locomotion_bends.LocomotionCrawlingBends(
            features_ref,
            nw.angles,
            self.motion_events.is_paused,
            video_info.is_segmented)

        self.foraging_bends = locomotion_bends.LocomotionForagingBends(
            features_ref,
            video_info.is_segmented,
            video_info.ventral_mode)

        is_stage_movement = video_info.is_stage_movement

        self.turns = locomotion_turns.LocomotionTurns(
            features_ref,
            nw.angles,
            is_stage_movement,
            self.velocity.get_midbody_distance(),
            nw.skeleton_x,
            nw.skeleton_y)

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):

        # TODO: Allow for a global config that provides more info ...
        # in case anything fails ...
        #
        #   JAH: I'm not sure how this will work. We might need to move
        #   away from the equality operator to a function that returns
        #   an equality result

        # The order here matches the order the properties are populated
        # in the constructor
        same_locomotion = True

        if not (self.velocity == other.velocity):
            same_locomotion = False

        if not (self.motion_events == other.motion_events):
            same_locomotion = False

        # Test motion codes
        if not utils.correlation(self.motion_mode, other.motion_mode,
                                 'locomotion.motion_mode'):
            same_locomotion = False

        # TODO: Define ne for all functions (instead of needing not(eq))
        if not (self.crawling_bends == other.crawling_bends):
            print('Mismatch in locomotion.crawling_bends events')
            same_locomotion = False

        if not (self.foraging_bends == other.foraging_bends):
            print('Mismatch in locomotion.foraging events')
            same_locomotion = False

        # TODO: Make eq in events be an error - use test_equality instead
        # NOTE: turns is a container class that implements eq, and is not
        # an EventList
        if not (self.turns == other.turns):
            print('Mismatch in locomotion.turns events')
            same_locomotion = False

        return same_locomotion

    @classmethod
    def from_disk(cls, m_var):
        """
        Parameters
        ----------
        m_var : type???? h5py.Group???
            ?? Why is this this called m_var????
        """

        self = cls.__new__(cls)

        self.velocity = locomotion_features.LocomotionVelocity.from_disk(m_var)

        self.motion_events = locomotion_features.MotionEvents.from_disk(m_var)

        self.motion_mode = self.motion_events.get_motion_mode()

        bend_ref = m_var['bends']
        self.crawling_bends = \
            locomotion_bends.LocomotionCrawlingBends.from_disk(bend_ref)

        self.foraging_bends = \
            locomotion_bends.LocomotionForagingBends.\
            from_disk(bend_ref['foraging'])

        self.turns = locomotion_turns.LocomotionTurns.from_disk(m_var['turns'])

        return self


"""
===============================================================================
===============================================================================
"""


class WormPosture(object):

    """
    Worm posture feature class.

    Notes
    -----
    Formerly:
    SegwormMatlabClasses/+seg_worm/@feature_calculator/getPostureFeatures.m

    Former usage:

    Prior to this, it was originally "schaferFeatures_process"

    Formerly,
    - Indices were inconsistently defined for bends relative to other code
    - stdDev for bends is signed as well, based on means ...

    Unfinished Status
    ---------------------------------------
    (@JimHokanson, is this still true?)
    - seg_worm.feature_helpers.posture.wormKinks - not yet examined
    - distance - missing input to function, need to process locomotion
      first

    """

    def __init__(self, features_ref, midbody_distance):
        """
        Initialization method for WormPosture

        Parameters
        ----------
        normalized_worm: a NormalizedWorm instance

        """
        print('Calculating Posture Features')

        self.bends = posture_features.Bends.create(features_ref)

        self.eccentricity, self.orientation = \
            posture_features.get_eccentricity_and_orientation(features_ref)

        amp_wave_track = posture_features.AmplitudeAndWavelength(
            self.orientation, features_ref)

        self.amplitude_max = amp_wave_track.amplitude_max
        self.amplitude_ratio = amp_wave_track.amplitude_ratio
        self.primary_wavelength = amp_wave_track.primary_wavelength
        self.secondary_wavelength = amp_wave_track.secondary_wavelength
        self.track_length = amp_wave_track.track_length

        self.kinks = posture_features.get_worm_kinks(features_ref)

        self.coils = posture_features.get_worm_coils(features_ref,
                                                     midbody_distance)

        self.directions = posture_features.Directions(features_ref)

        self.skeleton = posture_features.Skeleton(features_ref, 'temp')

        self.eigen_projection = posture_features.get_eigenworms(features_ref)

    """
    We need these six @property methods because otherwise eigen_projections
    are the only first-class sub-extended features that are not fully
    addressable by nested object references.  Without these, I'd have to say:

    worm_features_object.posture.eigen_projection[0]

    Instead I can say:

    worm_features_object.posture.eigen_projection0

    ...which is crucial for the object-data mapping, for example when
    pulling a pandas DataFrame via WormFeatures.getDataFrame.
    """
    @property
    def eigen_projection0(self):
        return self.eigen_projection[0]

    @property
    def eigen_projection1(self):
        return self.eigen_projection[1]

    @property
    def eigen_projection2(self):
        return self.eigen_projection[2]

    @property
    def eigen_projection3(self):
        return self.eigen_projection[3]

    @property
    def eigen_projection4(self):
        return self.eigen_projection[4]

    @property
    def eigen_projection5(self):
        return self.eigen_projection[5]

    @classmethod
    def from_disk(cls, p_var):

        self = cls.__new__(cls)

        self.bends = posture_features.Bends.from_disk(p_var['bends'])

        temp_amp = p_var['amplitude']

        self.amplitude_max = utils._extract_time_from_disk(temp_amp, 'max')
        self.amplitude_ratio = utils._extract_time_from_disk(temp_amp,
                                                             'ratio')

        temp_wave = p_var['wavelength']
        self.primary_wavelength = utils._extract_time_from_disk(temp_wave,
                                                                'primary')
        self.secondary_wavelength = utils._extract_time_from_disk(temp_wave,
                                                                  'secondary')

        self.track_length = utils._extract_time_from_disk(p_var,
                                                          'tracklength')
        self.eccentricity = utils._extract_time_from_disk(p_var,
                                                          'eccentricity')
        self.kinks = utils._extract_time_from_disk(p_var, 'kinks')

        self.coils = events.EventListWithFeatures.from_disk(p_var['coils'],
                                                            'MRC')

        self.directions = \
            posture_features.Directions.from_disk(p_var['directions'])

        # TODO: Add contours

        self.skeleton = posture_features.Skeleton.from_disk(p_var['skeleton'])

        temp_eigen_projection = \
            utils._extract_time_from_disk(p_var, 'eigenProjection',
                                          is_matrix=True)

        self.eigen_projection = temp_eigen_projection.transpose()

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):

        # TODO: It would be nice to see all failures before returning false
        # We might want to make a comparison class that handles these details
        # and then prints the results

        # Doing all of these comparisons and then computing the results
        # allows any failures to be printed, which at this point is useful for
        # getting the code to align

        # Note that the order of these matches the order in which they are
        # populated in the constructor
        eq_bends = self.bends == other.bends
        eq_amplitude_max = utils.correlation(self.amplitude_max,
                                             other.amplitude_max,
                                             'posture.amplitude_max')
        eq_amplitude_ratio = utils.correlation(self.amplitude_ratio,
                                               other.amplitude_ratio,
                                               'posture.amplitude_ratio',
                                               high_corr_value=0.985)

        eq_primary_wavelength = \
            utils.correlation(self.primary_wavelength,
                              other.primary_wavelength,
                              'posture.primary_wavelength',
                              merge_nans=True,
                              high_corr_value=0.97)

        eq_secondary_wavelength = \
            utils.correlation(self.secondary_wavelength,
                              other.secondary_wavelength,
                              'posture.secondary_wavelength',
                              merge_nans=True,
                              high_corr_value=0.985)

        # TODO: We need a more lazy evaluation for these since they don't match
        # Are they even close?
        # We could provide a switch for exactly equal vs mimicing the old setup
        # in which our goal could be to shoot for close
        eq_track_length = utils.correlation(self.track_length,
                                            other.track_length,
                                            'posture.track_length')
        eq_eccentricity = utils.correlation(self.eccentricity,
                                            other.eccentricity,
                                            'posture.eccentricity',
                                            high_corr_value=0.99)
        eq_kinks = utils.correlation(self.kinks, other.kinks,
                                     'posture.kinks')

        eq_coils = self.coils.test_equality(other.coils, 'posture.coils')
        eq_directions = self.directions == other.directions
        eq_skeleton = self.skeleton == other.skeleton
        eq_eigen_projection = \
            utils.correlation(np.ravel(self.eigen_projection),
                              np.ravel(other.eigen_projection),
                              'posture.eigen_projection')

        # TODO: Reorder these as they appear above
        return \
            eq_bends and \
            eq_eccentricity and \
            eq_amplitude_ratio and \
            eq_track_length and \
            eq_kinks and \
            eq_primary_wavelength and \
            eq_secondary_wavelength and \
            eq_amplitude_max and \
            eq_skeleton and \
            eq_coils and \
            eq_directions and \
            eq_eigen_projection


"""
===============================================================================
===============================================================================
"""


class WormPath(object):

    """
    Worm posture feature class.

    Properties
    ------------------------
    range :
    duration :
    coordinates :
    curvature :

    Notes
    ---------------------------------------
    Formerly SegwormMatlabClasses /
    +seg_worm / @feature_calculator / getPathFeatures.m

    """

    def __init__(self, features_ref):
        """
        Initialization method for WormPosture

        Parameters:
        -----------
        features_ref: a WormFeatures instance
        """
        print('Calculating Path Features')

        nw = features_ref.nw

        self.range = path_features.Range(nw.contour_x, nw.contour_y)

        # Duration (aka Dwelling)
        self.duration = path_features.Duration(features_ref, 'temp')

        self.coordinates = path_features.Coordinates(features_ref, 'temp')

        # Curvature
        self.curvature = path_features.worm_path_curvature(features_ref)

    # TODO: Move to class in path_features
    @classmethod
    def _create_coordinates(cls, x, y):
        Coordinates = collections.namedtuple('Coordinates', ['x', 'y'])
        return Coordinates(x, y)

    @classmethod
    def from_disk(cls, path_var):

        self = cls.__new__(cls)

        self.range = path_features.Range.from_disk(path_var)
        self.duration = path_features.Duration.from_disk(path_var['duration'])

        self.coordinates = \
            path_features.Coordinates.from_disk(path_var['coordinates'])

        # Make a call to utils loader
        self.curvature = path_var['curvature'].value[:, 0]

        return self

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):

        return \
            self.range == other.range and \
            self.duration == other.duration and \
            self.coordinates == other.coordinates and \
            utils.correlation(self.curvature, other.curvature,
                              'path.curvature',
                              high_corr_value=0.95,
                              merge_nans=True)

        # NOTE: Unfortunately the curvature is slightly different. It
        # looks the same but I'm guessing there are a few off-by-1 errors
        # in it.


"""
===============================================================================
===============================================================================
"""


class WormFeatures(object):

    """
    This is the new features class. It will eventually replace the old class
    when things are all ready.

    Accessing Features
    ------------------
    Features should normally be accessed via get_feature() or via iteration
    over this object.

    Design Decisions
    ----------------
    # When iterating over the features, should we include null features?
            => YES? => this introduces stability between different videos
    # What should be returned from get_feature when it doesn't exist?
            => An empty feature object NYI
    # What is returned when a feature can't be computed because of a missing
    dependency?
            => An empty feature object NYI


    . Alternatively
    it is possible to access features directly via the 'features' attribute
    however this attribute only contains computed features.


    Attributes
    ----------
    video_info :
    options :
    nw :
    timer :
    specs : {FeatureProcessingSpec}
    features : {Feature}
        Contains all computed features that have been requested by the user.


    When loading from Schafer File
    h : hdf5 file reference



    """

    def __init__(self, nw, processing_options=None, specs='all'):
        """

        Parameters
        ----------
        nw : NormalizedWorm object
        specs :

        #The options will most likely change. We should have the options
        #be accessible from the specs
        processing_options: movement_validation.features.feature_processing_options

        #TODO: Expose just passing in feature names as well


        """
        if processing_options is None:
            processing_options = \
                fpo.FeatureProcessingOptions()

        # These are saved locally for reference by others when processing
        self.video_info = nw.video_info

        self.options = processing_options
        self.nw = nw
        self.timer = utils.ElementTimer()

        self.initialize_features()

        # TODO: We should eventually support a list of specs as well
        # TODO: We might also allow transforming the specs (like changing options),
        # which this doesn't handle since we are only extracting the names
        if isinstance(specs, pd.core.frame.DataFrame):
            # This wouldn't be good if the specs have changed.
            # We would need to change the initialize_features() call
            self.get_features(specs['feature_name'])
        else:
            self._retrieve_all_features()

    def __iter__(self):
        """  Let's allow iteration over the features """
        all_features = self.features
        for temp in all_features:
            yield temp

    def copy(self, new_features):
        """
        This method was introduced for "feature expansion"

        Parameters
        ----------
        new_features : list or dict (only list supported)
        """
        new_self = self.__new__(self.__class__)

        # We need to setup features and specs
        # get specs from features

        # specs : {FeatureProcessingSpec}
        # features : {Feature}

        d = self.__dict__
        for key in d:
            temp = d[key]
            if key in ['features', 'specs', 'h', '_temp_features']:
                pass
                # do nothing
                # setattr(new_self,'spec',temp.copy())
            else:
                setattr(new_self, key, copy.copy(temp))

        # Currently assuming a list for features

        temp_features = collections.OrderedDict()
        new_specs = collections.OrderedDict()
        for cur_feature in new_features:
            feature_name = cur_feature.name
            temp_features[feature_name] = cur_feature
            new_specs[feature_name] = cur_feature.spec

        new_self._features = temp_features
        new_self.specs = new_specs

        return new_self

    @classmethod
    def from_disk(cls, data_file_path):
        """
        Creates an instance of the class from disk.

        Ideally we would support loading of any file type. For now
        we'll punt on building in any logic until we have more types to deal
        with.
        """
        # This ideally would allow us to load any file from disk.
        #
        # For now we'll punt on this logic
        return cls._from_schafer_file(data_file_path)

    @classmethod
    def _from_schafer_file(cls, data_file_path):
        """
        Load features from the Schafer lab feature (.mat) files.
        """

        self = cls.__new__(cls)
        self.timer = utils.ElementTimer()
        self.initialize_features()

        # I'm not thrilled about this approach. I think we should
        # move the source specification into intialize_features
        all_specs = self.specs
        for key in all_specs:
            spec = all_specs[key]
            spec.source = 'mrc'

        # Load file reference for getting files from disk
        h = h5py.File(data_file_path, 'r')
        worm = h['worm']
        self.h = worm

        # Retrieve all features
        # Do we need to differentiate what we can and can not load?
        self._retrieve_all_features()

        return self

    @property
    def features(self):
        """
        We need to filter out temporary features and features that are
        not user requested
        """
        d = self._features
        return [
            d[x] for x in d if (
                x is not None and not d[x].is_temporary and d[x].is_user_requested)]

    def _retrieve_all_features(self):
        """
        Simple function for retrieving all features.
        """
        spec_dict = self.specs
        # Trying to avoid 2v3 differences in Python dict iteration
        for key in spec_dict:
            spec = spec_dict[key]
            # TODO: We could pass in the spec instance ...
            # rather than resolving the instance from the name
            
            try:
                
                self._get_and_log_feature(spec.name)
            except Exception as e:
                msg_warn = '{} was NOT calculated. {}'.format(spec.name, e)
                warnings.warn(msg_warn)
            
    def initialize_features(self):
        """
        Reads the feature specs and initializes necessary attributes.
        """

        f_specs = get_feature_specs(as_table=False)

        self.specs = \
            collections.OrderedDict([(value.name, value) for value in f_specs])

        self._features = collections.OrderedDict()

        # This will be removed soon
        self._temp_features = collections.OrderedDict()

    def get_features(self, feature_names):
        """
        This is the public interface to the user for retrieving a feature.

        Parameters
        ----------
        feature_names : {string, list, pandas.Series}

        Returns
        -------
        TODO: finish

        """
        if isinstance(
                feature_names,
                list) or isinstance(
                feature_names,
                pd.Series):
            output = []
            for feature_name in feature_names:
                self._get_and_log_feature(feature_name)
                output.append(self._features[feature_name])
        else:
            # Currently assuming a string
            self._get_and_log_feature(feature_names)
            output = self._features[feature_names]

        return output

    def _get_and_log_feature(self, feature_name, internal_request=False):
        """
        TODO: Update documentation. get_features is now the public interface

        A feature is returned if it has already been computed. If it has not
        been previously computed it is computed then returned.

        This function may become recursive if the feature being computed
        requires other features to be computed.

        Improvements
        ------------
        - retrieve multiple features, probably via a different method
        - allow passing in specs
        - have a feature that returns specs by regex or wildcard
        - have this call an internal requester that exposes the internal_request
        rather than exposing it to the user

        See Also
        --------
        FeatureProcessingSpec.get_feature
        """

        # Early return if already computed
        #----------------------------------
        if feature_name in self._features:
            # Check if it is being requested now via the user instead of
            # internally
            cur_feature = self._features[feature_name]
            if not cur_feature.is_user_requested and not internal_request:
                # If the logged feature is not currently user requested but
                # this request is from the user, then toggle the value
                #
                # otherwise we don't care, leave it as it currently is
                cur_feature.is_user_requested = True

            return cur_feature

        # Ensure that the feature name is valid
        #-------------------------------------------
        if feature_name in self.specs:
            spec = self.specs[feature_name]
        else:
            raise KeyError(
                'Specified feature name not found in the feature specifications')

        temp = spec.compute_feature(self, internal_request=internal_request)

        # TODO: this will change
        # A feature can return None, which means we can't ask the feature
        # what the name is, so we go based on the spec
        self._features[spec.name] = temp
        return temp

    def __repr__(self):
        return utils.print_object(self)

    def get_histograms(self, other_feature_sets=None):
        """

        TODO:
        - Create histogram manager
        -

        Improvements:
        - allow filtering of which features should be included

        """

        pass

    @staticmethod
    def get_feature_spec(extended=False, show_temp_features=False):
        """
        TODO: This method needs to be documented!

        I think this method is old. Jim and Michael need to talk about what
        this is trying to do and how to incorporate this into the current
        design. Most likely we would load via pandas and then have our current
        methods rely on processing of the loaded pandas data.

        Parameters
        ----------
        extended: boolean (default False)
            If True, return the full 726 features, not just the 93.
        show_temp_features: boolean
            If False, return only actual features.  Raises an exception
            if both show_temp_features and extended are True

        Returns
        ------------
        A pandas.DataFrame object
            Contains all the feature specs in one table

        """
        # Use pandas to load the features specification
        feature_spec_path = os.path.join('..', 'documentation',
                                         'database schema',
                                         'Features Specifications.xlsx')

        # Let's ignore a PendingDeprecationWarning here since my release of
        # pandas seems to be using tree.getiterator() instead of tree.iter()
        # It's probably fixed in the latest pandas release already
        # Here's an exmaple of the issue in a different repo, along with the
        # fix.  https://github.com/python-excel/xlrd/issues/104
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            excel_file = pd.ExcelFile(feature_spec_path)

        feature_spec = excel_file.parse('FeatureSpecifications')

        # I haven't bothered to work on the logic that I'd need
        # if extended == True and show_temp_features == True, so
        # let's just raise an exception in that case
        assert(not (extended and show_temp_features))

        if not show_temp_features:
            feature_spec = feature_spec[feature_spec['is_feature'] == 'y']

        #feature_spec = feature_spec[['remove_partial_events']].astype(bool)
        #feature_spec = feature_spec[['make_zero_if_empty']].astype(bool)

        if not extended:
            feature_spec = feature_spec.set_index('sub-extended feature ID')
            return feature_spec
        else:
            # Extend the 93 features into 726, then return that
            all_data_types = ['all', 'absolute', 'positive', 'negative']
            all_motion_types = ['all', 'forward', 'paused', 'backward']

            motion_types = pd.DataFrame({'motion_type':
                                         ['All'] + all_motion_types})
            motion_types['is_time_series'] = [False, True, True, True, True]

            data_types = pd.DataFrame({'data_type': ['All'] + all_data_types})
            data_types['is_signed'] = [False, True, True, True, True]

            # The effect of these two left outer joins is to duplicate any
            # feature rows where we have multiple data or motion types.
            # Thus the number of rows expands from 93 to 726
            feature_spec_expanded = feature_spec.merge(motion_types,
                                                       on='is_time_series',
                                                       how='left')

            feature_spec_expanded = feature_spec_expanded.merge(data_types,
                                                                on='is_signed',
                                                                how='left')

            feature_spec_expanded = \
                feature_spec_expanded.set_index('sub-extended feature ID',
                                                'motion_type', 'data_type')

            return feature_spec_expanded


def get_feature_specs(as_table=True):
    """

    Loads all specs that specify how features should be processed/created.

    Currently in /features/feature_metadata/features_list.csv

    Parameters
    ----------
    as_table : logical
        If true, returns a Pandas dataframe. Otherwise it returns a list of
        FeatureProcessingSpec objects.

    See Also
    --------
    FeatureProcessingSpec

    Returns
    -------
    a list of FeatureProcessingSpec

    """

    if as_table:
        #spec_df = pd.read_csv(FEATURE_SPEC_CSV_PATH,dtype={'is_final_feature':bool},true_values=['y'],false_values =['f'])
        # I don't like not specifying data types initially since I don't trust
        # them to guess correctly. TODO: Need to make things explicit
        # there are already some incorrect guesses
        df = pd.read_csv(FEATURE_SPEC_CSV_PATH)
        df.is_final_feature = df.is_final_feature == 'y'
        df['is_temporary'] = ~df.is_final_feature

        #self.flags = d['processing_flags']

        # This number conversion shouldn't have happened since I think it is
        # better to compare to '1'
        df.is_signed = df.is_signed == 1
        df.has_zero_bin = df.has_zero_bin == 1
        df.remove_partial_events = df.remove_partial_events == 1

        # Why didn't these convert to numbers then?
        df.make_zero_if_empty = df.make_zero_if_empty == '1'
        df.is_time_series = df.is_time_series == '1'

        return df

    else:
        f_specs = []

        with open(FEATURE_SPEC_CSV_PATH) as feature_metadata_file:
            feature_metadata = csv.DictReader(feature_metadata_file)

            for row in feature_metadata:
                f_specs.append(FeatureProcessingSpec(row))

        return f_specs


class FeatureProcessingSpec(object):
    """
    Information on how to get a feature.

    These are all loaded from a csv specificaton file. See the function
    get_feature_processing_specs which instatiates these specifications.

    Attributes
    ----------
    source :
        - new - from the normalized worm
        - mrc
    name : string
        Feature name
    module_name : string
        Name of the module that contains the executing code
    module : module
        A module instance
    class_name : string
        Name of the class which should be called to create the feature
    class_method : method
        A method instance
    flags : string
        This is a string that can be passed to the class method

    See Also
    --------
    get_feature_processing_specs

    """

    # This is how I am resolving a string to a module.
    # Perhaps there is a better way ...
    modules_dict = {'morphology_features': morphology_features,
                    'locomotion_features': locomotion_features,
                    'generic_features': generic_features,
                    'locomotion_bends': locomotion_bends,
                    'locomotion_turns': locomotion_turns,
                    'path_features': path_features,
                    'posture_features': posture_features}

    def __init__(self, d):
        """
        Parameters
        ----------
        d: dict
            Data in a row of the features file

        See Also
        --------
        get_feature_processing_specs

        """

        self.source = 'new'

        self.is_temporary = d['is_final_feature'] == 'n'
        self.name = d['feature_name']
        self.module_name = d['module']

        # TODO: Wrap this in a try clause with clear error if the module
        # hasn't been specified in the dictionary
        self.module_name = self.module_name

        # We won't store these so as to facilitate pickeling
        #-----------------------------------------------------
        #self.module = self.modules_dict[self.module_name]
        #self.class_method = getattr(self.module, self.class_name)

        self.class_name = d['class_name']

        # We retrieve the class constructor or function from the module

        self.flags = d['processing_flags']

        # TODO: We might write a __getattr__ function and just hold
        # onto the dict
        self.type = d['type']
        self.category = d['category']
        self.display_name = d['display_name']
        self.short_display_name = d['short_display_name']
        self.units = d['units']
        if self.is_temporary:
            self.bin_width = 1
        else:
            self.bin_width = float(d['bin_width'])

        self.is_signed = d['is_signed'] == '1'
        self.has_zero_bin = d['has_zero_bin'] == '1'
        self.signing_field = d['signing_field']
        self.remove_partial_events = d['remove_partial_events'] == '1'
        self.make_zero_if_empty = d['make_zero_if_empty'] == '1'
        self.is_time_series = d['is_time_series'] == '1'

    def compute_feature(self, wf, internal_request=False):
        """
        Note, the only caller of this function should be from:
            WormFeaturesDos._get_and_log_feature()

        This method takes care of the logic of retrieving a feature.

        ALL features are created or loaded via this method.

        Arguments
        ---------
        wf : WormFeaturesDos
            This is primarily needed to facilitate requesting additional
            features from the feature currently being computed

        """

        #print("feature: " + self.name)

        # Resolve who is going to populate the feature
        #--------------------------------------------
        #'source' should be overwritten by the feature loading method
        # after loading all the specs

        module = self.modules_dict[self.module_name]
        class_method = getattr(module, self.class_name)

        if self.source == 'new':
            final_method = class_method
        else:  # mrc #TODO: make explicit check for MRC otherwise throw an error
            final_method = getattr(class_method, 'from_schafer_file')

        
        timer = wf.timer
        timer.tic()

        # The flags input is optional, if no flag is present
        # we currently assume that the constructor doesn't require
        # the input
        if len(self.flags) == 0:
            temp = final_method(wf, self.name)
        else:
            # NOTE: All current flags are just a single string. We don't have
            # anything fancy in place for multiple parameters or for doing
            # any fancy parsing
            temp = final_method(wf, self.name, self.flags)

        elapsed_time = timer.toc(self.name)

        # This is an assigment of global attributes that the spec knows about
        # This could eventually be handled by a super() call to Feature
        # but for now this works. The only problem is that we override values
        # after the constructor rather than letting the constructor override
        # values from super()

        if temp is None:
            import pdb
            pdb.set_trace()

        # TODO: This is confusing for children features that rely on a
        # temporary parent. Unfortunately this is populated after
        # the feature has been computed.
        # We could allow children to copy the value from the parent but
        # then we would need to check for that here ...
        temp.computation_time = elapsed_time

        # We can get rid of the name assignments in class and use this ...
        temp.name = self.name
        temp.is_temporary = self.is_temporary
        temp.spec = self
        temp.is_user_requested = not internal_request

        # Problem features
        #--------------------------
        if not hasattr(temp, 'missing_from_disk'):
            temp.missing_from_disk = False

        if not hasattr(temp, 'missing_dependency'):
            temp.missing_dependency = False

        if not hasattr(temp, 'empty_video'):
            temp.empty_video = False

        if not hasattr(temp, 'no_events'):
            temp.no_events = False

        return temp

    def __repr__(self):
        return utils.print_object(self)

    def copy(self):
        # Not sure if I'll need to do anything here ...
        return copy.copy(self)
