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


A translation of Matlab code written by Jim Hokanson, in the 
SegwormMatlabClasses GitHub repo.  

Original code path:
SegwormMatlabClasses/+seg_worm/@feature_calculator/features.m

"""

import csv, os, warnings
import h5py  # For loading from disk
import numpy as np
import collections  # For namedtuple
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
        self.area_per_length = utils._extract_time_from_disk(m_var, 'areaPerLength')
        self.width_per_length = utils._extract_time_from_disk(m_var, 'widthPerLength')

        return self

    def __eq__(self, other):

        return \
            utils.correlation(self.length, other.length, 'morph.length')  and \
            self.width == other.width and \
            utils.correlation(self.area, other.area, 'morph.area')      and \
            utils.correlation(self.area_per_length, other.area_per_length, 'morph.area_per_length') and \
            utils.correlation(self.width_per_length, other.width_per_length, 'morph.width_per_length')

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
        
        nw  = features_ref.nw
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

        #The order here matches the order the properties are populated
        #in the constructor
        same_locomotion = True

        if not (self.velocity == other.velocity):
            same_locomotion = False

        if not (self.motion_events == other.motion_events):
            same_locomotion = False

        # Test motion codes
        if not utils.correlation(self.motion_mode, other.motion_mode, 
                                  'locomotion.motion_mode'):
            same_locomotion = False

        #TODO: Define ne for all functions (instead of needing not(eq))
        if not (self.crawling_bends == other.crawling_bends):
            print('Mismatch in locomotion.crawling_bends events')
            same_locomotion = False
            
        if not (self.foraging_bends == other.foraging_bends):
            print('Mismatch in locomotion.foraging events')
            same_locomotion = False

        #TODO: Make eq in events be an error - use test_equality instead    
        #NOTE: turns is a container class that implements eq, and is not
        #an EventList    
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

        #TODO: I'd rather this be a formal class
        self.skeleton = posture_features.Skeleton(features_ref)

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

        #TODO: It would be nice to see all failures before returning false
        #We might want to make a comparison class that handles these details 
        #and then prints the results

        #Doing all of these comparisons and then computing the results
        #allows any failures to be printed, which at this point is useful for 
        #getting the code to align

        #Note that the order of these matches the order in which they are 
        #populated in the constructor
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
        
        
        #TODO: We need a more lazy evaluation for these since they don't match
        #Are they even close?
        #We could provide a switch for exactly equal vs mimicing the old setup
        #in which our goal could be to shoot for close
        eq_track_length = utils.correlation(self.track_length, 
                                             other.track_length, 
                                             'posture.track_length')
        eq_eccentricity = utils.correlation(self.eccentricity, 
                                             other.eccentricity, 
                                             'posture.eccentricity',
                                             high_corr_value=0.99)
        eq_kinks = utils.correlation(self.kinks, other.kinks, 
                                      'posture.kinks')
        
        eq_coils = self.coils.test_equality(other.coils,'posture.coils')       
        eq_directions = self.directions == other.directions
        eq_skeleton = self.skeleton == other.skeleton
        eq_eigen_projection = \
            utils.correlation(np.ravel(self.eigen_projection), 
                               np.ravel(other.eigen_projection), 
                               'posture.eigen_projection')
        
        
        #TODO: Reorder these as they appear above
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
        self.duration = path_features.Duration(features_ref)

        self.coordinates = path_features.Coordinates(features_ref)

        #Curvature
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

        #Make a call to utils loader
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
    WormFeatures: Takes a NormalizedWorm instance and
    during initialization calculates all the features of the worm.

    There are two ways to initialize a WormFeatures instance: 
    1. by passing a NormalizedWorm instance and generating the features, or
    2. by loading the already-calculated features from an HDF5 file.
         (via the from_disk method)
         
    Attributes
    ----------      
    video_info: VideoInfo object
    options: movement_validation.features.feature_processing_options
    nw: NormalizedWorm object
    morphology: WormMorphology object
    locomotion: WormLocomotion object
    posture: WormPosture object
    path: WormPath object

    """

    def __init__(self, nw, processing_options=None):
        """
        
        Parameters
        ----------
        nw: NormalizedWorm object
        processing_options: movement_validation.features.feature_processing_options

        """
        if processing_options is None:
            processing_options = \
                            fpo.FeatureProcessingOptions()

        # These are saved locally for reference by others when processing
        self.video_info = nw.video_info
        
        self.options = processing_options
        self.nw = nw
        self.timer = utils.ElementTimer()
        
        self.morphology = WormMorphology(self)
        self.locomotion = WormLocomotion(self)
        self.posture = \
            WormPosture(self, self.locomotion.velocity.get_midbody_distance())
        self.path = WormPath(self)

    @classmethod
    def from_disk(cls, file_path):

        """
        This from disk method is currently focused on loading the features
        files as computed by the Schafer lab. Alternative loading methods
        should be possible 
        """
        h = h5py.File(file_path, 'r')
        worm = h['worm']

        self = cls.__new__(cls)

        self.morphology = WormMorphology.from_disk(worm['morphology'])
        self.locomotion = WormLocomotion.from_disk(worm['locomotion'])
        self.posture = WormPosture.from_disk(worm['posture'])
        self.path = WormPath.from_disk(worm['path'])

        return self

    @property
    def feature_spec(self):
        """
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

        return feature_spec


    def get_DataFrame(self):
        """
        Returns
        ------------
        A pandas.DataFrame object
            Contains all the feature data in one table
            
        """
        feature_dataframe = pd.DataFrame(columns=['sub-extended feature ID',
                                                  'data_array'])

        
        #feature_dataframe = \
        #    pd.DataFrame({'sub-extended feature ID': pd.Series(),
        #                  'data_array': pd.Series()})
        
        #feature_dataframe = \
        #    pd.DataFrame({'sub-extended feature ID': pd.Series(dtype=int),
        #                  'data_array': pd.Series(dtype=object)})
        
        for index, row in self.feature_spec.iterrows():
            sub_extended_feature_ID = row['sub-extended feature ID']
            nested_attributes = row['feature_field'].split('.')
                
           
            attribute = self
            for attribute_name in nested_attributes:
                try:
                    attribute = getattr(attribute, attribute_name)
                except AttributeError:
                    import pdb
                    pdb.set_trace()

            feature_dataframe.loc[index] = [sub_extended_feature_ID, attribute]

        # Add a column just for length    
        def length(x):
            if x is None:
                return np.NaN
            try:
                return len(x)
            except TypeError:
                # e.g. len(5) will raise TypeError
                if isinstance(x, (int, float, complex)):
                    return 1
                else:
                    return np.NaN
    
        feature_dataframe['length'] = \
                            feature_dataframe['data_array'].map(length)
    
        pd.set_option('display.max_rows', 500)
        
        # Left Outer Join: add columns for feature_field, feature_type.
        fs = self.feature_spec[['sub-extended feature ID',
                                'feature_field',
                                'feature_type']]
        feature_dataframe = \
                feature_dataframe.merge(fs, how='left', 
                                        on=['sub-extended feature ID'])
    
        return feature_dataframe

    def get_movement_DataFrame(self):
        """
        Just the movement features, but pivoted
        so the rows are frames and the columns are the feature names

        NOTE: This only works for movement features.  
        
        """
        feature_df = self.get_DataFrame()
        movement_df = feature_df[feature_df.feature_type == 'movement']

        # All movement features must have same length
        assert(np.all(movement_df.length == movement_df.length.iloc[0]))    
    
        # Pivot so rows are frames, and features are columns.
        movement_data = np.array(movement_df.data_array.as_matrix().tolist())
    
        # Just the movement features, but pivoted
        # so the rows are frames and the columns are the feature names
        # shape (4642, 53)
        movement_df_pivoted = pd.DataFrame(movement_data.transpose(), 
                                           columns=movement_df.feature_field)
        
        return movement_df_pivoted

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        """
        Compare two WormFeatures instances by value

        """
        same_morphology = self.morphology == other.morphology
        same_locomotion = self.locomotion == other.locomotion
        same_posture = self.posture == other.posture
        same_path = self.path == other.path
        return same_morphology and \
             same_locomotion and \
             same_posture and \
             same_path
                      

class WormFeaturesDos(object):

    """
    This is the new features class. It will eventually replace the old class
    when things are all ready.
    """
    def __init__(self, nw, processing_options=None):
        """
        
        Parameters
        ----------
        nw: NormalizedWorm object
        processing_options: movement_validation.features.feature_processing_options

        """
        if processing_options is None:
            processing_options = \
                            fpo.FeatureProcessingOptions()

        # These are saved locally for reference by others when processing
        self.video_info = nw.video_info
        
        self.options = processing_options
        self.nw = nw
        self.timer = utils.ElementTimer()    

        #Old code
        #------------------
        #self.morphology = WormMorphology(self)
        #self.locomotion = WormLocomotion(self)
        #self.posture = \
        #    WormPosture(self, self.locomotion.velocity.get_midbody_distance())
        #self.path = WormPath(self)

        f_specs = get_feature_processing_specs()

        self.features = {}

        #TODO: Move this to the spec class ...
        modules = {'morphology_features':morphology_features,
        'locomotion_features':locomotion_features,
        'generic_features':generic_features,
        'locomotion_bends':locomotion_bends} 

        self.feature_list = []
        for spec in f_specs:
            #Some of this logic should move to the specs themselves
            module = modules[spec.module_name]

            method_to_call = getattr(module,spec.class_name)

            if len(spec.flags) == 0:
                temp = method_to_call(self)
            else:
                temp = method_to_call(self,spec.flags)

                
            self.feature_list.append(temp)
            self.features[temp.name] = temp

        #Wanted order, didn't feel like messsing with ordered_dict
        #This will all likely change
        #self.feature_list = [v for k,v in self.features.items()]

        import pdb
        pdb.set_trace()

    def __getitem__(self,key):
        #TODO: We should add on error checking here ...
        return self.features[key]

    def __repr__(self):
        return utils.print_object(self)    
        
def get_feature_processing_specs():
    
    csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'feature_metadata',
                                'features_list.csv')        
    
    f_specs = []    
    
    with open(csv_path) as feature_metadata_file:
        feature_metadata = csv.DictReader(feature_metadata_file)
        
        for row in feature_metadata: 
            f_specs.append(FeatureProcessingSpec(row))
            
    return f_specs


class FeatureProcessingSpec(object):
    
    """
    Information on how to get the feature
    """
    def __init__(self,d):
        """
        Parameters
        ----------
        d: dict
            Data in a row of the features_list file
        """
        self.name = d['Feature Name']
        self.module_name = d['Module']
        self.class_name = d['Class Name']
        
        #We don't really need the dependencies if we lazy load ...
        temp = d['Dependencies']
        
        #I'm not sure how I want to handle this yet
        self.flags = d['Flags']
        
        
        #TODO: Build module retrieval and code into here
        
    def __repr__(self):
        return utils.print_object(self)       