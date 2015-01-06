"""
movement_validation: A Python library
https://github.com/openworm/movement_validation

Takes raw videos of C. elegans worms and processes them into features and
statistics.

The purpose is to be able to compare the behavior of worms statistically,
and in particular to validate how closely the behaviour of OpenWorm's worm 
simulation is to the behaviour of real worms.

License
---------------------------------------
https://github.com/openworm/movement_validation/LICENSE.md

"""
from .SchaferExperimentFile import SchaferExperimentFile
from .NormalizedWorm import NormalizedWorm
from .video_info import VideoInfo
from .features.worm_features import WormFeatures
from .WormPlotter import WormPlotter

from .features.feature_processing_options import FeatureProcessingOptions

from .statistics.histogram_manager import HistogramManager
from .statistics.manager import StatisticsManager


__all__ = ['SchaferExperimentFile',
           'NormalizedWorm',
		   'VideoInfo',
           'WormFeatures',
           'FeatureProcessingOptions',
           'WormPlotter']
