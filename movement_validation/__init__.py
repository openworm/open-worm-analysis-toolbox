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
from .normalized_worm import NormalizedWorm
from .video_info import VideoInfo
from .features.worm_features import WormFeatures
from .worm_plotter import NormalizedWormPlottable
from .basic_worm import BasicWorm

from .features.feature_processing_options import FeatureProcessingOptions

from .statistics.histogram_manager import HistogramManager
from .statistics.manager import StatisticsManager

try:
    from . import user_config
except ImportError:
     raise Exception("user_config.py not found. Copy the "
                     "user_config_example.txt in the 'movement_validation' "
                     "package to user_config.py in the same directory and "
                     "edit the values")


__all__ = ['BasicWorm',
           'NormalizedWorm',
		'VideoInfo',
           'WormFeatures',
           'FeatureProcessingOptions',
           'NormalizedWormPlottable',
           'HistogramManager',
           'StatisticsManager']
