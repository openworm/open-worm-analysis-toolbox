"""
open-worm-analysis-toolbox: A Python library
https://github.com/openworm/open-worm-analysis-toolbox

Takes raw videos of C. elegans worms and processes them into features and
statistics.

The purpose is to be able to compare the behavior of worms statistically,
and in particular to validate how closely the behaviour of OpenWorm's worm
simulation is to the behaviour of real worms.

License
---------------------------------------
https://github.com/openworm/open-worm-analysis-toolbox/LICENSE.md

"""
from .version import __version__

from .prefeatures.video_info import VideoInfo
from .prefeatures.basic_worm import BasicWorm
from .prefeatures.normalized_worm import NormalizedWorm
from .prefeatures.worm_plotter import NormalizedWormPlottable

# This is temporary; we will eventually remove it when the code is ready
# to become WormFeatures
from .features.worm_features import WormFeatures
from .features.worm_features import get_feature_specs
from .features import feature_manipulations

from .features.worm_features import WormFeatures
from .features.feature_processing_options import FeatureProcessingOptions

from .statistics.histogram_manager import HistogramManager
from .statistics.statistics_manager import StatisticsManager
from .statistics.histogram import Histogram, MergedHistogram

# JAH: Putting this on hold for now 2016-02-17
#from .statistics.pathplot import *

try:
    from . import user_config
except ImportError:
    raise Exception(
        "user_config.py not found. Copy the "
        "user_config_example.txt in the 'open-worm-analysis-toolbox' "
        "package to user_config.py in the same directory and "
        "edit the values")


__all__ = ['__version__',
           'BasicWorm',
           'NormalizedWorm',
           'VideoInfo',
           'WormFeatures',
           'FeatureProcessingOptions',
           'NormalizedWormPlottable',
           'HistogramManager',
           'StatisticsManager',
           'Histogram',
           'MergedHistogram']
