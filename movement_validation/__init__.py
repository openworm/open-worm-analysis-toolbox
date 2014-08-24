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
from .features.WormFeatures import WormFeatures
from .WormPlotter import WormPlotter

#from .features.feature_comparisons import fp_isequal
#from .features.feature_comparisons import corr_value_high
#from .WormPlotter import plot_frame_codes

__all__ = ['SchaferExperimentFile',
           'NormalizedWorm',
           'WormFeatures',
           'WormPlotter']
