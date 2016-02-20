# -*- coding: utf-8 -*-
"""
This example examines only processing a subset of features. This may be useful
for users that are only concerned about trying to match specific features. It can
also save time.
"""

import sys
import os

# TODO: This seems like a bold move. Do we really want to do this. My guess is
# that we shouldn't . We need to decide how we want to handle this for all
# examples
sys.path.append('..')

import open - worm - analysis - toolbox as mv


def main():
    # Set up the necessary file paths for file loading
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    data_file_path = os.path.join(base_path, "example_video_norm_worm.mat")

    # Load the normalized worm from file
    nw = mv.NormalizedWorm.from_schafer_file_factory(data_file_path)

    # Generate the OpenWorm version of the features
    fpo = mv.FeatureProcessingOptions()
    fpo.disable_feature_sections(['morphology'])
    openworm_features = mv.WormFeatures(nw, fpo)

    openworm_features.timer.summarize()


if __name__ == '__main__':
    main()
