# -*- coding: utf-8 -*-
"""
Demonstrates loading and saving to a JSON file

"""
import sys
import os
import warnings

# We must add .. to the path so that we can perform the
# import of open-worm-analysis-toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
from open_worm_analysis_toolbox import user_config, BasicWorm, NormalizedWorm


def main():
    warnings.filterwarnings('error')

    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    JSON_path = os.path.join(base_path, 'test.JSON')

    b = BasicWorm()
    #b.contour[0] = 100.2
    #b.metadata['vulva'] = 'CCW'
    b.save_to_JSON(JSON_path)

    c = BasicWorm()
    c.load_from_JSON(JSON_path)
    print(c.contour)

    # dat.save_to_JSON(JSON_path)

if __name__ == '__main__':
    main()
