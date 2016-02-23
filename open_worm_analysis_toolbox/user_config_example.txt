#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
user_config.py: a module to specify installation-specific settings for
                the open-worm-analysis-toolbox repository
  
This file is in the .gitignore file for the open-worm-analysis-toolbox repo, so
you can put configuration settings here that apply to just your computer.
  
"""

EXAMPLE_DATA_PATH = r'/home/ubuntu/example_data'


# There is a step that is very slow that can be disabled
# if you are just debugging other parts of the code.
# For production, this should be True.
PERFORM_SLOW_ECCENTRICITY_CALC = True
