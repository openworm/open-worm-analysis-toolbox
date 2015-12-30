"""
Run this script to verify that:
1) All necessary packages are installed
2) The packages are sufficiently recent
3) User options are specified correctly



Jim's Note: I've found managing python packages, particularly on Windows, to
be frustratingly complex. It is possible that this process could be enhanced 
but the main goal of this script is to verify the setup rather than automate
the setup. That being said, help with automating the setup would be appreciated.

Links to windows installs for required packages
-----------------------------------------------
http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib
http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy

Manual user config specification
--------------------------------
Some settings must be set manually by the user. To do this make a copy of the
user_config_example.txt file in the open-worm-analysis-toolbox folder and rename
it in the same folder as user_config.py

Replace all values in the this file with their appropriate values.


"""

#TODO: This file should look for all package dependencies and setup issues
#and provide detailed information as to how to handle things when there are problems

import sys, os


"""
=====================     Python version testing    ===========================
"""

print("Python version: " + sys.version)

if sys.version_info[:2] < (2,7):
    raise Exception("open-worm-analysis-toolbox requires Python version 2.7 or greater")
else:
    print("The version used is acceptable, since it is >=2.7")


"""
=====================     Package Testing      ================================
"""


"""
Shapely
-------
Shapely is currently only used for calculating the worm eccentricity. More
specifically, it is only used for solving the point-in-polygon problem.

    http://en.wikipedia.org/wiki/Point_in_polygon
"""
try:
    import shapely
except ImportError as e:
    print("'Unable to import shapely")

    

"""
h5py
----
h5py can be used to read and write hdf5 files. It is currently used to load
old feature files from disk
"""
try:
    import h5py
except ImportError as e:
    print('Unable to import h5py')    


"""
numpy
-----

We use numpy for everything ...

"""

try:
    import numpy as np
except ImportError as e:
    print('Unable to import numpy')   
    
#TODO: test nan functionality

"""
scipy
-----

??? Why do we need scipy?

"""

try:
    import scipy
except ImportError as e:
    print('Unable to import scipy')   

"""
=====================     Configuration testing    ============================
"""

try:
    # We must add .. to the path so that we can perform the 
    # import of open-worm-analysis-toolbox while running this as 
    # a top-level script (i.e. with __name__ = '__main__')
    sys.path.append('..') 
    from open-worm-analysis-toolbox import user_config
except ImportError as e:
    print('Unable to import open-worm-analysis-toolbox/user_config.py module')
    
if not hasattr(user_config,'EXAMPLE_DATA_PATH'):
    print("user_config.py module is missing the 'EXAMPLE_DATA_PATH' attribute")
else:
    if not os.path.isdir(user_config.EXAMPLE_DATA_PATH):
       print("user_config.EXAMPLE_DATA_PATH doesn't point to a valid directory")
       
       
print('Finished running test_setup.py')