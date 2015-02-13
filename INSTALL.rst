Tools used
----------

**Language:** Python 3.x. The code requires use of scientific computing
packages (numpy, h5py), and as such getting the packages properly
installed can be tricky. As such, if working in Windows, we recommend
using `Spyder IDE <https://code.google.com/p/spyderlib/>`__ and the
`WinPython distribution <http://winpython.sourceforge.net/>`__ for
Windows. (Note, this isn't required)

**Plotting:** matplotlib is a plotting library for the Python
programming language and its NumPy numerical mathematics extension.
FFMPEG is used for video processing.

**File processing:** The Schafer Lab chose to structure their experiment
files using the “Heirarchical Data Format, Version 5” `(HDF5)<http://en.wikipedia.org/wiki/Hierarchical_Data_Format#HDF5>`__ format ,
ending with the extension .MAT. We are using the Python module H5PY to
extract the information from these files.

**Data storage:** Google Drive. To store examples of worm videos and
HDF5 (.mat) feature files so the movement\_validation package can be put
through its paces.

**Markdown editor:** `MarkdownPad <http://markdownpad.com/>`__.
Optional. For editing documents like this one.

**HDF reader:** `HDF
viewer <http://www.hdfgroup.org/hdf-java-html/hdfview/>`__. Optional.
This tool can be used for debugging the file structure of the data
files.

Installing and running the movement\_validation repository
----------------------------------------------------------

1.  Install Python 3.x, matplotlib, and Cython.
2.  If it's not already included with your Python installation, install
    numpy. Ideally use version 1.8 or greater. Otherwise, if you have
    numpy version less than 1.8, you will need to:

    1. Save
       `nanfunctions.py <https://github.com/numpy/numpy/blob/0cfa4ed4ee39aaa94e4059c6394a4ed75a8e3d6c/numpy/lib/nanfunctions.py>`__
       to your Python library directory, in the ``numpy/lib/``
       directory, and
    2. Append the lines ``__all__ += nanfunctions.__all__`` and
       ``from .nanfunctions import *`` at the relevant places to
       ``numpy/lib/__init__.py``.

3.  Install Shapely:

    a. Windows: `here <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__.
    b. OS X: Install geos (e.g. brew install geos), followed by shapely
       (e.g. pip install shapely).

4.  Clone this GitHub repository to your computer.
5.  If you don't already have an account, get a `Google
    Drive <https://www.google.com/intl/en/drive/>`__ account.
6.  Install `Google Drive for
    desktop <https://tools.google.com/dlpage/drive>`__.
7.  Using Google Drive, sync with the folder
    `example_movement_validation_data/ <https://drive.google.com/folderview?id=0B7to9gBdZEyGNWtWUElWVzVxc0E&usp=sharing>`__,
    which is a subfolder of
    ``OpenWorm/OpenWorm Public/movement_validation/``.
8.  In the ``movement_validation/movement_validation`` folder there
    should be a file ``user_config_example.txt``. Rename this file as
    ``user_config.py``. It will be ignored by GitHub since it is in the
    ``.gitignore`` file. So in ``user_config.py``, specify your
    computer's specific Google Drive root directory and other settings.
9.  Try running one of the scripts in the ``examples/`` folder.
10. Hopefully it runs successfully! If not:

Please contact the `OpenWorm-discuss mailing
list <https://groups.google.com/forum/#!forum/openworm-discuss>`__ if
you encounter issues with the above steps.

You can also try running ``test_setup.py`` in the ``/tools`` folder to
check if your setup is correctly configured. Note that this tool is not
complete.
