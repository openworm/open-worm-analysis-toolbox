## Tools used ##

**Language:** Python 3.x.  The code requires use of scientific computing packages (numpy, h5py), and as such getting the packages properly installed can be tricky. As such, if working in Windows, we recommend using [Spyder IDE](https://code.google.com/p/spyderlib/) and the [WinPython distribution](http://winpython.sourceforge.net/) for Windows.  ( (Note, this isn't required)

**HDF reader:** [HDF viewer](http://www.hdfgroup.org/hdf-java-html/hdfview/) - this can be used for debugging the file structure

**Plotting:** matplotlib is a plotting library for the Python programming language and its NumPy numerical mathematics extension.  FFMPEG is used for video processing.

**File processing:** The Schafer Lab chose to structure their experiment files using the  “Heirarchical Data Format, Version 5” (HDF5) format , ending with the extension .MAT.  We are using the Python module H5PY to extract the information from the Schafer Lab files.

**Markdown editor:** http://markdownpad.com/

**Data repository:** We store the .mat files that flow between the steps of our pipeline, in a DropBox shared folder "worm_data".


##Installing and running the movement_validation repository##

1. Install Python 3.x and matplotlib
2. numpy is a library for Python that is a dependency in this repo.  If you have numpy version less than 1.8, you will need to either install it, or:
     a. Save [nanfunctions.py](https://github.com/numpy/numpy/blob/0cfa4ed4ee39aaa94e4059c6394a4ed75a8e3d6c/numpy/lib/nanfunctions.py) to your Python library directory, in the `numpy/lib/` directory, and
     b. Append the lines ```__all__ += nanfunctions.__all__``` and ```from .nanfunctions import *``` at the relevant places to numpy/lib/```__init__.py```.
3. Install Shapely, which is available [for Windows here](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
4. Clone the repository to your computer
5. Ask @MichaelCurrie or @JimHokanson to share the worm_data DropBox folder with you
6. In the `movement_validation/movement_validation` directory there should be a file `user_config_example.txt`.  Rename this file as `user_config.py`.  It will be ignored by github since it is in the .gitignore file.  So in `user_config.py`, specify your computer's specific DropBox folder root directory and other settings.
7. Try running one of the scripts in the `examples` folder.
8. Hopefully it runs successfully!  If not:

Please contact the [OpenWorm-discuss mailing list](https://groups.google.com/forum/#!forum/openworm-discuss) if you encounter issues with the above steps.

