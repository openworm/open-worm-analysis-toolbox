# -*- coding: utf-8 -*-
"""
Support functions used in other modules.

Several, but not all, of the functions here are defined so we have functions
that are the equivalent of the same-named functions in Matlab, e.g. gausswin

"""
from __future__ import division

from itertools import groupby
import csv
import matplotlib.pyplot as plt
import numpy as np

__ALL__ = ['scatter',
           'plotxy',
           'plotx',
           'imagesc',
           'separated_peaks',
           'gausswin',
           'colon',
           'print_object'
           'write_to_CSV',
           'interpolate_with_threshold',
           'interpolate_with_threshold_2D',
           'gausswin',
           '_extract_time_from_disk']


def scatter(x, y):
    plt.scatter(x, y)
    plt.show()


def plotxy(x, y):
    plt.plot(x, y)
    plt.show()


def plotx(data):
    plt.plot(data)
    plt.show()


def imagesc(data):
    # http://matplotlib.org/api/pyplot_api.html?  ...
    # highlight=imshow#matplotlib.pyplot.imshow
    plt.imshow(data, aspect='auto')
    plt.show()

def find(data,n=None):

    """
    Similar to Matlab's find() function
    TODO: Finish documentation
    """
    temp = np.flatnonzero(data)
    if n is not None:
        if n >= temp.size:
            return temp
        else:
            return temp[0:n]
    else:
        return temp

def separated_peaks(x, dist, use_max, value_cutoff):
    """
    Find the peaks (either minimum or maximum) in an array. 
    The peaks must be separated by, at least, the given distance.

    Note that outputs are not sorted.


    Parameters
    ---------------------------------------    
    x: numpy array
      The values to be searched for peaks

    dist
      The minimum distance between peaks

    use_max: boolean
      True: find the maximum peaks
      False: find the minimum peaks

    chainCodeLengths
      The chain-code length at each index;
      if empty, the array indices are used instead


    Returns
    ---------------------------------------    
    peaks   
      The maximum peaks

    indices 
      The indices for the peaks


    Notes
    ---------------------------------------    
    Formerly seg_worm.util.maxPeaksDist
    i.e. [PEAKS INDICES] = seg_worm.util.maxPeaksDist \
        (x, dist,use_max,value_cutoff,*chain_code_lengths)
    i.e. https://github.com/JimHokanson/SegwormMatlabClasses/ ...
              blob/master/%2Bseg_worm/%2Butil/maxPeaksDist.m

    Used in seg_worm.feature_helpers.posture.getAmplitudeAndWavelength
    Used in locomotion_bends.py

    See also MINPEAKSDIST, COMPUTECHAINCODELENGTHS

    """

    chain_code_lengths = colon(1, 1, x.size)

    # Is the vector larger than the search window?
    winSize = 2 * dist + 1
    if chain_code_lengths[-1] < winSize:
        temp_I = np.argmax(x)
        return (x[temp_I], temp_I)

    # xt - "x for testing" in some places in the code below
    # it will be quicker (and/or easier) to assume that we want the largest
    # value. By negating the data we can look for maxima (which will tell us
    # where the minima are)
    if not use_max:
        xt = -1 * x
    else:
        xt = x

    # NOTE: I added left/right neighbor comparisions which really helped with
    # the fft ..., a point can't be a peak if it is smaller than either of its
    # neighbors

    np_true = np.ones(1, dtype=np.bool)
    if use_max:
        #                                           %> left                     > right
        # Matlab version:
        # could_be_a_peak = x > value_cutoff & [true x(2:end) > x(1:end-1)] & [x(1:end-1) > x(2:end) true];


        #                                        greater than values to the left
        could_be_a_peak = (x > value_cutoff) & np.concatenate((np_true, x[1:] > x[0:-1]))
        #could_be_a_peak = np.logical_and(
        #    x > value_cutoff, np.concatenate((np.ones(1, dtype=np.bool), x[1:] > x[0:-1])))

        #                                        greater than values to the right
        could_be_a_peak = could_be_a_peak & np.concatenate((x[0:-1] > x[1:], np_true))

        #could_be_a_peak = np.logical_and(
        #    could_be_a_peak, np.concatenate((x[0:-1] > x[1:], np.ones(1, dtype=np.bool))))

        I1 = could_be_a_peak.nonzero()[0]
        I2 = np.argsort(-1 * x[I1])  # -1 => we want largest first
        I = I1[I2]
    else:
        #raise Exception("Not yet implemented")
        # pdb.set_trace()
        # could_be_a_peak = x < value_cutoff & [true x(2:end) < x(1:end-1)] & [x(1:end-1) < x(2:end) true];
        #I1     = find(could_be_a_peak);
        #[~,I2] = sort(x(I1));
        #I = I1(I2);

        could_be_a_peak = (x < value_cutoff) & np.concatenate((np_true, x[1:] < x[0:-1]))
        could_be_a_peak = could_be_a_peak & np.concatenate((x[0:-1] < x[1:], np_true))

        I1 = could_be_a_peak.nonzero()[0]
        I2 = np.argsort(x[I1])
        I = I1[I2]

    n_points = x.size

    # This code would need to be fixed if real distances
    # are input ...
    too_close = dist - 1

    temp_I = colon(0, 1, n_points - 1)
    start_I = temp_I - too_close  # Note, separated by dist is ok
    # This sets the off limits area, so we go in by 1
    end_I = temp_I + too_close

    start_I[start_I < 0] = 0
    end_I[end_I > n_points] = n_points

    is_peak_mask = np.zeros(n_points, dtype=bool)
    # A peak and thus can not be used as a peak

    for cur_index in I:
        # NOTE: This gets updated in the loop so we can't just iterate
        # over these values
        if could_be_a_peak[cur_index]:
            # NOTE: Even if a point isn't the local max, it is greater
            # than anything that is by it that is currently not taken
            # (because of sorting), so it prevents these points
            # from undergoing the expensive search of determining
            # whether they are the min or max within their
            # else from being used, so we might as well mark those indices
            # within it's distance as taken as well
            temp_indices = slice(start_I[cur_index], end_I[cur_index])
            could_be_a_peak[temp_indices] = False

            # This line is really slow ...
            # It would be better to precompute the max within a window
            # for all windows ...
            is_peak_mask[cur_index] = np.max(xt[temp_indices]) == xt[cur_index]

    indices = is_peak_mask.nonzero()[0]
    peaks = x[indices]

    return (peaks, indices)


def colon(r1, inc, r2):
    """
      Matlab's colon operator, althought it doesn't although inc is required

    """

    s = np.sign(inc)

    if s == 0:
        return np.zeros(1)
    elif s == 1:
        n = ((r2 - r1) + 2 * np.spacing(r2 - r1)) // inc
        return np.linspace(r1, r1 + inc * n, n + 1)
    else:  # s == -1:
        # NOTE: I think this is slightly off as we start on the wrong end
        # r1 should be exact, not r2
        n = ((r1 - r2) + 2 * np.spacing(r1 - r2)) // np.abs(inc)
        temp = np.linspace(r2, r2 + np.abs(inc) * n, n + 1)
        return temp[::-1]


def print_object(obj):
    """ 
    Goal is to eventually mimic Matlab's default display behavior for objects 

    Example output from Matlab    

    morphology: [1x1 seg_worm.features.morphology]
       posture: [1x1 seg_worm.features.posture]
    locomotion: [1x1 seg_worm.features.locomotion]
          path: [1x1 seg_worm.features.path]
          info: [1x1 seg_worm.info]

    """

    # TODO - have some way of indicating nested function and not doing fancy
    # print for nested objects ...

    MAX_WIDTH = 70

    dict_local = obj.__dict__

    key_names = [x for x in dict_local]
    key_lengths = [len(x) for x in key_names]

    if len(key_lengths) == 0:
        return ""

    max_key_length = max(key_lengths)
    key_padding = [max_key_length - x for x in key_lengths]

    max_leadin_length = max_key_length + 2
    max_value_length = MAX_WIDTH - max_leadin_length

    lead_strings = [' ' * x + y + ': ' for x, y in zip(key_padding, key_names)]

    # TODO: Alphabatize the results ????
    # Could pass in as a option
    # TODO: It might be better to test for built in types
    #   Class::Bio.Entrez.Parser.DictionaryElement
    #   => show actual dictionary, not what is above

    value_strings = []
    for key in dict_local:
        value = dict_local[key]
        try:  # Not sure how to test for classes :/
            class_name = value.__class__.__name__
            module_name = inspect.getmodule(value).__name__
            temp_str = 'Class::' + module_name + '.' + class_name
        except:
            temp_str = repr(value)
            if len(temp_str) > max_value_length:
                #type_str = str(type(value))
                #type_str = type_str[7:-2]
                try:
                    len_value = len(value)
                except:
                    len_value = 1
                temp_str = str.format(
                    'Type::{}, Len: {}', type(value).__name__, len_value)

        value_strings.append(temp_str)

    final_str = ''
    for cur_lead_str, cur_value in zip(lead_strings, value_strings):
        final_str += (cur_lead_str + cur_value + '\n')

    return final_str




def write_to_CSV(data_dict, filename):
    """
    Writes data to a CSV file, by saving it to the directory os.getcwd()

    Parameters
    ---------------------------------------
    data_dict: a dictionary of 1-dim ndarrays of dtype=float
      What is to be written to the file.  data.keys() provide the headers,
      and each column in turn is provided by the value for that key
    filename: string
      Name of file to be saved (not including the '.csv' part of the name)

    """
    csv_file = open(filename + '.csv', 'w')
    writer = csv.writer(csv_file, lineterminator='\n')

    # The first row of the file is the keys
    writer.writerow(list(data_dict.keys()))

    # Find the maximum number of rows across all our columns:
    max_rows = max([len(x) for x in list(data_dict.values())])

    # Combine all the dictionary entries so we can write them
    # row-by-row.
    columns_to_write = []
    for column_key in data_dict.keys():
        column = list(data_dict[column_key])
        # Create a mask that shows True for any unused "rows"
        m = np.concatenate([np.zeros(len(column), dtype=bool),
                            np.ones(max_rows - len(column), dtype=bool)])
        # Create a masked array of size max_rows with unused entries masked
        column_masked = np.ma.array(np.resize(column, max_rows), mask=m)
        # Convert the masked array to an ndarray with the masked values
        # changed to NaNs
        column_masked = column_masked.filled(np.NaN)
        # Append this ndarray to our list
        columns_to_write.append(column_masked)

    # Combine each column's entries into an ndarray
    data_ndarray = np.vstack(columns_to_write)

    # We need the transpose so the individual data lists become transposed
    # to columns
    data_ndarray = data_ndarray.transpose()

    # We need in the form of nested sequences to satisfy csv.writer
    rows_to_write = data_ndarray.tolist()

    for row in rows_to_write:
        writer.writerow(list(row))

    csv_file.close()



def interpolate_with_threshold(array,
                               threshold=None,
                               make_copy=True,
                               extrapolate=False):
    """
    Linearly interpolate a numpy array along one dimension but only 
    for missing data n frames from a valid data point.  That is, 
    if there are too many contiguous missing data points, none of 
    those points get interpolated.


    Parameters
    ---------------------------------------
    array: 1-dimensional numpy array
      The array to be interpolated

    threshold: int
      The maximum size of a contiguous set of missing data points
      that gets interpolated.  Sets larger than this are left as NaNs.
      If threshold is set to NaN then all points are interpolated.

    make_copy: bool
      If True, do not modify the array parameter
      If False, interpolate the array parameter "in place"
      Either way, return a reference to the interpolated array

    extrapolate: bool
      If True, extrapolate linearly to the beginning and end of the array
      if there are NaNs on either end.

    Returns
    ---------------------------------------
    numpy array with the values interpolated


    Usage Example
    ---------------------------------------
    # example array  
    a = np.array([10, 12, 15, np.NaN, 17, \
                  np.NaN, np.NaN, np.NaN, -5], dtype='float')

    a2 = interpolate_with_threshold(a, 5)

    print(a)
    print(a2)


    Notes
    ---------------------------------------
    TODO: Extrapolation currently not implemented.  Perhaps try
    http://stackoverflow.com/questions/2745329/

    """

    """
  # (SKIP THIS, THIS IS FOR THE N-DIMENSIONAL CASE WHICH WE
  # HAVE NOT IMPLEMENTED YET)
    # Check that any frames with NaN in at least one dimension must
    # have it in all:
    frames_with_at_least_one_NaN = np.all(np.isnan(array), frame_dimension)
    frames_with_no_NaNs          = np.all(~np.isnan(array), frame_dimension)
    # check that each frame is either True for one of these arrays or 
    # the other but not both.
    assert(np.logical_xor(frames_with_at_least_one_NaN, frames_with_no_NaNs))
    frame_dropped = frames_with_at_least_one_NaN

  """

    assert(threshold == None or threshold >= 0)

    if(threshold == 0):  # everything gets left as NaN
        return array

    # Say array = [10, 12, 15, nan, 17, nan, nan, nan, -5]
    # Then np.isnan(array) =
    # [False, False, False, True, False True, True, True, False]
    # Let's obtain the "x-coordinates" of the NaN entries.
    # e.g. [3, 5, 6, 7]
    x = np.flatnonzero(np.isnan(array))

    # (If we weren't using a threshold and just interpolating all NaNs,
    # we could skip the next four lines.)
    if(threshold != None):
        # Group these together using a fancy trick from
        # http://stackoverflow.com/questions/2154249/, since
        # the lambda function x:x[0]-x[1] on an enumerated list will
        # group consecutive integers together
        # e.g. [[(0, 3)], [(1, 5), (2, 6), (3, 7)]]
        x_grouped = [list(group) for key, group in groupby(enumerate(x),
                                                           lambda i:i[0] - i[1])]

        # We want to know the first element from each "run", and the run's length
        # e.g. [(3, 1), (5, 3)]
        x_runs = [(i[0][1], len(i)) for i in x_grouped]

        # We need only interpolate on runs of length <= threshold
        # e.g. if threshold = 2, then we have only [(3, 1)]
        x_runs = [i for i in x_runs if i[1] <= threshold]

        # now expand the remaining runs
        # e.g. if threshold was 5, then x_runs would be [(3,1), (5,3)] so
        #      x would be [3, 5, 6, 7]
        # this give us the x-coordinates of the values to be interpolated:
        x = np.concatenate([(i[0] + list(range(i[1]))) for i in x_runs])

    # The x-coordinates of the data points, must be increasing.
    xp = np.flatnonzero(~np.isnan(array))
    # The y-coordinates of the data points, same length as xp
    yp = array[~np.isnan(array)]

    if make_copy:
        # Use a new array so we don't modify the original array passed to us
        new_array = np.copy(array)
    else:
        new_array = array

    if extrapolate:
        # TODO
        # :/  Might need to use scipy
        # also, careful of the "left" and "right" settings below
        pass

    # Place the interpolated values into the array
    # the "left" and "right" here mean that we want to leave NaNs in place
    # if the array begins and/or ends with a sequence of NaNs (i.e. don't
    # try to extrapolate)
    new_array[x] = np.interp(x, xp, yp, left=np.NaN, right=np.NaN)

    return new_array


def interpolate_with_threshold_2D(array, threshold=None, extrapolate=False):
    """
    Interpolate two-dimensional data along the second axis.  Each "row"
    is treated as a separate interpolation.  So if the first axis has 4 
    rows of n frames in the second axis, we are interpolating 4 times.

    Parameters
    ---------------------------------------    
    x_old       : [m x n_frames]
      The array to be interpolated along the second axis

    threshold: int (Optional)
      Number of contiguous frames above which no interpolation is done
      If none specified, all NaN frames are interpolated

    extrapolate: bool (Optional)
      If yes, values are extrapolated to the start and end, along the second
      axis (the axis being interpolated)

    Notes
    ---------------------------------------    
    This could be optimized with a specialized function for the case
    when all the NaN entries line up along the first dimension.  We'd 
    only need to calculate the mask once rather than m times.

    """
    new_array = array.copy()

    # NOTE: This version is a bit weird because the size of y is not 1d
    for i1 in range(np.shape(array)[0]):
        new_array[i1,:] = interpolate_with_threshold(array[i1,:],
                                                     threshold,
                                                     make_copy=True,
                                                     extrapolate=extrapolate)

    return new_array


def gausswin(L, alpha=2.5):
    """
    An N-point Gaussian window with alpha proportional to the 
    reciprocal of the standard deviation.  The width of the window
    is inversely related to the value of alpha.  A larger value of
    alpha produces a more narrow window.


    Parameters
    ----------------------------
    L : int
    alpha : float
      Defaults to 2.5

    Returns
    ----------------------------


    Notes
    ----------------------------
    TODO: I am ignoring some corner cases, for example:
      #L - negative, error  
      #L = 0
      #w => empty
      #L = 1
      #w = 1      

    Equivalent of Matlab's gausswin function.

    """

    N = L - 1
    n = np.arange(0, N + 1) - N / 2
    w = np.exp(-(1 / 2) * (alpha * n / (N / 2)) ** 2)

    return w




def _extract_time_from_disk(parent_ref, name, is_matrix = False):
    """
    This is for handling Matlab save vs Python save when we get to that point.
    """

    temp = parent_ref[name].value

    if is_matrix:
        wtf = temp     
    else:
        # Assuming vector, need to fix for eigenvectors
        if temp.shape[0] > temp.shape[1]:
            wtf = temp[:, 0]
        else:
            wtf = temp[0, :]

    return wtf
