# -*- coding: utf-8 -*-
"""
Posture features

"""


from __future__ import division

import scipy.ndimage.filters as filters
import numpy as np
import warnings
import time
import collections

# http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

from .. import utils
from .. import config

from . import events


class Bends(object):

    def __init__(self, nw):

        # TODO: I don't like this being in normalized worm
        p = nw.get_partition_subset('normal')

        for partition_key in p.keys():

            # retrieve the part of the worm we are currently looking at:
            bend_angles = nw.get_partition(partition_key, 'angles')

            # TODO: Should probably merge all three below ...

            # shape = (n):
            with warnings.catch_warnings(record=True):  # mean empty slice
                temp_mean = np.nanmean(a=bend_angles, axis=0)

            # degrees of freedom <= 0 for slice
            with warnings.catch_warnings(record=True):
                temp_std = np.nanstd(a=bend_angles, axis=0)

            # Sign the standard deviation (to provide the bend's dorsal/ventral orientation):
            #-------------------------------
            with warnings.catch_warnings(record=True):
                temp_std[temp_mean < 0] *= -1

            setattr(self, partition_key, BendSection(temp_mean, temp_std))

    def __repr__(self):
        return utils.print_object(self)

    @classmethod
    def from_disk(cls, saved_bend_data):

        self = cls.__new__(cls)

        for partition_key in saved_bend_data.keys():
            setattr(self, partition_key, BendSection.from_disk(
                saved_bend_data[partition_key]))

        return self


class BendSection(object):

    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    @classmethod
    def from_disk(cls, saved_bend_data):

        self = cls.__new__(cls)

        self.mean = saved_bend_data['mean'].value
        self.stdDev = saved_bend_data['stdDev'].value

        return self

    def __repr__(self):
        return utils.print_object(self)


def get_eccentricity_and_orientation(contour_x, contour_y):
    """
      get_eccentricity   

        [eccentricity, orientation] = seg_worm.utils.posture.getEccentricity(xOutline, yOutline, gridSize)

        Given x and y coordinates of the outline of a region of interest, fill
        the outline with a grid of evenly spaced points and use these points in
        a center of mass calculation to calculate the eccentricity and
        orientation of the equivalent ellipse.

        Placing points in the contour is a well known computer science problem
        known as the Point-in-Polygon problem.

        http://en.wikipedia.org/wiki/Point_in_polygon

        This function became a lot more complicated in an attempt to make it 
        go much faster. The complication comes from the simplication that can
        be made when the worm doesn't bend back on itself at all.


        OldName: getEccentricity.m


        Inputs:
        =======================================================================
        xOutline : [96 x num_frames] The x coordinates of the contour. In particular the contour
                    starts at the head and goes to the tail and then back to
                    the head (although no points are redundant)
        yOutline : [96 x num_frames]  The y coordinates of the contour "  "

        N_ECCENTRICITY (a constant from config.py):
                   (scalar) The # of points to place in the long dimension. More points
                   gives a more accurate estimate of the ellipse but increases
                   the calculation time.

        Outputs: a namedtuple containing:
        =======================================================================
        eccentricity - [1 x num_frames] The eccentricity of the equivalent ellipse
        orientation  - [1 x num_frames] The orientation angle of the equivalent ellipse

        Nature Methods Description
        =======================================================================
        Eccentricity. 
        ------------------
        The eccentricity of the worm’s posture is measured using
        the eccentricity of an equivalent ellipse to the worm’s filled contour.
        The orientation of the major axis for the equivalent ellipse is used in
        computing the amplitude, wavelength, and track length (described
        below).

        Status
        =======================================================================
        The code below is finished although I want to break it up into smaller
        functions. I also need to submit a bug report for the inpoly FEX code.

    Translation of: SegwormMatlabClasses / 
    +seg_worm / +utils / +posture / getEccentricity.m
    """

    #t_obj = time.time()

    N_GRID_POINTS = config.N_ECCENTRICITY  # TODO: move to options



    xo,yo,rot_angle = h__centerAndRotateOutlines(contour_x, contour_y)

    #In this function we detect "simple worms" and if they are detected
    #get interpolated y-contour values at each x grid location.
    y_interp_bottom,y_interp_top,x_interp,is_simple_worm = \
        h__getSimpleWormInfo(xo,yo,N_GRID_POINTS)
    
    grid_spacings = x_interp[1,:] - x_interp[0,:]

    y_bottom_bounds,y_top_bounds = \
        h__computeYBoundsOfSimpleWorms(y_interp_bottom,y_interp_top,grid_spacings)    

    n_frames = contour_x.shape[1]
    eccentricity = np.zeros(n_frames)
    eccentricity[:] = np.NaN
    orientation = np.zeros(n_frames)    
    orientation[:] = np.NaN
    
    eccentricity[is_simple_worm],orientation[is_simple_worm] = \
    h__computeOutputsFromSimpleWorms(x_interp,y_bottom_bounds,y_top_bounds,grid_spacings)    

    #Use slow grid method for all unfinished worms
    #--------------------------------------------------------------------------
    #
    #   This code is still a bit messy ...
    #
    
    x_range_all = np.ptp(contour_x, axis=0)
    y_range_all = np.ptp(contour_y, axis=0)
    
    grid_aspect_ratio = x_range_all / y_range_all

    run_mask = ~np.isnan(grid_aspect_ratio) & ~is_simple_worm

    eccentricity,orientation = h__getEccentricityAndOrientation(
        xo,yo,x_range_all,y_range_all,grid_aspect_ratio,N_GRID_POINTS,
        eccentricity,orientation,run_mask)

    #elapsed_time = time.time() - t_obj
    #print('Elapsed time in seconds for eccentricity: %d' % elapsed_time)
    

    #Fix the orientation - we undo the rotation that we originally did
    #--------------------------------------------------------------------------
    with np.errstate(invalid='ignore'):
        orientation_fixed = orientation + rot_angle*180/np.pi
        orientation_fixed[orientation_fixed > 90]  -= 180
        orientation_fixed[orientation_fixed < -90] += 180
    
    orientation = orientation_fixed;



  
    return (eccentricity, orientation)

def h__getEccentricityAndOrientation(x_mc,y_mc,xRange_all,yRange_all,gridAspectRatio_all,N_GRID_POINTS,eccentricity,orientation,run_mask):
    
    # h__getEccentricityAndOrientation
    for iFrame in np.nditer(utils.find(run_mask)):
        cur_aspect_ratio = gridAspectRatio_all[iFrame]
       # x_range = xRange_all[iFrame]
        #y_range = yRange_all[iFrame]


        #------------------------------------------------------



        cur_cx = x_mc[:, iFrame]
        cur_cy = y_mc[:, iFrame]
        poly = Polygon(zip(cur_cx, cur_cy))

        if cur_aspect_ratio > 1:
            # x size is larger so scale down the number of grid points in
            # the y direction
            n1 = N_GRID_POINTS
            n2 = np.round(N_GRID_POINTS / cur_aspect_ratio)
        else:
            # y size is larger so scale down the number of grid points in
            # the x direction
            n1 = np.round(N_GRID_POINTS * cur_aspect_ratio)
            n2 = N_GRID_POINTS

        wtf1 = np.linspace(np.min(x_mc[:, iFrame]), np.max(x_mc[:, iFrame]), num=n1)
        wtf2 = np.linspace(np.min(y_mc[:, iFrame]), np.max(y_mc[:, iFrame]), num=n2)

        m, n = np.meshgrid(wtf1, wtf2)

        n_points = m.size
        m_lin = m.reshape(n_points)
        n_lin = n.reshape(n_points)
        in_worm = np.zeros(n_points, dtype=np.bool)
        for i in range(n_points):
            p = Point(m_lin[i], n_lin[i])
#        try:
            in_worm[i] = poly.contains(p)
#        except ValueError:
#          import pdb
#          pdb.set_trace()

        x = m_lin[in_worm]
        y = n_lin[in_worm]

        eccentricity[iFrame],orientation[iFrame] = h__calculateSingleValues(x,y)

# First eccentricity value should be: 0.9743
        """
    TODO: Finish this
    plot(xOutline_mc(:,iFrame),yOutline_mc(:,iFrame),'g-o')
    hold on
    scatter(x,y,'r')
    hold off
    axis equal
    title(sprintf('%d',iFrame))
    pause
  """

    return (eccentricity,orientation)   

    


def h__calculateSingleValues(x,y):
    
    N = float(len(x))
    # Calculate normalized second central moments for the region.
    uxx = np.sum(x * x) / N
    uyy = np.sum(y * y) / N
    uxy = np.sum(x * y) / N
    
    # Calculate major axis length, minor axis length, and eccentricity.
    common = np.sqrt((uxx - uyy) ** 2 + 4 * (uxy ** 2))
    majorAxisLength = 2 * np.sqrt(2) * np.sqrt(uxx + uyy + common)
    minorAxisLength = 2 * np.sqrt(2) * np.sqrt(uxx + uyy - common)
    eccentricity_s = 2 * np.sqrt((majorAxisLength / 2) ** 2 - (minorAxisLength / 2) ** 2) / majorAxisLength
    
    # Calculate orientation.
    if (uyy > uxx):
        num = uyy - uxx + np.sqrt((uyy - uxx) ** 2 + 4 * uxy ** 2)
        den = 2 * uxy
    else:
        num = 2 * uxy
        den = uxx - uyy + np.sqrt((uxx - uyy) ** 2 + 4 * uxy ** 2)
    
    orientation_s = (180 / np.pi) * np.arctan(num / den) 
    
    return (eccentricity_s,orientation_s)
    
def h__computeOutputsFromSimpleWorms(x_interp,y_bottom_bounds,y_top_bounds,grid_spacings):

    """
    %
    %
    %
    %   Parameters:
    %   -------
    %   x_interp        : 
    %   y_bottom_bounds :
    %   y_top_bounds    :
    %
    %   Outputs:
    %   =======================================================================
    %   eccentricity : [1 x n_simple_worms]
    %   orientation  : [1 x n_simple_worms]
    %
    """

    n_simple_worms = x_interp.shape[1]
    n_grid_points  = x_interp.shape[0]

    #Initialize outputs of the loop
    #--------------------------------------------------------------------------
    
    #JAH: AT THIS POINT
    
    eccentricity = np.zeros(n_simple_worms)
    eccentricity[:] = np.NaN
    orientation = np.zeros(n_simple_worms)
    orientation[:] = np.NaN    


    #These are temporary arrays for holding the location of grid points that
    #fit inside the worm. They are a linerization of all points, so they don't
    #have a second dimension, we just pile new points from a worm frame onto
    #any old points from that frame.
    x_all = np.zeros(n_grid_points*n_grid_points)
    y_all = np.zeros(n_grid_points*n_grid_points)

    for iFrame in range(n_simple_worms):
        count = 0
        
        cur_d_unit = grid_spacings[iFrame]
        
        #For each x position, we increment from the minimum y value at that x location
        #to the maximum at that location, in the specified steps. We need
        #to hold onto the values for doing the eccentricity and orientation
        #calculations.
        #
        #NOTE: First and last grid points will not contain useful data
        #for iIndex = 2:(n_grid_points-1):
        for iIndex in range(1,n_grid_points-1):
            
            #Generate appropriate y-values on grid
            temp = utils.colon(
                y_bottom_bounds[iIndex,iFrame],
                cur_d_unit,
                y_top_bounds[iIndex,iFrame])
            
            #and store ...
            y_all[count:(count+len(temp))] = temp;
            #y_all[count:count+length(temp)] = temp;
            x_all[count:(count+len(temp))] = x_interp[iIndex,iFrame];
            count = count + len(temp)
    
        eccentricity[iFrame],orientation[iFrame] = h__calculateSingleValues(x_all[0:count],y_all[0:count]);    
    
    return (eccentricity,orientation)
    
def h__computeYBoundsOfSimpleWorms(y_interp_bottom,y_interp_top,grid_spacings):
    """
    %
    %
    %   Inputs
    %   =======================================================================
    %
    %   Outputs
    %   =======================================================================
    %   y_bottom_bounds : [n_grid_points x n_simple_worms]
    %   y_top_bounds    : [n_grid_points x n_simple_worms]
    
    %JAH: The bounds were being computed after aligning to the minimum (so that
    %the minimum was always a point), but:
    %- this seems biased
    %- removing this step speeds up the calculation
    %- "        " simplifies the code ...
    """

    #NOTE: The key point is that we round up on the lower value and down on the
    #
    
    y_bottom_bounds = np.ceil(y_interp_bottom/grid_spacings)*grid_spacings
    y_top_bounds    = np.floor(y_interp_top/grid_spacings)*grid_spacings
    
    #y_bottom_bounds = bsxfun(@times,ceil(bsxfun(@rdivide,y_interp_bottom,grid_spacings)),grid_spacings);
    #y_top_bounds    = bsxfun(@times,floor(bsxfun(@rdivide,y_interp_top,grid_spacings)),grid_spacings); 

    return (y_bottom_bounds,y_top_bounds)

def h__getSimpleWormInfo(xo,yo,n_grid_points):
    """
%
%
%   Inputs
%   =======================================================================
%   xo : x outline after being mean centered and rotated
%   yo : y "   "
%   n_grid_points : # of points to use for filling worm to determine
%   center of mass
%
%   Outputs
%   =======================================================================
%   y_interp_bottom : [n_grid_points x n_simple_worms], interpolated y
%                      values for the bottom contour of simple worms
%   y_interp_top    : [n_grid_points x n_simple_worms]
%   is_simple_worm  : [1 x n_frames]
%
%
%Determine 'simple' worms
%--------------------------------------------------------------------------
%
%   Simple worms have two sides in which x goes from - to + (or + to -), 
%   with no bending backwards. Importantly, this means that an x grid point
%   which is in the worm will only have two y-bounds. The values of the
%   y-bounds are the values of y on the two sides of the worm at that
%   x-location.
%
%
%   x => a [x,y] value from the outline
%
%               x
%             x  x 
%            x   x x
%             x       x
%               x x   x         This is not a simple worm.
%                 x   x
%                x    x         NOTE: This wouldn't happen in
%             x      x          this function because the worm isn't
%        x x x     x            rotated
%       x       x
%        x x x 
%
%
%
%                   x    x
%       x   x   x            x   x  x
%     x                                x    This is a simple worm!
%       x   x  x    x   x   x    x    x
%
%     |  |  |  |  |  |  |  |  |  |  |  |  <- grid locations where we 
%   will interpolate the outline.
%
%
%   For a simple worm this removes the need to sort the x
%   values in the grid with respect to the x-values of the contour. 
%
%   Normally a point-in-polygon algorithm would first have to find which
%   set of x side values the point is in between. A algorithm would also
%   not know that a set of x grid locations all have the same value (i.e.
%   there is repetition of the x-values for different y grid values), which
%   would save time as well.
%
%   Once we have the interpolated y-values, we round the y-values to grid
%   points to the nearest grid points. Doing this allows us to simply count
%   the points off that are between the two y-values.
%
%   This is illustrated below:
%
%           33 - y value of higher contour
%       
%           13 - y value of lower contour
%           11 - min
%
%
%   If 11 is min, and 5 is our spacing, then our grid points to test will
%   be 11,16,21,26,31,36,41, etc
%
%   If we round 13 up and 33 down to their appropriate locations (16 and
%   31), then we know that at this x value, the grid points 16:5:31 will
%   all be in the worm
    """ 
    
    n_contour_points = xo.shape[0]
    n_frames = xo.shape[1]
    
    y_interp_bottom = np.zeros((n_grid_points,n_frames))
    y_interp_bottom[:] = np.NaN
    y_interp_top = np.zeros((n_grid_points,n_frames))
    y_interp_top[:] = np.NaN
    x_interp = np.zeros((n_grid_points,n_frames))
    x_interp[:] = np.NaN
    
    mx_x_I = np.argmax(xo,axis=0)
    mn_x_I = np.argmin(xo,axis=0)
    
    """
    %We don't know if the worm will be going from left to right or right to
%left, we need slightly different code later on depending on which
%
%NOTE: we are testing the indices, not the values

%JAH: The following code could be simplified a bit to remove the two loops
%...
    """

    """
    %--------------------------------------------------------------------------
    %NOTE: The basic difference between these loops is in how x1 and x2 are
    %defined. For interpolation we must always go from a lower value to a
    %higher value (to use the quick method of interpolation in Matlab). Note, 
    %the sign on the comparison is also different.
    """   
   
    min_first_mask = mn_x_I < mx_x_I
    min_last_mask  = mx_x_I < mn_x_I

    d = np.vstack((np.diff(xo,axis=0),xo[0,:]-xo[-1,:]))

    is_simple_worm = np.zeros(n_frames,dtype=bool)    
    
    for iFrame in np.nditer(utils.find(min_first_mask)):

        x1 = utils.colon(mn_x_I[iFrame],1,mx_x_I[iFrame])
        x1 = x1.astype(np.int32,copy=False)
        
        if np.all(d[x1[:-1],iFrame] > 0):
            
            x2 = np.hstack((utils.colon(mx_x_I[iFrame],1,n_contour_points-1),
                            utils.colon(0,1,mn_x_I[iFrame])))
            x2 = x2.astype(np.int32,copy=False)          
                            
            if np.all(d[x2[:-1],iFrame] < 0):
                
                is_simple_worm[iFrame] = True
                    
                #import pdb
                #pdb.set_trace()
                y_interp_bottom[:,iFrame],y_interp_top[:,iFrame],x_interp[:,iFrame] = \
                    h__getInterpValuesForSimpleWorm(x1,x2,xo,yo,n_grid_points,iFrame,mn_x_I,mx_x_I)    

    for iFrame in np.nditer(utils.find(min_last_mask)):

        x2 = utils.colon(mx_x_I[iFrame],1,mn_x_I[iFrame])
        x2 = x2.astype(np.int32,copy=False)
        
        if np.all(d[x2[:-1],iFrame] < 0):
            
            x1 = np.hstack((utils.colon(mn_x_I[iFrame],1,n_contour_points-1),
                            utils.colon(0,1,mx_x_I[iFrame])))
            x1 = x1.astype(np.int32,copy=False)          
                            
            if np.all(d[x1[:-1],iFrame] > 0):
                
                is_simple_worm[iFrame] = True
                    
                y_interp_bottom[:,iFrame],y_interp_top[:,iFrame],x_interp[:,iFrame] = \
                    h__getInterpValuesForSimpleWorm(x1,x2,xo,yo,n_grid_points,iFrame,mn_x_I,mx_x_I)  

    y_interp_bottom = y_interp_bottom[:,is_simple_worm]
    y_interp_top    = y_interp_top[:,is_simple_worm]
    x_interp        = x_interp[:,is_simple_worm]

    """
    %Ensure that y1 < y2 for all frames (DONE), if not then flip (NYI)
    %--------------------------------------------------------------------------
    %NOTE: we skip the first and last point because they should be the same, although after rounding (floor, ceil)
    %they actually will tend to be apart by the amount 'dy', with the opposite relationship that the rest of the data has
    %I also filter this out in the loop below by skipping the
    %first and last grid point
    %
    %We need to ensure y1 is less than y2, because below we create vectors
    %going from low to high, and if these are reversed, then the vector will
    %be empty and the result empty
    %
    %NOTE: We skip the first and last value as they should essentially be the
    %same for the top and bottom contours
    
    
    %NOTE: This can be fixed easily. Any violations just need to be swapped ...
    """
    is_bottom_correct = np.all(y_interp_bottom[1:-1,:] < y_interp_top[1:-1,:],axis=0)
    
    if not np.all(is_bottom_correct):
        raise Exception('Assumption violated, need to fix code')


    """
    
    %This code needs to be tested ...
    
    temp_top = y_interp_bottom(:,~is_bottom_correct);
    
    y_interp_bottom(:,~is_bottom_correct) = y_interp_top(:,~is_bottom_correct);
    y_interp_top(:,~is_bottom_correct)    = temp_top;
    
    """
    
    return (y_interp_bottom,y_interp_top,x_interp,is_simple_worm)



def h__getInterpValuesForSimpleWorm(x1,x2,xo,yo,gridSize,iFrame,mn_x_I,mx_x_I):
    """
    %
    %
    %   [y_interp_1,y_interp_2,x_out_all] = h__getInterpValues(x1,x2,xOutline_mc,yOutline_mc,gridSize,iFrame,mn_x_I,mx_x_I)
    %
    %
    %   This function computes the interpolated y-values on the contour
    %   for the x grid locations that we will be testing at. It also computes
    %   the x grid (TODO: Might move this ...)
    %   
    %   Inputs
    %   =======================================================================
    %   x1          : [1 x m] contour indices with values going from low to high
    %   x2          : [1 x n] contour indices with values going from high to low
    %
    %                   NOTE: x1 and x2 do not need to have the same length
    %                   although m and n are usually around 48 - 50
    %       
    %   xOutline_mc : [96 x n_frames] x values of the contour
    %   yOutline_mc : [96 x n_frames] y values of the contour
    %   gridSize    : (scalar, normally 50) # of points to use between the minimum and maximum value
    %   iFrame      : (scalar) current frame index
    %   mn_x_I      : [1 x n_frames] array of minimum x values for all frames
    %   mx_x_I      : [1 x n_frames] "       " maximum "           "
    """
    
    X_in_1 = xo[x1,iFrame];
    X_in_2 = xo[x2,iFrame];
    Y_in_1 = yo[x1,iFrame];
    Y_in_2 = yo[x2,iFrame];
    
    X_out  = np.linspace(
        xo[mn_x_I[iFrame],iFrame],
        xo[mx_x_I[iFrame],iFrame],
        num=gridSize)
    

    y_interp_bottom = np.interp(X_out,X_in_1,Y_in_1)
    y_interp_top = np.interp(X_out,X_in_2[::-1],Y_in_2[::-1])    
 
        
    return (y_interp_bottom,y_interp_top,X_out)

def h__centerAndRotateOutlines(x_outline, y_outline):
    """
    #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getEccentricity.m#L391
    """
    
    x_mc = x_outline - np.mean(x_outline, axis=0)  # mc - mean centered
    y_mc = y_outline - np.mean(y_outline, axis=0)    
    
    """
%Rough rotation of the worm
%--------------------------------------------------------------------------
%
%   We rotate the worm by the vector formed from going from the first point
%   to the last point. This accomplishes two things.
%
%   1) For the brute force approach, this makes it so the worm is
%   encompassed better by a rectangle instead of a square, which means that
%   there are fewer grid points to test that are not in the worm (as
%   opposed to a worm that is on a 45 degree angle that would have many
%   points on the grid outside of the worm). In other words, the bounding
%   box is a better approximation of the worm when the worm is rotated then
%   when it is not. In the brute force case we place points in the bounding
%   box, so the smaller the bounding box, the faster the code will run.
%
%   2) This allows us to hardcode only looking for "simple" worms 
%   (see description below) that vary in the x-direction. Otherwise we
%   might need to also look for simple worms in the y direction. Along
%   similar lines this makes it more likely that we will get simple worms.

%NOTE: Here we are assuming that the head or tail is located at this middle
%index    
    """
    
    n_outline_points = x_outline.shape[0]

    """
%        6  5   <= 
%      1      4
%  =>    2  3
%
%   NOTE: we want indices 1 and 4 in this example, 4 is half of 6, + 1
%    
   """
   
    head_or_tail_index = round(n_outline_points/2)   
   

    y = y_mc[head_or_tail_index,:] - y_mc[0,:] #_mc - mean centered
    x = x_mc[head_or_tail_index,:] - x_mc[0,:]

    rot_angle = np.arctan2(y,x);   

    """
    %I expanded the rotation matrix to allow processing all frames at once
    %
    %   i.e. rather than R*X (Matrix multiplication)
    %
    %   I have r(1)*X(1) + r(2)*X(2) etc, but where X(1), X(2), etc is really
    %   a vector of values, not just a singular value like you would need for
    %   R*X
    """
    xo = x_mc*np.cos(rot_angle)  + y_mc*np.sin(rot_angle)
    yo = x_mc*-np.sin(rot_angle) + y_mc*np.cos(rot_angle)   
   
    return (xo,yo,rot_angle)


def get_amplitude_and_wavelength(theta_d, sx, sy, worm_lengths):

    # https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getAmplitudeAndWavelength.m
    N_POINTS_FFT = 512
    HALF_N_FFT = N_POINTS_FFT / 2
    MIN_DIST_PEAKS = 5
    WAVELENGTH_PCT_MAX_CUTOFF = 0.5  # TODO: Describe
    WAVELENGTH_PCT_CUTOFF = 2

    # TODO: Write in Python
    # assert(size(sx,1) <= N_POINTS_FFT,'# of points used in the FFT must be
    # more than the # of points in the skeleton')

    theta_r = theta_d * (np.pi / 180)

    # Unrotate worm
    #------------------------------

    wwx = sx * np.cos(theta_r) + sy * np.sin(theta_r)
    wwy = sx * -np.sin(theta_r) + sy * np.cos(theta_r)

    # Subtract mean
    #-----------------------------------------------------------------
    wwx = wwx - np.mean(wwx, axis=0)
    wwy = wwy - np.mean(wwy, axis=0)

    # Calculate track amplitude
    #--------------------------------------------------------------------------
    amp1 = np.amax(wwy, axis=0)
    amp2 = np.amin(wwy, axis=0)
    amplitude_max = amp1 - amp2
    amp2 = np.abs(amp2)
    with np.errstate(invalid='ignore'):
        amplitude_ratio = np.divide(np.minimum(amp1, amp2), np.maximum(amp1, amp2))

    # Calculate track length
    #--------------------------------------------------------------------------
    # NOTE: This is the x distance after rotation, and is different from the worm
    # length which follows the skeleton. This will always be smaller than the
    # worm length.
    track_length = np.amax(wwx, axis=0) - np.amin(wwx, axis=0)

    # Wavelength calculation
    #--------------------------------------------------------------------------
    dwwx = np.diff(wwx, 1, axis=0)

    # Does the sign change? This is a check to make sure that the change in x is
    # always going one way or the other. Is sign of all differences the same as
    # the sign of the first, or rather, are any of the signs not the same as the
    # first sign, indicating a "bad worm orientation".
    #
    # NOT: This means that within a frame, if the worm x direction changes, then
    # it is considered a bad worm and is not evaluated for wavelength
    #

    with np.errstate(invalid='ignore'):
        bad_worm_orientation = np.any(
            np.not_equal(np.sign(dwwx), np.sign(dwwx[0, :])), axis=0)

    n_frames = bad_worm_orientation.size

    primary_wavelength = np.zeros(n_frames)
    primary_wavelength[:] = np.NaN
    secondary_wavelength = np.zeros(n_frames)
    secondary_wavelength[:] = np.NaN

    # NOTE: Right now this varies from worm to worm which means the spectral
    # resolution varies as well from worm to worm
    spatial_sampling_frequency = (wwx.shape[0] - 1) / track_length

    ds = 1 / spatial_sampling_frequency

    frames_to_calculate = (np.logical_not(bad_worm_orientation)).nonzero()[0]

    for cur_frame in frames_to_calculate:

        # Create an evenly sampled x-axis, note that ds varies
        x1 = wwx[0, cur_frame]
        x2 = wwx[-1, cur_frame]
        if x1 > x2:
            iwwx = utils.colon(x1, -ds[cur_frame], x2)
            iwwy = np.interp(iwwx, wwx[::-1, cur_frame], wwy[::-1, cur_frame])
            iwwy = iwwy[::-1]
        else:
            iwwx = utils.colon(x1, ds[cur_frame], x2)
            iwwy = np.interp(iwwx, wwx[:, cur_frame], wwy[:, cur_frame])
            iwwy = iwwy[::-1]

        temp = np.fft.fft(iwwy, N_POINTS_FFT)

        if config.MIMIC_OLD_BEHAVIOUR:
            iY = temp[0:HALF_N_FFT]
            iY = iY * np.conjugate(iY) / N_POINTS_FFT
        else:
            iY = np.abs(temp[0:HALF_N_FFT])

        # Find peaks that are greater than the cutoff
        peaks, indx = utils.separated_peaks(
            iY, MIN_DIST_PEAKS, True, WAVELENGTH_PCT_MAX_CUTOFF * np.amax(iY))

        # This is what the supplemental says, not what was done in the previous
        # code. I'm not sure what was done for the actual paper, but I would
        # guess they used power.
        #
        # This gets used when determining the secondary wavelength, as it must
        # be greater than half the maximum to be considered a secondary
        # wavelength.

        # NOTE: True Amplitude = 2*abs(fft)/(length_real_data i.e. 48 or 49, not 512)
        #
        # i.e. for a sinusoid of a given amplitude, the above formula would give
        # you the amplitude of the sinusoid

        # We sort the peaks so that the largest is at the first index and will
        # be primary, this was not done in the previous version of the code
        I = np.argsort(-1 * peaks)
        indx = indx[I]

        frequency_values = (indx - 1) / N_POINTS_FFT * \
            spatial_sampling_frequency[cur_frame]

        all_wavelengths = 1 / frequency_values

        p_temp = all_wavelengths[0]

        if indx.size > 1:
            s_temp = all_wavelengths[1]
        else:
            s_temp = np.NaN

        worm_wavelength_max = WAVELENGTH_PCT_CUTOFF * worm_lengths[cur_frame]

        # Cap wavelengths ...
        if p_temp > worm_wavelength_max:
            p_temp = worm_wavelength_max

        # ??? Do we really want to keep this as well if p_temp == worm_2x?
        # i.e., should the secondary wavelength be valid if the primary is also
        # limited in this way ?????
        if s_temp > worm_wavelength_max:
            s_temp = worm_wavelength_max

        primary_wavelength[cur_frame] = p_temp
        secondary_wavelength[cur_frame] = s_temp

    if config.MIMIC_OLD_BEHAVIOUR:
        # Suppress warnings so we can compare a numpy array that may contain NaNs
        # without triggering a Runtime Warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mask = secondary_wavelength > primary_wavelength

        temp = secondary_wavelength[mask]
        secondary_wavelength[mask] = primary_wavelength[mask]
        primary_wavelength[mask] = temp

    amp_wave_track = \
        collections.namedtuple('amp_wave_track',
                               ['amplitude_max', 'amplitude_ratio', 'primary_wavelength',
                                'secondary_wavelength', 'track_length'])

    amp_wave_track.amplitude_max = amplitude_max
    amp_wave_track.amplitude_ratio = amplitude_ratio
    amp_wave_track.primary_wavelength = primary_wavelength
    amp_wave_track.secondary_wavelength = secondary_wavelength
    amp_wave_track.track_length = track_length

    return amp_wave_track

"""

Old Vs New Code:
  - power instead of magnitude is used for comparison
  - primary and secondary wavelength may be switched ...
  - error in maxPeaksDist for distance threshold, not sure where in code
        - see frame 880 for example
        - minus 1 just gives new problem - see 1794

"""


def get_worm_kinks(bend_angles):
    # https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getWormKinks.m

    # Determine the bend segment length threshold.
    n_angles = bend_angles.shape[0]
    length_threshold = np.round(n_angles * config.KINK_LENGTH_THRESHOLD_PCT)

    # Compute a gaussian filter for the angles.
    #--------------------------------------------------------------------------
    # JAH NOTE: This is a nice way of getting the appropriate odd value
    # unlike the other code with so many if statements ...
    #- see window code which tries to get an odd value ...
    #- I'd like to go back and fix that code ...
    half_length_thr = np.round(length_threshold / 2)
    gauss_filter    = utils.gausswin(half_length_thr * 2 + 1) \
        / half_length_thr

    # Compute the kinks for the worms.
    n_frames = bend_angles.shape[1]
    n_kinks_all = np.zeros(n_frames, dtype=float)
    n_kinks_all[:] = np.NaN

    #(np.any(np.logical_or(mask_pos,mask_neg),axis=0)).nonzero()[0]

    nan_mask = np.isnan(bend_angles)

    for iFrame in (~np.all(nan_mask, axis=0)).nonzero()[0]:
        smoothed_bend_angles = filters.convolve1d(
            bend_angles[:, iFrame], gauss_filter, cval=0, mode='constant')

        # This code is nearly identical in getForaging
        #-------------------------------------------------------
        n_frames = smoothed_bend_angles.shape[0]

        with np.errstate(invalid='ignore'):
            dataSign = np.sign(smoothed_bend_angles)

        if np.any(np.equal(dataSign, 0)):
            # I don't expect that we'll ever actually reach 0
            # The code for zero was a bit weird, it keeps counting if no sign
            # change i.e. + + + 0 + + + => all +
            #
            # but if counts for both if sign change
            # + + 0 - - - => 3 +s and 4 -s
            raise Exception("Unhandled code case")

        sign_change_I = (
            np.not_equal(dataSign[1:], dataSign[0:-1])).nonzero()[0]

        end_I = np.concatenate(
            (sign_change_I, n_frames * np.ones(1, dtype=np.result_type(sign_change_I))))

        wtf1 = np.zeros(1, dtype=np.result_type(sign_change_I))
        wtf2 = sign_change_I + 1
        start_I = np.concatenate((wtf1, wtf2))  # +2? due to inclusion rules???

        # All NaN values are considered sign changes, remove these ...
        keep_mask = np.logical_not(np.isnan(smoothed_bend_angles[start_I]))

        start_I = start_I[keep_mask]
        end_I = end_I[keep_mask]

        # The old code had a provision for having NaN values in the middle
        # of the worm. I have not translated that feature to the newer code. I
        # don't think it will ever happen though for a valid frame, only on the
        # edges should you have NaN values.
        if start_I.size != 0 and np.any(np.isnan(smoothed_bend_angles[start_I[0]:end_I[-1]])):
            raise Exception("Unhandled code case")

        #-------------------------------------------------------
        # End of identical code ...

        lengths = end_I - start_I + 1

        # Adjust lengths for first and last:
        # Basically we allow NaN values to count towards the length for the
        # first and last stretches
        if lengths.size != 0:
            if start_I[0] != 0:  # Due to leading NaNs
                lengths[0] = end_I[0] + 1
            if end_I[-1] != n_frames:  # Due to trailing NaNs
                lengths[-1] = n_frames - start_I[-1]

        n_kinks_all[iFrame] = np.sum(lengths >= length_threshold)

    return n_kinks_all


def get_worm_coils(frame_codes, midbody_distance):

    # This function is very reliant on the MRC processor

    # https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bfeatures/%40posture/getCoils.m

    COIL_FRAME_THRESHOLD = np.round(1 / 5 * config.FPS)
    COIL_START_CODES = [105, 106]
    FRAME_SEGMENTED = 1  # Go back 1 frame, this is the end of the coil ...

    # Algorithm: Whenever a new start is found, find the first segmented frame,
    # that's the end.

    # Add on a frame to allow closing a coil at the end ...
    coil_start_mask = (frame_codes == COIL_START_CODES[0]) | (
        frame_codes == COIL_START_CODES[1])
    np_false = np.zeros((1,), dtype=bool)
    coil_start_mask = np.concatenate((coil_start_mask, np_false))

    # NOTE: These are not guaranteed ends, just possible ends ...
    end_coil_mask = frame_codes == FRAME_SEGMENTED
    np_true = ~np_false
    end_coil_mask = np.concatenate((end_coil_mask, np_true))

    in_coil = False
    coil_frame_start = -1
    n_coils = 0
    n_frames_plus1 = len(frame_codes) + 1

    starts = []
    ends = []

    for iFrame in range(n_frames_plus1):
        if in_coil:
            if end_coil_mask[iFrame]:
                n_coil_frames = iFrame - coil_frame_start
                if n_coil_frames >= COIL_FRAME_THRESHOLD:
                    n_coils += 1

                    starts.append(coil_frame_start)
                    ends.append(iFrame - 1)

                in_coil = False
        elif coil_start_mask[iFrame]:
            in_coil = True
            coil_frame_start = iFrame

    if config.MIMIC_OLD_BEHAVIOUR:
        if (len(starts) > 0) & (ends[-1] == len(frame_codes) - 1):
            ends[-1] += -1
            starts[-1] += -1

    temp = events.EventList(np.transpose(np.vstack((starts, ends))))

    return events.EventListWithFeatures(temp, midbody_distance)

    """
  coiled_frames = h__getWormTouchFrames(frame_codes, config.FPS);

  COIL_FRAME_THRESHOLD = np.round(1/5 * config.FPS);
  """

    # Algorithm: Whenever a new start is found, find the first segmented frame,
    # that's the end.

    # Add on a frame to allow closing a coil at the end ...

# pdb.set_trace()    # DEBUG

    """
  coil_start_mask = [frameCodes == COIL_START_CODES(1) | frameCodes == COIL_START_CODES(2) false];
  
  #NOTE: These are not guaranteed ends, just possible ends ...
  end_coil_mask   = [frameCodes == FRAME_SEGMENTED true];
  
  in_coil = false;
  coil_frame_start = 0;
  
  n_coils = 0;
  
  n_frames_p1 = length(frameCodes) + 1;
  
  for iFrame = 1:n_frames_p1
      if in_coil
          if end_coil_mask(iFrame)
              
              n_coil_frames = iFrame - coil_frame_start;
              if n_coil_frames >= COIL_FRAME_THRESHOLD
                  n_coils = n_coils + 1;
                  
                  touchFrames(n_coils).start = coil_frame_start; 
                  touchFrames(n_coils).end   = iFrame - 1;
              end
              in_coil = false;
          end
      elseif coil_start_mask(iFrame)
          in_coil = true;
          coil_frame_start = iFrame;
      end
  end
  
  
  
  
  
  if d_opts.mimic_old_behavior
      if ~isempty(coiled_frames) && coiled_frames(end).end == length(frame_codes)
         coiled_frames(end).end   = coiled_frames(end).end - 1;
         coiled_frames(end).start = coiled_frames(end).start - 1;
      end
  end
  
  coiled_events = seg_worm.feature.event(coiled_frames,FPS,midbody_distance,DATA_NAME,INTER_DATA_NAME);
  
  return coiled_events.getFeatureStruct;
   
  """

    return None


class Directions(object):

    """

    tail2head
    head
    tail

    """

    def __init__(self, sx, sy, wp):
        """

        wp : (worm paritions) from normalized worm, 

        """

        # These are the names of the final fields
        NAMES = ['tail2head', 'head', 'tail']

        # For each set of indices, compute the centroids of the tip and tail then
        # compute a direction vector between them (tip - tail)

        # I - "indices" - really a tuple of start,stop
        TIP_I = [wp['head'], wp['head_tip'], wp['tail_tip']]
        TAIL_I = [wp['tail'], wp['head_base'], wp['tail_base']]

        TIP_S = [slice(*x) for x in TIP_I]  # S - slice
        TAIL_S = [slice(*x) for x in TAIL_I]

        for iVector in range(3):
            tip_x = np.mean(sx[TIP_S[iVector], :], axis=0)
            tip_y = np.mean(sy[TIP_S[iVector], :], axis=0)
            tail_x = np.mean(sx[TAIL_S[iVector], :], axis=0)
            tail_y = np.mean(sy[TAIL_S[iVector], :], axis=0)

            dir_value = 180 / np.pi * \
                np.arctan2(tip_y - tail_y, tip_x - tail_x)
            setattr(self, NAMES[iVector], dir_value)

    @classmethod
    def from_disk(cls, data):

        self = cls.__new__(cls)

        for key in data:
            setattr(self, key, data[key])

        return self

    def __repr__(self):
        return utils.print_object(self)


def get_eigenworms(sx, sy, eigen_worms, N_EIGENWORMS_USE):
    """

    Parameters:
    -----------
    eigen_worms: [7,48]  

    """

    angles = np.arctan2(np.diff(sy, n=1, axis=0), np.diff(sx, n=1, axis=0))

    n_frames = sx.shape[1]

    # need to deal with cases where angle changes discontinuously from -pi
    # to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
    # respectively to all remaining points.  This effectively extends the
    # range outside the -pi to pi range.  Everything is re-centred later
    # when we subtract off the mean.
    false_row = np.zeros((1, n_frames), dtype=bool)

    # NOTE: By adding the row of falses, we shift the trues
    # to the next value, which allows indices to match. Otherwise after every
    # find statement we would need to add 1, I think this is a bit faster ...

    with np.errstate(invalid='ignore'):
        mask_pos = np.concatenate(
            (false_row, np.diff(angles, n=1, axis=0) > np.pi), axis=0)
        mask_neg = np.concatenate(
            (false_row, np.diff(angles, n=1, axis=0) < -np.pi), axis=0)

    # Only fix the frames we need to, in which there is a jump in going from one
    # segment to the next ...
    fix_frames_I = (
        np.any(np.logical_or(mask_pos, mask_neg), axis=0)).nonzero()[0]

    for cur_frame in fix_frames_I:

        positive_jump_I = (mask_pos[:, cur_frame]).nonzero()[0]
        negative_jump_I = (mask_neg[:, cur_frame]).nonzero()[0]

        # subtract 2pi from remainging data after positive jumps
        # Note that the jumps impact all subsequent frames
        for cur_pos_jump in positive_jump_I:
            angles[cur_pos_jump:, cur_frame] -= 2 * np.pi

        # add 2pi to remaining data after negative jumps
        for cur_neg_jump in negative_jump_I:
            angles[cur_neg_jump:, cur_frame] += 2 * np.pi

    angles = angles - np.mean(angles, axis=0)

    # DEBUG: hiding this error for now - @MichaelCurrie
    return None  # np.dot(eigen_worms[0:N_EIGENWORMS_USE,:],angles)