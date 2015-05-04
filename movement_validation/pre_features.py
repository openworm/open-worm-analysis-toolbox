# -*- coding: utf-8 -*-
"""
Pre-features calculation

Written by @JimHokanson

"""
import numpy as np

from . import config

from scipy.signal import savgol_filter as sgolay
#Why didn't this work?
#import scipy.signal.savgol_filter as sgolay
#http://stackoverflow.com/questions/29324814/what-are-the-rules-for-importing-with-as-in-python-without-using-from

from . import utils



class WormParsing(object):

    """
    This might eventually move somewhere else, but at least it is contained within
    the class. It was originally in the Normalized Worm code which was making things
    a bit overwhelming.
    
    TODO: Self does not refer to WormParsing ...
    
    """

    @staticmethod
    def h__computeNormalVectors(data):
        dx = np.gradient(data[0,:])
        dy = np.gradient(data[1,:])
        
        #This approach gives us -1 for the projection
        #We could also use:
        #dx_norm = -dy;
        #dy_norm = dx;
        #
        #and we would get 1 for the projection
        dx_norm = dy;
        dy_norm = -dx;
        
        vc_d_magnitude = np.sqrt(dx_norm**2 + dy_norm**2);
        
        norm_x = dx_norm/vc_d_magnitude;
        norm_y = dy_norm/vc_d_magnitude;
        
        return norm_x,norm_y

    @staticmethod
    def h__roundToOdd(value):
        value = np.floor(value)
        if value % 2 == 0:
            value = value + 1
            
        return value

    @staticmethod
    def h__getBounds(n1,n2,p_left,p_right):
        """
        
        Returns slice starts and stops
        #TODO: Rename everything to start and stop
        """
        pct = np.linspace(0,1,n1)
        left_pct = pct - p_left
        right_pct = pct + p_right

        left_I = np.floor(left_pct*n2)
        right_I = np.ceil(right_pct*n2)
        left_I[left_I < 0] = 0;
        right_I[right_I >= n2] = n2-1
        return left_I,right_I
    
    @staticmethod
    def h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across,d_across,left_I,right_I):
        
        n_s1 = s1.shape[1]
        match_I = np.zeros(n_s1,dtype=np.int)
        match_I[0] = 0
        match_I[-1] = s2.shape[1]
        
        dp_values = np.zeros(n_s1)
        all_signs_used = np.zeros(n_s1)

        #There is no need to do the first and last point
        for I,(lb,rb) in enumerate(zip(left_I[1:-1],right_I[1:-1])):

            I = I + 1
            [abs_dp_value,dp_I,sign_used] = WormParsing.h__getProjectionIndex(norm_x[I],norm_y[I],dx_across[I,lb:rb],dy_across[I,lb:rb],lb,d_across[I,lb:rb],0)
            all_signs_used[I] = sign_used
            dp_values[I] = abs_dp_value
            match_I[I] = dp_I
            
        if not np.all(all_signs_used[1:-1] == all_signs_used[1]):
            if np.sum(all_signs_used) > 0:
                sign_use = 1
                I_bad = utils.find(all_signs_used[1:-1] != 1) + 1
            else:
                I_bad = utils.find(all_signs_used[1:-1] != -1) + 1
                sign_use = -1
                
            for I in I_bad:    
                lb = left_I[I]
                rb = right_I[I]
                [abs_dp_value,dp_I,sign_used] = WormParsing.h__getProjectionIndex(norm_x[I],norm_y[I],dx_across[I,lb:rb],dy_across[I,lb:rb],lb,d_across[I,lb:rb],sign_use)
                all_signs_used[I] = sign_used
                dp_values[I] = abs_dp_value
                match_I[I] = dp_I    


        return (dp_values,match_I)
    
    @staticmethod
    def h__getProjectionIndex(vc_dx_ortho,vc_dy_ortho,dx_across_worm,dy_across_worm,left_I,d_across,sign_use):
        
        
        #% nvc_local = nvc(nvc_indices_use,:);
        #%
        #% dx_across_worm = cur_point(1) - nvc_local(:,1);
        #% dy_across_worm = cur_point(2) - nvc_local(:,2);
        #%
        #% d_magnitude = sqrt(dx_across_worm.^2+dy_across_worm.^2);
        #%
        #% dx_across_worm = dx_across_worm./d_magnitude;
        #% dy_across_worm = dy_across_worm./d_magnitude;

                
        #SPEED: Compute normalized distances for all pairs ...
        #Might need to downsample
        
        dp = dx_across_worm*vc_dx_ortho + dy_across_worm*vc_dy_ortho;
        
        #I'd like to not have to do this step, it has to do with the relationship
        #between the vulva and non-vulva side. This should be consistent across the
        #entire animal and could be passed in, unless the worm rolls.
        sign_used = -1;
        if sign_use == 0 and np.sum(dp) > 0:
            #Instead of multiplying by -1 we could hardcode the flip of the logic
            #below (e.g. max instead of min, > vs <)
            dp = -1*dp
            sign_used = 1
        elif sign_use == 1:
            dp = -1*dp
            sign_used = 1
        
        #This is slow, presumably due to the memory allocation ...
        #               < right                         < left
        #possible = [dp(1:end-1) < dp(2:end) false] & [false dp(2:end) < dp(1:end-1)];
        
        #In Matlab
        #possible = (dp(2:end-1) < dp(3:end)) & (dp(2:end-1) < dp(1:end-2));
        possible = (dp[1:-2] < dp[2:-1]) & (dp[1:-2] < dp[0:-3])
        
        Ip = utils.find(possible)
        if len(Ip) == 1:
            dp_I = Ip+1
            dp_value = dp[dp_I]
        elif len(Ip) > 1:
            temp_I = np.argmin(d_across[Ip])
            dp_I = Ip[temp_I]+1
            dp_value = dp[dp_I]
        else:
            dp_I = np.argmin(dp)
            dp_value = dp[dp_I]
        
        I = left_I + dp_I
    
        return (dp_value,I,sign_used)
    
    @staticmethod
    def h__updateEndsByWalking(d_across,match_I1,s1,s2,END_S1_WALK_PCT):
        
        """
        
        Parameters
        ----------
        d_across
        match_I1
        s1
        s2
        END_S1_WALK_PCT :
        
        Returns
        -------
        (I_1,I_2)
        
        """
        n_s1 = s1.shape[1]
        n_s2 = s2.shape[1]       
        
        
        end_s1_walk_I = np.ceil(n_s1*END_S1_WALK_PCT)
        end_s2_walk_I = 2*end_s1_walk_I
        p1_I,p2_I = WormParsing.h__getPartnersViaWalk(0,end_s1_walk_I,0,end_s2_walk_I,d_across,s1,s2)
        
        match_I1[p1_I] = p2_I
        
        keep_mask = np.zeros(len(match_I1),dtype=np.bool)
        keep_mask[p1_I] = True
        
        #Add 
        end_s1_walk_backwards = n_s1 - end_s1_walk_I + 1
        end_s2_walk_backwards = n_s2 - end_s2_walk_I + 1
        
        
        p1_I,p2_I = WormParsing.h__getPartnersViaWalk(n_s1-1,end_s1_walk_backwards,n_s2-1,end_s2_walk_backwards,d_across,s1,s2)

        match_I1[p1_I] = p2_I
        keep_mask[p1_I] = True
        
        #anything in between we'll use the projection appproach
        keep_mask[end_s1_walk_I+1:end_s1_walk_backwards] = True
        
        #Always keep ends
        keep_mask[0]   = True
        keep_mask[-1] = True
    
        match_I1[0] = 0
        match_I1[-1] = n_s2-1
        
    
        #This isn't perfect but it removes some back and forth behavior
        #of the matching. We'd rather drop points and smooth
        I_1 = utils.find(keep_mask)
        I_2 = match_I1[keep_mask]

        return (I_1,I_2)
        
    @staticmethod
    def h__getPartnersViaWalk(s1,e1,s2,e2,d,xy1,xy2):
    #%
    #%   s1: start index for side 1
    #%   e1: end index for side 1
    #%
    #%   d :
    #%       distance from I1 to I2 is d(I1,I2)
    #%
    #%   d1 : [n x 2]
    #%       x,y pairs for side 1
    #%
    #%
    #%


        #TODO: remove hardcode
        p1_I = np.zeros(200,dtype=np.int)
        p2_I = np.zeros(200,dtype=np.int)
    
        c1 = s1 #current 1 index
        c2 = s2 #current 2 index
        cur_p_I = -1 #current pair index
        
        while c1 != e1 and c2 != e2:
            cur_p_I += 1
            
            #We are either going up or down based on which end we are
            #starting from (beggining or end)
            if e1 < s1:
                next1 = c1-1
                next2 = c2-1        
            else:
                next1 = c1+1
                next2 = c2+1
            
            #JAH: At this point
            #Need to handle indexing () vs [] and indexing spans (if any)
            #as well as 0 vs 1 based indexing (if any)
            try:
                v_n1c1 = xy1[:,next1] - xy1[:,c1]
            except:
                import pdb
                pdb.set_trace()
                
            v_n2c2 = xy2[:,next2] - xy2[:,c2]
            
            #216,231
            d_n1n2 = d[next1,next2]
            d_n1c2 = d[next1,c2]
            d_n2c1 = d[c1,next2]
            
            
            if d_n1c2 == d_n2c1 or (d_n1n2 <= d_n1c2 and d_n1n2 <= d_n2c1):
                #Advance along both contours
                
                p1_I[cur_p_I] = next1;
                p2_I[cur_p_I] = next2;
                
                c1 = next1;
                c2 = next2;
                
            elif np.sum((v_n1c1*v_n2c2) > 0):
                #contours go similar directions
                #follow smallest width
                if d_n1c2 < d_n2c1:
                    #consume smaller distance, then move the base of the vector
                    #further forward
                    p1_I[cur_p_I] = next1
                    p2_I[cur_p_I] = c2
                    
                    #This bit always confuses me
                    #c1  n1
                    #
                    #
                    #c2  x  x  x  n2
                    #
                    #Advance c1 so that d_n2_to_c1 is smaller next time
                    c1 = next1
                else:
                    p1_I[cur_p_I] = c1
                    p2_I[cur_p_I] = next2
                    c2 = next2
            else:
                
                if cur_p_I == 1:
                    prev_width = 0
                else:
                    prev_width = d[p1_I[cur_p_I-1],p2_I[cur_p_I-1]]

                if (d_n1c2 > prev_width and d_n2c1 > prev_width):
                    p1_I[cur_p_I] = next1
                    p2_I[cur_p_I] = next2
                    
                    c1 = next1
                    c2 = next2
                elif d_n1c2 < d_n2c1:
                    p1_I[cur_p_I] = next1
                    p2_I[cur_p_I] = c2
                    c1 = next1
                else:
                    p1_I[cur_p_I] = c1
                    p2_I[cur_p_I] = next2
                    c2 = next2

        p1_I = p1_I[:cur_p_I]
        p2_I = p2_I[:cur_p_I]
        
        return (p1_I,p2_I)
        #p1_I[cur_p_I+1:] = []
        #p2_I[cur_p_I+1:] = []


    @staticmethod
    def computeWidths(vulva_contours, non_vulva_contours):
        """
        
        """        

        #Widths:
        #------------------------------------
        #The caller:
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/linearSkeleton.m
        #see helper__skeletonize - callls seg_worm.cv.skeletonize
        #
        #
        #Initial skeletonization:
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bcv/skeletonize.m
        #
        #Some refinement:
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/cleanSkeleton.m        
        
        #Widths are simply the distance between two "corresponding" sides of
        #the contour. The question is how to get these two locations. 

        FRACTION_WORM_SMOOTH = 1.0/12.0
        SMOOTHING_ORDER = 3
        PERCENT_BACK_SEARCH = 0.3;
        PERCENT_FORWARD_SEARCH = 0.3;
        END_S1_WALK_PCT = 0.15;
        

        s_all = []
        widths_all = []

        for iFrame, (s1,s2) in enumerate(zip(vulva_contours,non_vulva_contours)):
            
            # * I'm writing the code based on awesome_contours_oh_yeah_v2
            #   in Jim's testing folder            
            
            if len(s1) == 0:
                s_all.append([])
                widths_all.append([])
                continue
            
            #Step 1: filter
            filter_width_s1 = WormParsing.h__roundToOdd(s1.shape[1]*FRACTION_WORM_SMOOTH)    
            s1[0,:] = sgolay(s1[0,:],filter_width_s1,SMOOTHING_ORDER)
            s1[1,:] = sgolay(s1[1,:],filter_width_s1,SMOOTHING_ORDER)

            filter_width_s2 = WormParsing.h__roundToOdd(s2.shape[1]*FRACTION_WORM_SMOOTH)    
            s2[0,:] = sgolay(s2[0,:],filter_width_s2,SMOOTHING_ORDER)
            s2[1,:] = sgolay(s2[1,:],filter_width_s2,SMOOTHING_ORDER)

            #TODO: Allow downsampling if the # of points is rediculous
            #200 points seems to be a good #
            #This operation gives us a matrix that is len(s1) x len(s2)
            dx_across = np.transpose(s1[0:1,:]) - s2[0,:]
            dy_across = np.transpose(s1[1:2,:]) - s2[1,:]
            d_across = np.sqrt(dx_across**2 + dy_across**2)
            dx_across = dx_across/d_across
            dy_across = dy_across/d_across
            
            #All s1 matching to s2
            #---------------------------------------
            left_I,right_I = WormParsing.h__getBounds(s1.shape[1],s2.shape[1],PERCENT_BACK_SEARCH,PERCENT_FORWARD_SEARCH)
            
            #%For each point on side 1, calculate normalized orthogonal values
            norm_x,norm_y = WormParsing.h__computeNormalVectors(s1)
                   
            #%For each point on side 1, find which side 2 the point pairs with
            dp_values1,match_I1 = WormParsing.h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across,d_across,left_I,right_I)

            I_1,I_2 = WormParsing.h__updateEndsByWalking(d_across,match_I1,s1,s2,END_S1_WALK_PCT)

            #TODO: Make this a function
            #------------------------------------
            #We're looking to the left and to the right to ensure that things are ordered
            #                           current is before next    current after previous
            is_good = np.hstack((True, np.array((I_2[1:-1] <= I_2[2:]) & (I_2[1:-1] >= I_2[:-2])), True))
            
            #is_good = [true; ((I_2(2:end-1) <= I_2(3:end)) & (I_2(2:end-1) >= I_2(1:end-2))); true];
            
            I_1 = I_1[is_good]
            I_2 = I_2[is_good]
            
            s1_x  = s1[0,I_1]
            s1_y  = s1[1,I_1]
            s1_px = s2[0,I_2] #s1_pair x
            s1_py = s2[1,I_2]
        
            #Final calculations
            #-----------------------
            #TODO: Allow smoothing on x & y
            widths1 = np.sqrt((s1_px-s1_x)**2 + (s1_py - s1_y)**2); #widths
            widths_all.append(widths1)
            
            skeleton_x = 0.5*(s1_x + s1_px)
            skeleton_y = 0.5*(s1_y + s1_py)
            s_all.append(np.vstack((skeleton_x,skeleton_y)));
            

        return (widths_all,s_all)
                
        """
            import matplotlib.pyplot as plt
            plt.scatter(vc[0,:],vc[1,:])
            plt.scatter(nvc[0,:],nvc[1,:])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
            
            plt.plot(x_plot,y_plot)
            plt.show()
            
            plt.scatter(s1[0,:],s1[1,:])
            plt.scatter(s2[0,:],s2[1,:])
            plt.scatter(skeleton_x,skeleton_y)
            plt.show()
        """




    @staticmethod
    def computeSkeletonLengths(nw,xy_all):
        """

        Computes the running length (cumulative distance from start - head?) 
        for each skeleton.
        
        
        Parameters
        ----------
        xy_all : [numpy.array]
            Contains the skeleton positions for each frame.
            List length: # of frames
            Each element contains a numpy array of size [n_points x 2]
            Skeleton 
        """
        n_frames = len(xy_all)
        data = np.full([config.N_POINTS_NORMALIZED,n_frames],np.NaN)
        for iFrame, cur_xy in enumerate(xy_all):
            if len(cur_xy) is not 0:
                sx = cur_xy[0,:]
                sy = cur_xy[1,:]
                cc = WormParsing.computeChainCodeLengths(sx,sy)
                data[:,iFrame] = WormParsing.normalizeParameter(nw,cc,cc)
                
        return data

    @staticmethod
    def computeChainCodeLengths(x,y):
        """
        Calculate the distance between a set of points and then calculate
        their cumulative distance from the first point.
        
        The first value returned has a value of 0 by definition.
        """
        
        #TODO: Should handle empty set - remove adding 0 as first element        
        
        #TODO: We need this for lengths as well, but the matrix vs vector 
        #complicates things
        
        dx = np.diff(x)
        dy = np.diff(y)
        
        distances = np.concatenate([np.array([0.0]), np.sqrt(dx**2 + dy**2)])
        return np.cumsum(distances)

    @staticmethod
    def normalizeAllFramesXY(nw,prop_to_normalize):
            
        n_frames = len(prop_to_normalize)
        norm_data = np.full([config.N_POINTS_NORMALIZED,2,n_frames],np.NaN)
        for iFrame, cur_frame_value in enumerate(prop_to_normalize):
            if len(cur_frame_value) is not 0:
                sx = cur_frame_value[0,:]
                sy = cur_frame_value[1,:]
                cc = WormParsing.computeChainCodeLengths(sx,sy)
                norm_data[:,0,iFrame] = WormParsing.normalizeParameter(nw,sx,cc)
                norm_data[:,1,iFrame] = WormParsing.normalizeParameter(nw,sy,cc)
        
        return norm_data            
    
    @staticmethod
    def normalizeAllFrames(nw,prop_to_normalize,xy_data):
            
        n_frames = len(prop_to_normalize)
        norm_data = np.full([config.N_POINTS_NORMALIZED,n_frames],np.NaN)
        for iFrame, (cur_frame_value,cur_xy) in enumerate(zip(prop_to_normalize,xy_data)):
            if len(cur_frame_value) is not 0:
                sx = cur_xy[0,:]
                sy = cur_xy[1,:]
                cc = WormParsing.computeChainCodeLengths(sx,sy)
                norm_data[:,iFrame] = WormParsing.normalizeParameter(nw,cur_frame_value,cc)
        
        return norm_data 

    @staticmethod
    def calculateAngles(self,skeletons):
    
        """
        #Angles
        #----------------------------------
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/skeleton.m
        #https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/%2Bseg_worm/%2Bcv/curvature.m
        #
        #   Note, the above code is written for the non-normalized worm ...
        #   edge_length= total_length/12
        #
        #   Importantly, the above approach calculates angles not between
        #   neighboring pairs but over a longer stretch of pairs (pairs that
        #   exceed the edge length). The net effect of this approach is to
        #   smooth the angles
        
        #vertex index - first one where the distance from the tip to this point
        #is greater than the edge length
        
        
        #s = norm_data[]
        
        
        #temp_s = np.full([config.N_POINTS_NORMALIZED,n_frames],np.NaN)
        #for iFrame in range(n_frames):
        #   temp_   
        """                  

        temp_angle_list = []
                      
        for iFrame, cur_frame_value in enumerate(skeletons):
            if len(cur_frame_value) is 0:
                temp_angle_list.append([])
            else:
                sx = cur_frame_value[0,:]
                sy = cur_frame_value[1,:]
                cc = WormParsing.computeChainCodeLengths(sx,sy)
    
                #This is from the old code
                edge_length = cc[-1]/12               
                
                #We want all vertices to be defined, and if we look starting
                #at the left_I for a vertex, rather than vertex for left and right
                #then we could miss all middle points on worms being vertices
                
                left_lengths = cc - edge_length
                right_lengths = cc + edge_length
    
                valid_vertices_I = utils.find((left_lengths > cc[0]) & (right_lengths < cc[-1]))
                
                left_lengths = left_lengths[valid_vertices_I]
                right_lengths = right_lengths[valid_vertices_I]                
                
                left_x = np.interp(left_lengths,cc,sx)
                left_y = np.interp(left_lengths,cc,sy)
            
                right_x = np.interp(right_lengths,cc,sx)
                right_y = np.interp(right_lengths,cc,sy)
    
                d2_y = sy[valid_vertices_I] - right_y
                d2_x = sx[valid_vertices_I] - right_x
                d1_y = left_y - sy[valid_vertices_I]
                d1_x = left_x - sx[valid_vertices_I] 
    
                frame_angles = np.arctan2(d2_y,d2_x) - np.arctan2(d1_y,d1_x)
                
                frame_angles[frame_angles > np.pi] -= 2*np.pi
                frame_angles[frame_angles < -np.pi] += 2*np.pi
                
                frame_angles *= 180/np.pi
                
                all_frame_angles = np.full_like(cc,np.NaN)
                all_frame_angles[valid_vertices_I] = frame_angles
                
                temp_angle_list.append(all_frame_angles)
                
        return WormParsing.normalizeAllFrames(self,temp_angle_list,skeletons)
    
    @staticmethod
    def normalizeParameter(self,orig_data,old_lengths):
        """
        
        This function finds where all of the new points will be when evenly
        sampled (in terms of chain code length) from the first to the last 
        point in the old data.

        These points are then related to the old points. If a new points is at
        an old point, the old point data value is used. If it is between two
        old points, then linear interpolation is used to determine the new value
        based on the neighboring old values.

        NOTE: For better or worse, this approach does not smooth the new data
        
        Old Code:
        https://github.com/openworm/SegWorm/blob/master/ComputerVision/chainCodeLengthInterp.m  
        
        Parameters:
        -----------
        non_normalizied_data :
            - ()
        """
        n_old = len(old_lengths)
        I = np.array(range(n_old))
        
        new_lengths = np.linspace(old_lengths[0],old_lengths[-1],config.N_POINTS_NORMALIZED)
        evaluation_I = np.interp(new_lengths,old_lengths,I)
        
        norm_data = np.interp(evaluation_I,I,orig_data)
        #TODO: Might just replace all of this with an interpolation call
        
        
        
        #For each point, get the bordering points
        #Sort, with old coming before new
        I = np.argsort(np.concatenate([old_lengths, new_lengths]), kind='mergesort')
        #Find new points, an old point will be to the left
        new_I = utils.find(I >= len(old_lengths)) #indices 0 to n-1, look for >= not >
        
        norm_data = np.empty_like(new_lengths)        
        
        #Can we do this without a loop (YES!)
        #find those that are equal
        #those that are not equal (at an old point) then do vector math        

        for iSeg,cur_new_I in enumerate(new_I):
            cur_left_I = I[cur_new_I-1]
            cur_right_I = cur_left_I + 1
            try:
                if iSeg == 0 or (iSeg == len(new_lengths) - 1) or (new_lengths[iSeg] == old_lengths[cur_left_I]):
                    norm_data[iSeg] = orig_data[cur_left_I]
                else:
                    new_position = new_lengths[iSeg]
                    left_position = old_lengths[cur_left_I]
                    right_position = old_lengths[cur_right_I]                    
                    total_length = right_position - left_position
                    #NOTE: If we are really close to left, then we want mostly
                    #left, which means right_position - new_position will almost
                    #be equal to the total length, and left_pct will be close to 1
                    left_pct = (right_position - new_position)/total_length
                    right_pct = (new_position - left_position)/total_length
                    norm_data[iSeg] = left_pct*orig_data[cur_left_I] + right_pct*orig_data[cur_right_I]
            except:
                import pdb
                pdb.set_trace()


        return norm_data        
