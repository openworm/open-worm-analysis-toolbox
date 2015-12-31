function s = awesome_contours_oh_yeah_v2(frame_values)
%
%   too much Despicable Me watching ...
%
%   Algorithm:
%   ----------
%   Find which point when taking the line from the current point to that
%   opposite point and the normal of the current point, makes the 2
%   parallel.
%
%   x   c   x
%      | |      left | is the normal for c, based on its gradient
%        |      right | is the line between c and the middle y
%        |      find which y makes the 2 lines most parallel (dot product maximized
%   y   y   y
%
%
%
%   Status:
%   -------
%   1) FIXED algorithm doesn't work on non parallel surfaces, consider a diamond
%    /|\
%   / | \
%   \ | /
%    \|/
%       This practically has an effect at the ends and needs to be fixed.
%      Interpolation of good values might fix this problem.
%
%   2) SOMEWHAT HANDLED Points are not guaranteed to be ordered, so a dot product would
%   need to be computed for subsequent points to look for reversals.
%
%   i.e., this would be fine for midpoints - no backtracking
%   1 2
%       3
%         4
%
%   this would not: point 3 would need to be removed ...
%   1   3 2 4
%
%   3) Spurs are not handled. It would be good to get an example of this
%   since I think we could handle this.
%
%   Interesting Frames:
%   -------------------
%   11 - good example of ends not being nicely shaped
%   261 - failure when the back/forth search is not pretty wide (0.3
%   instead of 0.2)
%   3601 - coiling that fails with dot product unless a search is done
%       for multiple peaks and the smaller width chosen
%
%   Problem Frames:
%   -------------------
%   1200 - ends are a bit messed up
%
% Original example contour & skeleton data from:
%

FRACTION_WORM_SMOOTH = 1/12;
PERCENT_BACK_SEARCH = 0.3;
PERCENT_FORWARD_SEARCH = 0.3;
END_S1_WALK_PCT = 0.15;

%file_path = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_contour_and_skeleton_info.mat';
%fp2 = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_video_norm_worm.mat';

base_path = 'C:\Users\RNEL\Google Drive\OpenWorm\OpenWorm Public\movement_analysis\example_data';

file_path = fullfile(base_path,'example_contour_and_skeleton_info.mat');
fp2 = fullfile(base_path,'example_video_norm_worm.mat');


%Intersection is a dangerous game because of the problem of being very
%close ...

h2 = load(fp2);

nw_widths = h2.s.widths;
nw_sx = h2.s.x;
nw_sy = h2.s.y;

h = load(file_path);

n_frames = length(h.all_vulva_contours);


n_frames = length(frame_values);
sx_all = cell(1,n_frames);
sy_all = cell(1,n_frames);
widths_all = cell(1,n_frames);
s1x_all = cell(1,n_frames);
s2x_all = cell(1,n_frames);
s1y_all = cell(1,n_frames);
s2y_all = cell(1,n_frames);

s = struct;

tic
%frame_value = 1;
for iFrame = frame_values %1:100:4642
    s1 = h.all_vulva_contours{iFrame};
    s2 = h.all_non_vulva_contours{iFrame};
    
    if isempty(s1)
        continue
    end
    


    filter_width_s1 = sl.math.roundToOdd(size(s1,1)*FRACTION_WORM_SMOOTH);
    s1(:,1) = sgolayfilt(s1(:,1),3,filter_width_s1);
    s1(:,2) = sgolayfilt(s1(:,2),3,filter_width_s1);
    
    filter_width_s2 = sl.math.roundToOdd(size(s2,1)*FRACTION_WORM_SMOOTH);
    s2(:,1) = sgolayfilt(s2(:,1),3,filter_width_s2);
    s2(:,2) = sgolayfilt(s2(:,2),3,filter_width_s2);
    
    
    %TODO: Allow downsampling if the # of points is rediculous
    %200 points seems to be a good #
    %This operation gives us a matrix that is len(s1) x len(s2)
    dx_across = bsxfun(@minus,s1(:,1),s2(:,1)');
    dy_across = bsxfun(@minus,s1(:,2),s2(:,2)');
    d_across  = sqrt(dx_across.^2 + dy_across.^2);
    dx_across = dx_across./d_across;
    dy_across = dy_across./d_across;
    
    %all s1 matching to s2
    %-------------------------
    
    %For every s1 point, compute the furthest left and right (backwards and
    %forwards we will go)
    [left_I,right_I] = h__getBounds(size(s1,1),size(s2,1),PERCENT_BACK_SEARCH,PERCENT_FORWARD_SEARCH);
    
    %For each point on side 1, calculate normalized orthogonal values
    [norm_x,norm_y]  = h__computeNormalVectors(s1);
    
    %For each point on side 1, find which side 2 the point pairs with
    [dp_values1,match_I1] = h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across,d_across,left_I,right_I);
    
    %{
    I_1 = 1:length(match_I1);
    I_2 = match_I1;
    %}
    
    %The ends don't project well across, so we start at the ends and 
    %walk a bit towards the center
    [I_1,I_2] = h__updateEndsByWalking(d_across,match_I1,s1,s2,END_S1_WALK_PCT);
    
    

    %We'll filter out 
    is_good = [true; ((I_2(2:end-1) <= I_2(3:end)) & (I_2(2:end-1) >= I_2(1:end-2))); true];
    
    I_1(~is_good) = [];
    I_2(~is_good) = [];
    
    
    s1_x  = s1(I_1,1);
    s1_y  = s1(I_1,2);
    s1_px = s2(I_2,1); %s1_pair x
    s1_py = s2(I_2,2);

    s1x_all{iFrame} = s1_x;
s2x_all{iFrame} = s1_y;
s1y_all{iFrame} = s1_px;
s2y_all{iFrame} = s1_py;
    
    %TODO: Allow smoothing on x & y
    skeleton_x = 0.5*(s1_x + s1_px);
    skeleton_y = 0.5*(s1_y + s1_py);
    widths1 = sqrt((s1_px-s1_x).^2 + (s1_py - s1_y).^2); %widths

    sx_all{iFrame} = skeleton_x;
    sy_all{iFrame} = skeleton_y;
    widths_all{iFrame} = widths1;

    %skeleton_x = csaps(1:length(skeleton_x),skeleton_x,

    
    %toc
    %Plotting Results
    %-------------------
    if true
        toc
        %     plot_s2_match = false;
        %
        %     if plot_s2_match
        %         dp_values = dp_values2;
        %         match_I   = match_I2;
        %         [s1,s2]   = deal(s2,s1);
        %         d_across  = d_across';
        %     else
        dp_values = dp_values1;
        %     end
        
        
%         offsets = s1(1,:);
%         
%         s1 = bsxfun(@minus,s1,offsets);
%         s2 = bsxfun(@minus,s2,offsets);
%         s1_x = bsxfun(@minus,s1_x,offsets(1));
        
        vc_raw  = s1;
        nvc_raw = s2;
    
        clf
        subplot(1,3,[1 2])
        hold on
        
        %Raw
%         plot(vc_raw(:,1),vc_raw(:,2),'r.')
%         plot(nvc_raw(:,1),nvc_raw(:,2),'b.')
        
        %Smooth
        plot(s1(2:end-1,1),s1(2:end-1,2),'ro')
        plot(s2(2:end-1,1),s2(2:end-1,2),'bo')
        
        plot(skeleton_x,skeleton_y,'d-')
        %     plot(fx,fy,'-k')
        
        
        for iPlot = 1:length(s1_x)
            %I2 = match_I(iPlot);
            
%             if iPlot == 5 %Start a bit in so we see it
%                 c = 'm';
%             elseif abs(dp_values(iPlot)) > 0.99
                c = 'g';
%             else
%                 c = 'k';
%             end
            x = [s1_x(iPlot) s1_px(iPlot)];
            y = [s1_y(iPlot) s1_py(iPlot)];
            %midpoint = [0.5*(x(1)+x(2)),0.5*(y(1)+y(2))];
            plot(x,y,c)
            %plot(midpoint(1),midpoint(2),'k.')
        end
        
        %     for iPlot = 1:size(s2,1)
        %         I1 = match_I2(iPlot);
        %         x = [s1(I1,1) s2(iPlot,1)];
        %         y = [s1(I1,2) s2(iPlot,2)];
        %         midpoint = [0.5*(x(1)+x(2)),0.5*(y(1)+y(2))];
        %         plot(midpoint(1),midpoint(2),'ko')
        %     end
        
        %plot(nw_sx(:,iFrame),nw_sy(:,iFrame),'x','Color',[0.3 0.3 0.3])
        
        hold off
        axis equal
        
%         subplot(1,3,3)
%         
%         plot(dp_values,'o-')
%         set(gca,'ylim',[-1 -0.5])
        
        
        %Width should really be plotted as a function of distance along the skeleton
        
        
        cum_dist = h__getSkeletonDistance(skeleton_x,skeleton_y);
        
        subplot(1,3,3)
        plot(cum_dist./cum_dist(end),widths1,'r.-')
%         hold on
%         plot(linspace(0,1,49),nw_widths(:,iFrame),'g.-')
%         hold off
        
        title(sprintf('iFrame %d',iFrame))
        xlabel('Normalized length')
        ylabel('Width (uM)')
        
        if length(frame_values) > 1
            %pause
            drawnow
            pause
            %pause(0.1)
        end
        
    end
end
toc

s.sx_all = sx_all;
s.sy_all = sy_all;
s.widths_all = widths_all;
s.s1x_all = s1x_all;
s.s2x_all = s2x_all;
s.s1y_all = s1y_all;
s.s2y_all = s2y_all;

end

function [I_1,I_2] = h__updateEndsByWalking(d_across,match_I1,s1,s2,END_S1_WALK_PCT)

    end_s1_walk_I = ceil(length(s1)*END_S1_WALK_PCT);
    end_s2_walk_I = 2*end_s1_walk_I;
    [p1_I,p2_I] = h__getPartnersViaWalk(1,end_s1_walk_I,1,end_s2_walk_I,d_across,s1,s2);
    
%     [wtf,wtf2] = skeletonize(1,end_s1_walk_I,1,1,end_s2_walk_I,1,s1,s2,false);
%     
%     wtf3 = zeros(1,36);
%     for i = 1:36
%         wtf3(i) = d_across(p1_I(i),p2_I(i));
%     end
    
    
    match_I1(p1_I) = p2_I;
    
    keep_mask = false(1,length(match_I1));
    keep_mask(p1_I) = true;
    
    n_s1 = length(s1);
    n_s2 = length(s2);
    end_s1_walk_backwards = n_s1 - end_s1_walk_I;
    end_s2_walk_backwards = n_s2 - end_s2_walk_I;
    
    
    [p1_I,p2_I] = h__getPartnersViaWalk(n_s1,end_s1_walk_backwards,n_s2,end_s2_walk_backwards,d_across,s1,s2);
    match_I1(p1_I) = p2_I;
    keep_mask(p1_I) = true;
    
    %anything in between we'll use the projection appproach
    keep_mask(end_s1_walk_I+1:end_s1_walk_backwards-1) = true;
    
    %Always keep ends
    keep_mask(1)   = true; 
    keep_mask(end) = true;

    match_I1(1) = 1;
    match_I1(end) = length(s2);
    

    %This isn't perfect but it removes some back and forth behavior
    %of the matching. We'd rather drop points and smooth
    I_1 = find(keep_mask);
    I_2 = match_I1(keep_mask);
end

function cum_dist = h__getSkeletonDistance(mid_x,mid_y)
dx = diff(mid_x);
dy = diff(mid_y);
d = [0; sqrt(dx.^2+dy.^2)];
cum_dist = cumsum(d);
end

function [left_I,right_I] = h__getBounds(n1,n2,p_left,p_right)

pct = linspace(0,1,n1);
left_pct = pct - p_left;
right_pct = pct + p_right;

left_I = floor(left_pct*n2);
right_I = ceil(right_pct*n2);
left_I(left_I < 1) = 1;
right_I(right_I > n2) = n2;
end

function [dp_values,match_I] = h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across,d_across,left_I,right_I)


n_s1 = size(s1,1);
match_I = zeros(n_s1,1);
match_I(1) = 1;
match_I(end) = size(s2,1);
dp_values = ones(size(s1,1),1);
dp_values(1) = -1;
dp_values(end) = -1;

all_signs_used = zeros(1,n_s1);

for I = 2:n_s1-1
    
    lb = left_I(I);
    rb = right_I(I);
    
    [abs_dp_value,dp_I,sign_used] = h__getProjectionIndex(norm_x(I),norm_y(I),dx_across(I,lb:rb),dy_across(I,lb:rb),lb,d_across(I,lb:rb),0);
    
    all_signs_used(I) = sign_used;
    
    dp_values(I) = abs_dp_value;
    match_I(I)   = dp_I;
end

%Here we check that the projection direction was similar across the
%skeleton. Sometimes some positions get unlucky and get projections that
%are close to -1 and 1, in which case it is unclear which we should keep.
%
%Problem Example: Frame 261 of example video
if ~all(all_signs_used(2:end-1) == all_signs_used(2))
    if sum(all_signs_used) > 0
        sign_use = 1;
        I_bad = find(all_signs_used(2:end-1) ~= 1) + 1;
    else
        I_bad = find(all_signs_used(2:end-1) ~= -1) + 1;
        sign_use = -1;
    end
    for I = I_bad
    
    lb = left_I(I);
    rb = right_I(I);
    
    [abs_dp_value,dp_I,sign_used] = h__getProjectionIndex(norm_x(I),norm_y(I),dx_across(I,lb:rb),dy_across(I,lb:rb),lb,d_across(I,lb:rb),sign_use);
    
    all_signs_used(I) = sign_used;
    
    dp_values(I) = abs_dp_value;
    match_I(I)   = dp_I;
    end
end

end

function [norm_x,norm_y] = h__computeNormalVectors(data)

dx = gradient(data(:,1));
dy = gradient(data(:,2));

%This approach gives us -1 for the projection
%We could also use:
%dx_norm = -dy;
%dy_norm = dx;
%
%and we would get 1 for the projection
dx_norm = dy;
dy_norm = -dx;

vc_d_magnitude = sqrt(dx_norm.^2 + dy_norm.^2);

norm_x = dx_norm./vc_d_magnitude;
norm_y = dy_norm./vc_d_magnitude;


end

function [dp_value,I,sign_used] = h__getProjectionIndex(vc_dx_ortho,vc_dy_ortho,dx_across_worm,dy_across_worm,left_I,d_across,sign_use)

%
%   sign_use : 
%       -  0, use anything
%       -  1, use positive
%       - -1, use negative

% nvc_local = nvc(nvc_indices_use,:);
%
% dx_across_worm = cur_point(1) - nvc_local(:,1);
% dy_across_worm = cur_point(2) - nvc_local(:,2);
%
% d_magnitude = sqrt(dx_across_worm.^2+dy_across_worm.^2);
%
% dx_across_worm = dx_across_worm./d_magnitude;
% dy_across_worm = dy_across_worm./d_magnitude;


%SPEED: Compute normalized distances for all pairs ...
%Might need to downsample

dp = dx_across_worm*vc_dx_ortho + dy_across_worm*vc_dy_ortho;

%I'd like to not have to do this step, it has to do with the relationship
%between the vulva and non-vulva side. This should be consistent across the
%entire animal and could be passed in, unless the worm rolls.
sign_used = -1;
if sign_use == 0 && sum(dp) > 0
    %Instead of multiplying by -1 we could hardcode the flip of the logic
    %below (e.g. max instead of min, > vs <)
    dp = -1*dp;
    sign_used = 1;
elseif sign_use == 1
    dp = -1*dp;
    sign_used = 1;
end

%This is slow, presumably due to the memory allocation ...
%               < right                         < left
%possible = [dp(1:end-1) < dp(2:end) false] & [false dp(2:end) < dp(1:end-1)];
possible = (dp(2:end-1) < dp(3:end)) & (dp(2:end-1) < dp(1:end-2));

Ip = find(possible);
if length(Ip) == 1
    dp_I = Ip+1;
    dp_value = dp(dp_I);
elseif length(Ip) > 1
    [~,temp_I] = min(d_across(Ip));
    dp_I     = Ip(temp_I)+1;
    dp_value = dp(dp_I);
else
    [dp_value,dp_I] = min(dp);
end

I = left_I + dp_I - 1;

end

function [p1_I,p2_I] = h__getPartnersViaWalk(s1,e1,s2,e2,d,xy1,xy2)
%
%   s1: start index for side 1
%   e1: end index for side 1
%
%   d :
%       distance from I1 to I2 is d(I1,I2)
%
%   d1 : [n x 2]
%       x,y pairs for side 1
%
%
%

%https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bcv/skeletonize.m


%TODO: remove hardcode
p1_I = zeros(1,200);
p2_I = zeros(1,200);

c1 = s1; %current 1 index
c2 = s2; %current 2 index
cur_p_I = 0; %current pair index


while c1 ~= e1 && c2 ~= e2
    cur_p_I = cur_p_I + 1;
    
    if e1 < s1
        next1 = c1-1;
        next2 = c2-1;        
    else
        next1 = c1+1;
        next2 = c2+1;
    end
    
%     scatter(xy1(:,1),xy1(:,2))
%     hold on
%     scatter(xy2(:,1),xy2(:,2))
%     plot(xy1(c1,1),xy1(c1,2),'+k')
%     plot(xy2(c2,1),xy2(c2,2),'+r')
%     plot(xy1(next1,1),xy1(next1,2),'dk')
%     plot(xy2(next2,1),xy2(next2,2),'dr')    
%     hold off
%     axis equal
    
    v_n1c1 = xy1(next1,:) - xy1(c1,:);
    
    %v_n2c2 = xy1(next2,:) - xy1(c2,:);
    v_n2c2 = xy2(next2,:) - xy2(c2,:);
    
    d_n1n2 = d(next1,next2);
    d_n1c2 = d(next1,c2); %d1
    d_n2c1 = d(c1,next2); %d2
    
    
    if d_n1c2 == d_n2c1 || (d_n1n2 <= d_n1c2 && d_n1n2 <= d_n2c1)
        %Advance along both contours
        
        p1_I(cur_p_I) = next1;
        p2_I(cur_p_I) = next2;
        
        c1 = next1;
        c2 = next2;
        option = 1;
    elseif v_n1c1*v_n2c2' > 0
    %Bug in old code
    %* Contour was the same, so this somehow never got caught
    %* Changed new code to a true dot product
    %elseif all((v_n1c1.*v_n2c2) > -1)
        %contours go similar directions
        %follow smallest width
        if d_n1c2 < d_n2c1
            %consume smaller distance, then move the base of the vector
            %further forward
            p1_I(cur_p_I) = next1;
            p2_I(cur_p_I) = c2;
            
            %This bit always confuses me
            %c1  n1
            %
            %
            %c2  x  x  x  n2
            %
            %Advance c1 so that d_n2_to_c1 is smaller next time
            c1 = next1;
            option = 2;
        else
            p1_I(cur_p_I) = c1;
            p2_I(cur_p_I) = next2;
            c2 = next2;
            option = 3;
        end
    else
        
        if cur_p_I == 1
            prev_width = 0;
        else
            prev_width = d(p1_I(cur_p_I-1),p2_I(cur_p_I-1));
        end
        
        if (d_n1c2 > prev_width && d_n2c1 > prev_width)
            p1_I(cur_p_I) = next1;
            p2_I(cur_p_I) = next2;
            
            c1 = next1;
            c2 = next2;
            option = 4;
        elseif d_n1c2 < d_n2c1
            %d1 < d2
            %use d1 i.e. n1,c2
            p1_I(cur_p_I) = next1;
            p2_I(cur_p_I) = c2;
            c1 = next1;
            option = 5;
        else
            p1_I(cur_p_I) = c1;
            p2_I(cur_p_I) = next2;
            c2 = next2;
            option = 6;
        end
        
    end
    
%     fprintf(1,'Option: %d\n',option);
%     for line_I = 1:cur_p_I
%        cur_1I = p1_I(line_I);
%        cur_2I = p2_I(line_I);
%        line([xy1(cur_1I,1),xy2(cur_2I,1)],[xy1(cur_1I,2),xy2(cur_2I,2)],'Color','k')
%     end
%     cur_1I = p1_I(cur_p_I);
%     cur_2I = p2_I(cur_p_I);
%     line([xy1(cur_1I,1),xy2(cur_2I,1)],[xy1(cur_1I,2),xy2(cur_2I,2)],'Color','k')
    
end

p1_I(cur_p_I+1:end) = [];
p2_I(cur_p_I+1:end) = [];


end

