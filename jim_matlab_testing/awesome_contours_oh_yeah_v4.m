function awesome_contours_oh_yeah_v4(frame_values)
%
%
%   SEEMS TO SENSITIVE TO PARAMETERS
%
%   This algorithm grows the skeleton by going out a certain
%   distance from the last skeleton point and then rotating that point
%   until the widths to either side are roughly equal.
%
%   Issues:
%   -------
%   1) This is really slow, but I think this can be sped up significantly.
%   
%       - # of angles to test - use line fits to estimate best rotation
%       - is testing for an intersection any quicker than just trying to
%       find the intersection? - presumably if we hit the intersection 
%       often, then we're better off just calculating it
%       - Take advantage of the smooth change in rotation to know which
%         segment to test for the intersection
%       - start with small rotations then expand, since it is likely
%       that we only need small rotations as we are advancing
%
%   2) We need to hold onto the solution values
%
%   3) The ends are bit messy. We should probably start at both ends
%   and merge in the middle.
%
%   
%

%

%Frame 877 - the frame that motivated this version


STEP_SIZE = 2; %This should really be a percentage
%As this gets small, the angles we need to test get really large

FRACTION_WORM_SMOOTH = 1/12;

file_path = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_contour_and_skeleton_info.mat';
fp2 = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_video_norm_worm.mat';

%Intersection is a dangerous game because of the problem of being very
%close ...

h2 = load(fp2);

nw_widths = h2.s.widths;
nw_sx = h2.s.x;
nw_sy = h2.s.y;

h = load(file_path);

n_frames = length(h.all_vulva_contours);
sx_all = cell(1,n_frames);
sy_all = cell(1,n_frames);
widths_all = cell(1,n_frames);

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
    
    s2(1,:)   = s1(1,:);
    s2(end,:) = s1(end,:);
    
    figure(1)
    clf
    scatter(s1(:,1),s1(:,2))
    hold on
    scatter(s2(:,1),s2(:,2))
    axis equal
    
    
    %line([s1(:,1) s1(:,1)+norm_xs1*sc1]',[s1(:,2) s1(:,2)+norm_ys1*sc1]','Color','k')
    %line([s2(:,1) s2(:,1)+norm_xs2*sc1]',[s2(:,2) s2(:,2)+norm_ys2*sc1]','Color','r')
    
    %line([s1(:,1) s2(I2,1)]',[s1(:,2) s2(I2,2)]','Color','k')
 
    sc = 1000;
    sc_plot = 100;
    cur_p = s1(1,:);
    next1 = 1+STEP_SIZE;
    next2 = 1+STEP_SIZE;
    tic
    while true

    hold on
    plot(s1(next1,1),s1(next1,2),'k+')
    plot(s2(next2,1),s2(next2,2),'k+')
    hold off
    
    
    temp_mid = h__getMidpoint(s1(next1,:),s2(next2,:));
    s_vector = temp_mid - cur_p;
    orig_s_vector = s_vector;
    %Some angles to select from ...
    
    %This span needs to be smarter ...
        rotation_angles = -16:2:16;

    
    widths_all = zeros(length(rotation_angles),2);
    I1_used = zeros(1,length(rotation_angles));
    I2_used = zeros(1,length(rotation_angles));
    indices_try_1 = next1+[0 -1 1 -2 2 -3 3 -4 4 -5 5 -6 6 -7 7 -8 8 -9 9];
    indices_try_2 = next2+[0 -1 1 -2 2 -3 3 -4 4 -5 5 -6 6 -7 7 -8 8 -9 9];
    if next1 < 10
    indices_try_1(indices_try_1 < 1) = [];
    end
    if next2 < 10
    indices_try_2(indices_try_2 < 1) = [];
    end
    %TODO: Add on checks for near end
    for I = 1:length(rotation_angles)
        rotation_angle = rotation_angles(I);
        s_vector = h__rotateVector(orig_s_vector,rotation_angle*pi/180);
        temp_mid = cur_p+s_vector;
        norm1 = h__computeNormalVectors2(s_vector,0);
        norm2 = h__computeNormalVectors2(s_vector,1);
    
        [widths_all(I,1),I1_used(I)] = h__getWidth(temp_mid,temp_mid+norm1*sc,s1,indices_try_1,true);
        [widths_all(I,2),I2_used(I)] = h__getWidth(temp_mid,temp_mid+norm2*sc,s2,indices_try_2,false);
    end
    
    metric = abs((widths_all(:,1) - widths_all(:,2)))./mean(widths_all,2);
    
    %TODO: Try and improve the accuracy by estimating 0 from the data
    %points obtained and then recompute with that rotation value
    
    [~,I] = min(metric);
    
    s_vector = h__rotateVector(orig_s_vector,rotation_angles(I)*pi/180);    
    temp_mid = cur_p+s_vector;
    
    %For plotting result ...
    norm1 = h__computeNormalVectors2(s_vector,0);
    norm2 = h__computeNormalVectors2(s_vector,1);
    hold on
    h__drawLine(cur_p,temp_mid,'Color','k')
    h__drawLine(temp_mid,temp_mid+norm1*sc_plot,'Color','r')
    h__drawLine(temp_mid,temp_mid+norm2*sc_plot,'Color','g')
    hold off
    
    
    
    cur_p = temp_mid(1,:);
    next1 = I1_used(I)+STEP_SIZE;
    next2 = I2_used(I)+STEP_SIZE;
    
    if (next1+10 >= length(s1)) || (next2+10 >= length(s2))
       break 
    end
    

    
    %{
        s_vector = h__rotateVector(orig_s_vector,12*pi/180);    
    temp_mid = cur_p+s_vector;
     norm1 = h__computeNormalVectors2(s_vector,0);
    norm2 = h__computeNormalVectors2(s_vector,1);
    h__drawLine(cur_p,temp_mid,'Color','k')
    h__drawLine(temp_mid,temp_mid+norm1*sc,'Color','r')
    h__drawLine(temp_mid,temp_mid+norm2*sc,'Color','g')   
    
    
    %}
    
    
    end
    toc
    %return

    if length(frame_values) > 1
        title(sprintf('Frame: %d',iFrame))
        pause
    end

end
toc




end

function [width,I] = h__getWidth(temp_mid,end_point,s,indices_to_try,is_side_1)

    width = [];
    for I = indices_to_try
        

        if intersects(temp_mid,end_point,s(I:I+1,:));

            [x,y] = getIntersectionPoint(temp_mid,end_point,s(I:I+1,:));
            width = sqrt((x - temp_mid(1))^2 + (y - temp_mid(2))^2);
            
%             hold on
%             plot(x,y,'ks')
%             hold off
            
            break
        end
    end
    
    if isempty(width)
        if is_side_1
            width = 10000;
        else
            width = 1000;
        end
    end
%     if isempty(width)
%        error('Unable to find intersection point') 
%     end



end

function mask = intersects(A,B,s)
%I'm not sure that I trust this when the line nearly hits one of the points
%in the other line
%
% dx = B(1) - A(1);
% dy = B(2) - A(2);
% 
% m1 = [-dy*1000 + A(1), dx*1000 + A(2)];
% m2 = [dy*1000 + A(1), -dx*1000 + A(2)];
% 
% s = 11;
% C = vc(11,:);
% D = vc(12,:);
% 
% intersects(m1,m2,C,D)

C = s(1,:);
D = s(2,:);

    mask =  ccw(A,C,D) ~= ccw(B,C,D) && ccw(A,B,C) ~= ccw(A,B,D);
end

function mask = ccw(A,B,C)
    mask = ((C(2)-A(2)) * (B(1)-A(1))) >= ((B(2)-A(2)) * (C(1)-A(1)));
end

function [x,y] = getIntersectionPoint(p1,p2,p34)

%[P1X P1Y P2X P2Y], you must provide two has arguments
%Example:
%l1=[93 388 120 354];
%l2=[102 355 124 377];

l1 = [p1 p2];
l2 = [p34(1,:) p34(2,:)];

ml1=(l1(4)-l1(2))/(l1(3)-l1(1));
ml2=(l2(4)-l2(2))/(l2(3)-l2(1));
bl1=l1(2)-ml1*l1(1);
bl2=l2(2)-ml2*l2(1);
b=[bl1 bl2]';
a=[1 -ml1; 1 -ml2];
Pint=a\b;

x=Pint(2);
y=Pint(1);

% % % %http://www.mathworks.com/matlabcentral/fileexchange/32827-lines-intersection/content//findintersection.m
% % % x1 = p1(1);
% % % y1 = p1(2);
% % % x2 = p2(1);
% % % y2 = p2(2);
% % % x3 = p34(1,1);
% % % y3 = p34(1,2);
% % % x4 = p34(2,1);
% % % y4 = p34(2,1);
% % % 
% % % x = det([det([x1 y1;x2 y2]), (x1-x2);det([x3 y3;x4 y4]), (x3-x4) ])/det([(x1-x2),(y1-y2) ;(x3-x4),(y3-y4)]);
% % % 
% % % y = det([det([x1 y1;x2 y2]), (y1-y2);det([x3 y3;x4 y4]), (y3-y4) ])/det([(x1-x2),(y1-y2) ;(x3-x4),(y3-y4)]);

end


function dp = h__getDP(side_xy,mid_point,skeleton_norm)

v_to_mid = [side_xy(:,1)-mid_point(1) side_xy(:,2)-mid_point(2)];

v_length = sqrt(sum(v_to_mid.^2,2));
v_to_mid(:,1) = v_to_mid(:,1)./v_length;
v_to_mid(:,2) = v_to_mid(:,2)./v_length;

dp = v_to_mid*skeleton_norm';

end
function result = h__rotateVector(v,angle)

%
%   v : [1 x 2]

result = v*[cos(angle) sin(angle); -sin(angle) cos(angle)];

end
function h__drawLine(p1,p2,varargin)

line([p1(:,1) p2(:,1)],[p1(:,2) p2(:,2)],varargin{:});
end



function mid = h__getMidpoint(p1,p2)
mid = 0.5*[p1(1)+p2(1) p1(2)+p2(2)];
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

function [norm_x,norm_y] = h__computeNormalVectors(data,option)

dx = gradient(data(:,1));
dy = gradient(data(:,2));

%This approach gives us -1 for the projection
%We could also use:
if option == 0
    dx_norm = -dy;
    dy_norm = dx;
else
    dx_norm = dy;
    dy_norm = -dx;
end

vc_d_magnitude = sqrt(dx_norm.^2 + dy_norm.^2);

norm_x = dx_norm./vc_d_magnitude;
norm_y = dy_norm./vc_d_magnitude;


end

function norm_vectors = h__computeNormalVectors2(d_data,option)
%
%
%   d_data : [n x 2]
%This approach gives us -1 for the projection
%We could also use:
if option == 0
    dx_norm = -d_data(:,2);
    dy_norm = d_data(:,1);
else
    dx_norm = d_data(:,2);
    dy_norm = -d_data(:,1);
end

vc_d_magnitude = sqrt(dx_norm.^2 + dy_norm.^2);

norm_x = dx_norm./vc_d_magnitude;
norm_y = dy_norm./vc_d_magnitude;
norm_vectors = [norm_x norm_y];
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
    
    scatter(xy1(:,1),xy1(:,2))
    hold on
    scatter(xy2(:,1),xy2(:,2))
    plot(xy1(c1,1),xy1(c1,2),'+k')
    plot(xy2(c2,1),xy2(c2,2),'+r')
    plot(xy1(next1,1),xy1(next1,2),'dk')
    plot(xy2(next2,1),xy2(next2,2),'dr')    
    hold off
    axis equal
    
    v_n1c1 = xy1(next1,:) - xy1(c1,:);
    v_n2c2 = xy2(next2,:) - xy2(c2,:);
    
    d_n1n2 = d(next1,next2);
    d_n1c2 = d(next1,c2);
    d_n2c1 = d(c1,next2);
    
    
    if d_n1c2 == d_n2c1 || (d_n1n2 <= d_n1c2 && d_n1n2 <= d_n2c1)
        %Advance along both contours
        
        p1_I(cur_p_I) = next1;
        p2_I(cur_p_I) = next2;
        
        c1 = next1;
        c2 = next2;
        option = 1;
    elseif all((v_n1c1.*v_n2c2) > -1)
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
    
    fprintf(1,'Option: %d\n',option);
    for line_I = 1:cur_p_I
       cur_1I = p1_I(line_I);
       cur_2I = p2_I(line_I);
       line([xy1(cur_1I,1),xy2(cur_2I,1)],[xy1(cur_1I,2),xy2(cur_2I,2)],'Color','k')
    end
    cur_1I = p1_I(cur_p_I);
    cur_2I = p2_I(cur_p_I);
    line([xy1(cur_1I,1),xy2(cur_2I,1)],[xy1(cur_1I,2),xy2(cur_2I,2)],'Color','k')
    
end

p1_I(cur_p_I+1:end) = [];
p2_I(cur_p_I+1:end) = [];


end

