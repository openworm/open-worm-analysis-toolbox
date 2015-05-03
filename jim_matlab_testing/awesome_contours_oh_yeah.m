function awesome_contours_oh_yeah()
%
%   too much Despicable Me watching ...
%
file_path = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_contour_and_skeleton_info.mat';

%Intersection is a dangerous game because of the problem of being very
%close ...



h = load(file_path);

vc = h.all_vulva_contours{1};
nvc = h.all_non_vulva_contours{1};

plot(vc(2:end-1,1),vc(2:end-1,2),'ro')
hold on
plot(nvc(2:end-1,1),nvc(2:end-1,2),'bo')
hold off



vc(:,1) = sgolayfilt(vc(:,1),3,sl.math.roundToOdd(size(vc,1)/12));
vc(:,2) = sgolayfilt(vc(:,2),3,sl.math.roundToOdd(size(vc,1)/12));

nvc(:,1) = sgolayfilt(nvc(:,1),3,sl.math.roundToOdd(size(nvc,1)/12));
nvc(:,2) = sgolayfilt(nvc(:,2),3,sl.math.roundToOdd(size(nvc,1)/12));


%A plot of the smoothed values ...
hold on
plot(vc(2:end-1,1),vc(2:end-1,2),'ko')

plot(nvc(2:end-1,1),nvc(2:end-1,2),'ko')
hold off


vc_dx = gradient(vc(:,1));
vc_dy = gradient(vc(:,2));

vc_dx_ortho = vc_dy;
vc_dy_ortho = -vc_dx;

vc_d_magnitude = sqrt(vc_dx_ortho.^2 + vc_dy_ortho.^2);

vc_dx_ortho_unit = vc_dx_ortho./vc_d_magnitude;
vc_dy_ortho_unit = vc_dy_ortho./vc_d_magnitude;

%The other orthogonal is [vc_dy,-vc_dx]


%Step 1:
%------------------------



%Let's look at an example point and see how this algorithm works
%=================================================================

TEST_POINT = 100; %VC

cur_point = vc(TEST_POINT,:);

indices_plot = TEST_POINT-2:TEST_POINT+2;
indices_plot2 = TEST_POINT-10:TEST_POINT+10;

clf
hold on
plot(vc(indices_plot,1),vc(indices_plot,2),'ko')

nvc_local = nvc(indices_plot2,:);


plot(nvc_local(:,1),nvc_local(:,2),'ko')



%For visualization:
%------------------
%The scale factor is made up for this example
close_x = cur_point(1)+100*vc_dx_ortho(TEST_POINT);
close_y = cur_point(2)+100*vc_dy_ortho(TEST_POINT);

d_across_worm = zeros(length(indices_plot2),2);
d_across_worm(:,1) = cur_point(1) - nvc_local(:,1);
d_across_worm(:,2) = cur_point(2) - nvc_local(:,2);

d_magnitude = sqrt(d_across_worm(:,1).^2+d_across_worm(:,2).^2);

d_across_worm = bsxfun(@rdivide,d_across_worm,d_magnitude);

%SPEED: Compute normalized distances for all pairs ...
%Might need to downsample

dp = d_across_worm(:,1)*vc_dx_ortho(TEST_POINT) + d_across_worm(:,2)*vc_dy_ortho(TEST_POINT);

[max_dp,max_dp_I] = max(abs(dp));

plot([vc(TEST_POINT,1) close_x],[vc(TEST_POINT,2) close_y],'r')
plot(nvc_local(max_dp_I,1),nvc_local(max_dp_I,2),'go')

hold off
axis equal





A = vc(10,:);
B = vc(11,:);

dx = B(1) - A(1);
dy = B(2) - A(2);

m1 = [-dy*1000 + A(1), dx*1000 + A(2)];
m2 = [dy*1000 + A(1), -dx*1000 + A(2)];

s = 11;
C = vc(11,:);
D = vc(12,:);

intersects(m1,m2,C,D)
%Might need a forward filter ... - don't include widths that come forward
%points if a better backwards points is later found

hold on
plot(vc(10:11,1),vc(10:11,2),'go')
plot(nvc(10:11,1),nvc(10:11,2),'ko')

end

%This algorithm doesn't work well ...
%http://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
function mask = intersects(A,B,C,D)
    mask =  ccw(A,C,D) ~= ccw(B,C,D) && ccw(A,B,C) ~= ccw(A,B,D);
end

function mask = ccw(A,B,C)
    mask = ((C(2)-A(2)) * (B(1)-A(1))) >= ((B(2)-A(2)) * (C(1)-A(1)));
end

% % % 
% % % def ccw(A,B,C):
% % %     return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
% % % 
% % % # Return true if line segments AB and CD intersect
% % % def intersect(A,B,C,D):
% % %     