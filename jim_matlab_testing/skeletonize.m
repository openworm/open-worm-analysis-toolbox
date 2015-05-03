function [skeleton,cWidths] = skeletonize(s1, e1, i1, s2, e2, i2, c1, c2, ...
    isAcross)
%SKELETONIZE Skeletonize takes the 2 pairs of start and end points on a
%contour(s), then traces the skeleton between them using the specified
%increments.
%
%   [SKELETON] = SKELETONIZE(S1, E1, I1, S2, E2, I2, C1, C2)
%
%   Inputs:
%       s1       - The starting index for the first contour segment.
%       e1       - The ending index for the first contour segment.
%       i1       - The increment to walk along the first contour segment.
%                  Note: a negative increment means walk backwards.
%                  Contours are circular, hitting an edge wraps around.
%       s2       - The starting index for the second contour segment.
%       e2       - The ending index for the second contour segment.
%       i2       - The increment to walk along the second contour segment.
%                  Note: a negative increment means walk backwards.
%                  Contours are circular, hitting an edge wraps around.
%       c1       - The contour for the first segment.
%       c2       - The contour for the second segment.
%
%       In the usage I've seen c1 and c2 are equivalent.
%
%
%       isAcross - Does the skeleton cut across, connecting s1 with e2?
%                  Otherwise, the skeleton simply traces the midline
%                  between both contour segments.
%
%       isAcross seems like it is basically always false ...
%
%   Output:
%       skeleton - the skeleton traced between the 2 sets of contour points.
%       cWidths  - the widths between the 2 sets of contour points.
%                  Note: there are no widths when cutting across.
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% The first starting index is before the ending one.
we1 = [];
if s1 <= e1
    
    % We are going forward.
    if i1 > 0
        size1 = (e1 - s1 + 1) / i1;
        
    % We are wrapping backward.
    else
        we1 = 1;
        ws1 = size(c1, 1);
        size1 = (s1 + size(c1, 1) - e1 + 1) / -i1;
    end
    
% The first starting index is after the ending one.
else
    % We are going backward.
    if i1 < 0
        size1 = (s1 - e1 + 1) / -i1;

    % We are wrapping forward.
    else
        we1 = size(c1, 1);
        ws1 = 1;
        size1 = (size(c1, 1) - s1 + 1 + e1) / i1;
    end
end

% The second starting index is before the ending one.
we2 = [];
if s2 <= e2
    
    % We are going forward.
    if i2 > 0
        size2 = (e2 - s2 + 1) / i2;
        
    % We are wrapping backward.
    else
        we2 = 1;
        ws2 = size(c2, 1);
        size2 = (s2 + size(c2, 1) - e2 + 1) / -i2;
    end
    
% The second starting index is after the ending one.
else
    % We are going backward.
    if i2 < 0
        size2 = (s2 - e2 + 1) / -i2;

    % We are wrapping forward.
    else
        we2 = size(c2, 1);
        ws2 = 1;
        size2 = (size(c2, 1) - s2 + 1 + e2) / i2;
    end
end

%{
plot(c1(:,1),c1(:,2),'ro')
hold all
plot(c1(j1,1),c1(j1,2),'go')
plot(c1(j2,1),c1(j2,2),'bo')
plot(c1(nextJ2,1),c1(nextJ2,2),'co')
plot(c1(nextJ1,1),c1(nextJ1,2),'ko')


%}

p1_I = zeros(1,100);
p2_I = zeros(1,100);
% Trace the midline between the contour segments.
% Note: the next few pages of code represent multiple, nearly identical
% algorithms. The reason they are inlined as separate instances is to
% mildly speed up one of the tightest loops in our program.
skeleton = zeros(floor(size1 + size2), 2); % pre-allocate memory
cWidths = []; % there are no widths when cutting across
j1 = s1;
j2 = s2;
if ~isAcross
    
    % Initialize the skeleton and contour widths.
    skeleton(1,:) = round((c1(j1,:) + c2(j2,:)) ./ 2);
    cWidths = zeros(size(skeleton, 1), 1); % pre-allocate memory
    cWidths(1) = sqrt(sum((c1(j1,:) - c2(j2,:)) .^ 2));
    if j1 == we1 % wrap
        j1 = ws1;
    end
    if j2 == we2 % wrap
        j2 = ws2;
    end
    sLength = 2;
    
    % Skeletonize the contour segments and measure the width.
    while j1 ~= e1 && j2 ~= e2
        
        % Compute the widths.
        nextJ1 = j1 + i1;
        if nextJ1 == we1 % wrap
            nextJ1 = ws1;
        end
        nextJ2 = j2 + i2;
        if nextJ2 == we2 % wrap
            nextJ2 = ws2;
        end
        dnj1 = c1(nextJ1,:) - c1(j1,:);
        dnj2 = c1(nextJ2,:) - c1(j2,:);
        d12 = sum((c1(nextJ1,:) - c2(nextJ2,:)) .^ 2);
        d1 = sum((c1(nextJ1,:) - c2(j2,:)) .^ 2);
        d2 = sum((c1(j1,:) - c2(nextJ2,:)) .^ 2);
        
% % Code for debugging purposes.
%         disp(['j1=' num2str(j1) ' j2=' num2str(j2) ...
%             ' *** dnj1=[' num2str(dnj1) '] dnj2=[' num2str(dnj2) ...
%             '] *** d12=' num2str(sqrt(d12)) ...
%             ' d1=' num2str(sqrt(d1)) ' d2=' num2str(sqrt(d2))])
        
        % Advance along both contours.
        if (d12 <= d1 && d12 <= d2) || d1 == d2
            j1 = nextJ1;
            j2 = nextJ2;
            cWidths(sLength) = sqrt(d12);
            option = 1
        % The contours go in similar directions.
        % Follow the smallest width.
        elseif all((dnj1 .* dnj2) > -1)
            
            % Advance along the the first contour.
            if d1 <= d2
                j1 = nextJ1;
                cWidths(sLength) = sqrt(d1);
                option = 2
            % Advance along the the second contour.
            else
                j2 = nextJ2;
                cWidths(sLength) = sqrt(d2);
                option = 3
            end
            
        % The contours go in opposite directions.
        % Follow decreasing widths or walk along both contours.
        % In other words, catch up both contours, then walk along both.
        % Note: this step negotiates hairpin turns and bulges.
        else
            
            % Advance along both contours.
            prevWidth = cWidths(sLength - 1) ^ 2;
            if (d12 <= d1 && d12 <= d2) || d1 == d2 ...
                    || (d1 > prevWidth && d2 > prevWidth )
                j1 = nextJ1;
                j2 = nextJ2;
                cWidths(sLength) = sqrt(d12);
                option = 4
            % Advance along the the first contour.
            elseif d1 < d2
                j1 = nextJ1;
                cWidths(sLength) = sqrt(d1);
                option = 5
            % Advance along the the second contour.
            else
                j2 = nextJ2;
                cWidths(sLength) = sqrt(d2);
                option = 6
            end
        end
        
        % Compute the skeleton.
        skeleton(sLength,:) = round((c1(j1,:) + c2(j2,:)) ./ 2);
        sLength = sLength + 1;
    end
    
    % Add the last point.
    if j1 ~= e1 || j2 ~= e2
        skeleton(sLength,:) = round((c1(e1,:) + c2(e2,:)) ./ 2);
        cWidths(sLength) = sqrt(sum((c1(e1,:) - c2(e2,:)) .^ 2));
        sLength = sLength + 1;
    end

    % Collapse any extra memory.
    skeleton(sLength:end,:) = [];
    cWidths(sLength:end) = [];
    
% The skeleton cuts across, connecting s1 with e2.
else

    % Initialize the connections.
    connect = zeros(size(skeleton, 1), 2); % pre-allocate memory
    connect(1,1) = j1;
    connect(1,2) = j2;
    if j1 == we1 % wrap
        j1 = ws1;
    end
    if j2 == we2 % wrap
        j2 = ws2;
    end
    prevWidth = sum((c1(j1,:) - c2(j2,:)) .^ 2);
    sLength = 2;
    
    % Connect the contour segments.
    while j1 ~= e1 && j2 ~= e2
        
        % Compute the widths.
        nextJ1 = j1 + i1;
        if nextJ1 == we1 % wrap
            nextJ1 = ws1;
        end
        nextJ2 = j2 + i2;
        if nextJ2 == we2 % wrap
            nextJ2 = ws2;
        end
        dnj1 = c1(nextJ1,:) - c1(j1,:);
        dnj2 = c1(nextJ2,:) - c1(j2,:);
        d12 = sum((c1(nextJ1,:) - c2(nextJ2,:)) .^ 2);
        d1 = sum((c1(nextJ1,:) - c2(j2,:)) .^ 2);
        d2 = sum((c1(j1,:) - c2(nextJ2,:)) .^ 2);
        
% Code for debugging purposes.
%         
%         disp(['j1=' num2str(j1) ' j2=' num2str(j2) ...
%             ' *** dnj1=[' num2str(dnj1) '] dnj2=[' num2str(dnj2) ...
%             '] *** d12=' num2str(sqrt(d12)) ...
%             ' d1=' num2str(sqrt(d1)) ' d2=' num2str(sqrt(d2))])
        
        % Advance along both contours.
        if (d12 <= d1 && d12 <= d2) || d1 == d2
            j1 = nextJ1;
            j2 = nextJ2;
            prevWidth = d12;
            
        % The contours go in similar directions.
        % Follow the smallest width.
        elseif all((dnj1 .* dnj2) > -1)
            
            % Advance along the the first contour.
            if d1 <= d2
                j1 = nextJ1;
                prevWidth = d1;
                
            % Advance along the the second contour.
            else
                j2 = nextJ2;
                prevWidth = d2;
            end
            
        % The contours go in opposite directions.
        % Follow decreasing widths or walk along both contours.
        % In other words, catch up both contours, then walk along both.
        % Note: this step negotiates hairpin turns and bulges.
        else
            
            % Advance along both contours.
            if (d12 <= d1 && d12 <= d2) || d1 == d2 ...
                    || (d1 > prevWidth && d2 > prevWidth )
                j1 = nextJ1;
                j2 = nextJ2;
                prevWidth = d12;
                
            % Advance along the the first contour.
            elseif d1 < d2
                j1 = nextJ1;
                prevWidth = d1;
                
            % Advance along the the second contour.
            else
                j2 = nextJ2;
                prevWidth = d2;
            end
        end
            
        % Compute the skeleton.
        connect(sLength,1) = j1;
        connect(sLength,2) = j2;
        sLength = sLength + 1;
    end
    
    % Add the last point.
    if j1 ~= e1 || j2 ~= e2
        connect(sLength,1) = e1;
        connect(sLength,2) = e2;
        sLength = sLength + 1;
    end
    
    % Collapse any extra memory.
    skeleton(sLength:end,:) = [];
    sLength = sLength - 1;
    
    % Setup the weights for the connections.
    i = 1:sLength;
    weights = 2 * (sLength - i) / (sLength - 1);
    w1 = [weights; weights]';
    w2 = flipud(w1);
    
    % Compute the skeleton across the connections.
    skeleton(i,:) = (w1 .* c1(connect(i,1),:) + ...
        w2 .* c2(connect(i,2),:)) ./ 2;
end
end
