function createCleanPreNormData()
%
%   NOTE: We're starting with temporary files that were created by
%   running the MRC GUI on a video
%
%   This function should take the awkward file structures and output:
%   - contours, non-normalized, oriented from head to tail
%       - ??? a single contour with indices
%       - left and right contours???
%       - in real units
%   - skeletons
%   - is_segmented mask
%   - anything related to the vulva or
%
%   - stage movement and origins need to be handled .... akdsfkakjdsfkasdfk
%
%       load(getRelativePath(stageMovementFile), 'movesI', 'locations');
%       origins = locations;
%
%   Where does the scaling factor come from???
%   fileInfo.expList.xml;

%This is the main function of importance
%https://github.com/openworm/SegWorm/blob/master/Worms/Util/normWorms.m
%See Also:
%https://github.com/openworm/SegWorm/blob/master/Worms/Util/norm2Worm.m
%%https://github.com/openworm/SegWorm/blob/master/Pipeline/normWormProcess.m

SAMPLES = 49;

base_path = 'F:\Projects\OpenWorm\worm_data\segworm_data\video\testing_with_GUI\.data';

s = h__getInfo(base_path);

%--------------------------------------------------------------------------

%SAMPLES
%rotation
%pixel2MicronScale
%origins
%movesI

mat_path = fullfile(base_path,'mec-4 (u253) off food x_2010_04_21__17_19_20__1_seg');

n_files  = 10;
n_frames = 9*500 + 142; %part 10 only has 142 frames
all_vulva_contours = cell(1,n_frames);
all_non_vulva_contours = cell(1,n_frames);
all_skeletons = cell(1,n_frames);
is_valid  = true(1,n_frames);


is_stage_movement = false(1,n_frames);

moves_I = s.movesI;
%NOTE: This is a 2 column vector of starts and stops, seems to be
%1 based but the first row is [0 0]

if moves_I(1) == 0
    if moves_I(1,2) == 0
        moves_I(1,:) = [];
    else
        error('Not sure what to do')
    end
end

for iRow = 1:size(moves_I)
    cur_start = moves_I(iRow,1);
    cur_end   = moves_I(iRow,2);
    is_stage_movement(cur_start:cur_end) = true;
end

iFrame = 0;
cur_stage_origin_I = 1;
for iFile = 1:n_files
    h = load(fullfile(mat_path,sprintf('block%d.mat',iFile)));
    
    all_entries = h.(sprintf('block%d',iFile));
    
    for iEntry = 1:length(all_entries)
        iFrame = iFrame + 1;
        cur_entry = all_entries{iEntry};
                
        if cur_stage_origin_I <= size(moves_I,1) && moves_I(cur_stage_origin_I,1) == iFrame
            cur_stage_origin_I = cur_stage_origin_I + 1;
        end

        
        if ~iscell(cur_entry) || is_stage_movement(iFrame)
            %empty - dropped frame - d
            %stage - 2 - m
            %parse failure - 1 - f
            
            
            is_valid(iFrame) = false;
            continue
        end
        
        
        frame_number = cur_entry{1}{1} + 1;
        if frame_number ~= iFrame
            error('Frame # mismatch')
        end
        
        %Contour information - in pixels, not microns :/
        %-----------------------------------------------
        c_pixels = cur_entry{2}{1}; %[n x 2]
        c_head_I = cur_entry{2}{6};
        c_tail_I = cur_entry{2}{7};
        
        %I think this is redundant information if we have the pixels ...
        c_cc_lengths = cur_entry{2}{8};
        %This is apparently cumulatively summed and for some
        %reason the wraparound length seems to be at the first index
        %rather than at the end ...
        %temp_cc_lengths = sqrt(sum(diff(c_pixels,1,1).^2,2));
        
        
        %Skeleton information
        %-----------------------------------------------
        s_pixels = cur_entry{3}{1}; %[n x 2]
        
        s_angles = cur_entry{3}{6}; %[n x 1] %How is this computed???
        %There appears to be some sort of smoothing going on.
        %The first 20 values are not valid - NaN
        %The last 20 values are not valid - NaN
        
        s_cc_lengths = cur_entry{3}{8}; %[n x 1], first value is 0
        s_widths = cur_entry{3}{9};
        
        %I think this means that the currently identified head and tail
        %are incorrect. I am choosing what I think is a much simpler
        %approach than was used in the old code
        head_flipped = cur_entry{8}{1}{1};
        
        %This however - based on the old code - seems to be correct
        %for the true head (it would be good to check this ...)
        vulva_clockwise_from_head = cur_entry{8}{2}{1};
        
        
        
        c_microns = h__pixels2Microns(s,c_pixels,cur_stage_origin_I);
        s_microns = h__pixels2Microns(s,s_pixels,cur_stage_origin_I);     
        
        
        %Step 1 - orient everything ...
        if head_flipped
            [c_head_I,c_tail_I] = deal(c_tail_I,c_head_I);
            s_microns = flipud(s_microns);
        end

        n_contour = size(c_microns,1);
        
        if c_head_I <= c_tail_I
            cw_I = c_head_I:c_tail_I;
            ccw_I = fliplr([c_tail_I:n_contour, 1:c_head_I]);
        else % cHeadI > cTailI
            cw_I = [c_head_I:n_contour, 1:c_tail_I];
            ccw_I = fliplr(c_tail_I:c_head_I);
        end
                
        if vulva_clockwise_from_head
            vulva_contours = c_microns(cw_I,:);
            non_vulva_contours = c_microns(ccw_I,:);
        else
            vulva_contours = c_microns(ccw_I,:);
            non_vulva_contours = c_microns(cw_I,:);            
        end
        
        all_vulva_contours{iFrame} = vulva_contours;
        all_non_vulva_contours{iFrame} = non_vulva_contours;
        all_skeletons{iFrame} = s_microns;
    end
    
end

save('example_contour_and_skeleton_info.mat','-v7.3','all_vulva_contours','all_non_vulva_contours','all_skeletons','is_stage_movement','is_valid');

end

function s = h__getInfo(base_path)

stage_file_name = 'mec-4 (u253) off food x_2010_04_21__17_19_20__1_stageMotion.mat';
stage_file_path = fullfile(base_path,stage_file_name);

h = load(stage_file_path);
s.origins = h.locations;
s.movesI = h.movesI;

calibration_file_name = 'mec-4 (u253) off food x_2010_04_21__17_19_20__1.info.xml';
calibration_file_path = fullfile(base_path,calibration_file_name);

xml = xmlread(calibration_file_path);

%This isn't clean, but it works for now.
%configuration.info.stage.steps.equivalent.microns
%configuration.info.stage.steps.equivalent.pixels

x_nodes = xml.getElementsByTagName('x');
y_nodes = xml.getElementsByTagName('y');

micronsX = str2double(x_nodes.item(0).item(0).getNodeValue);
micronsY = str2double(y_nodes.item(0).item(0).getNodeValue);

pixelsX = str2double(x_nodes.item(1).item(0).getNodeValue);
pixelsY = str2double(y_nodes.item(1).item(0).getNodeValue);

% Compute the microns/pixels.
%pixel2MicronScale = [pixelsX / micronsX, pixelsY / micronsY];
pixel2MicronX = pixelsX / micronsX;
pixel2MicronY = pixelsY / micronsY;
normScale = sqrt((pixel2MicronX ^ 2 + pixel2MicronY ^ 2) / 2);
s.pixel2MicronScale =  normScale * [sign(pixel2MicronX) sign(pixel2MicronY)];

% Compute the rotation matrix.
angle = atan(pixel2MicronY / pixel2MicronX);
if angle > 0
    angle = pi / 4 - angle;
else
    angle = pi / 4 + angle;
end
cosAngle = cos(angle);
sinAngle = sin(angle);
s.rotation = [cosAngle, -sinAngle; sinAngle, cosAngle];



end

function microns = h__pixels2Microns(s,pixels,cur_stage_origin)
%
%   s : struct
%       Example:
%               origins: [174x2 double]
%                movesI: [175x2 double]
%     pixel2MicronScale: [-3.9462 -3.9462]
%              rotation: [2x2 double]
%
%   https://github.com/openworm/SegWorm/blob/master/Worms/StageMovement/microns2Pixels.m
%   https://github.com/openworm/SegWorm/blob/master/Worms/StageMovement/pixels2Microns.m
%   

origins  = s.origins;
rotation = s.rotation;
pixels_to_micron_scale = s.pixel2MicronScale;

%I think the switch needs to go on first here ...
%aaaaaaaahhhhh
pixels = fliplr(pixels);

pixels = (rotation * pixels')';

microns = pixels;
% Convert the pixels coordinates to micron locations.
microns(:,1) = origins(cur_stage_origin,1) - pixels(:,1) * pixels_to_micron_scale(1);
microns(:,2) = origins(cur_stage_origin,2) - pixels(:,2) * pixels_to_micron_scale(2);

end

