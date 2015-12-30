function showParsedFrame

% file_path = 'C:\Backup\Google Drive\OpenWorm\OpenWorm Public\movement_analysis\example_data\example_contour_and_skeleton_info.mat';
% fp2 = 'C:\Backup\Google Drive\OpenWorm\OpenWorm Public\movement_analysis\example_data\example_video_norm_worm.mat';

FRAME_USE = 980;

video_path = 'C:\Users\RNEL\Google Drive\OpenWorm\OpenWorm Public\movement_analysis\example_data\example_video_data\mec-4 (u253) off food x_2010_04_21__17_19_20__1.avi';

vr = sl.video.avi.reader(video_path);

temp = vr.getFrame(FRAME_USE);

figure(1)
image(temp)
set(gca,'xlim',[250 400],'ylim',[130 280],'YDir','normal')

figure(2)
awesome_contours_oh_yeah_CNS_2015(FRAME_USE);
c = get(2,'children');
set(c(2),'xlim',[13400 14000],'ylim',[18200 18800]);