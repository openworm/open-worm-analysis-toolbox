classdef manager < handle
    %
    %   Class:
    %   seg_worm.stats.manager
    
    properties
       stats
       p_worm
       q_worm
    end
    
    methods
        function obj = manager(exp_hist_man,ctl_hist_man)
           
           %seg_worm.stats.manager.initObject
           obj.initObject(exp_hist_man.hists,ctl_hist_man.hists) 
        end
    end
    
end

