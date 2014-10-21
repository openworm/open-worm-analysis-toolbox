# -*- coding: utf-8 -*-
"""
This is the Python port of 
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/specs.m
and its subclass,
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/movement_specs.m



classdef movement_specs < seg_worm.stats.specs
    %
    %   Class:
    %   seg_worm.stats.movement_specs
    %
    %   This class specifies how to treat each movement related feature for
    %   histogram processing.
    %
    %
    %   Access via static method:
    %   seg_worm.stats.movement_specs.getSpecs()
    %
    %   See Also:
    %   seg_worm.stats.hist.createHistograms
    %
    %   TODO:
    %   - might need to incorporate seg_worm.w.stats.wormStatsInfo
    %   - remove is_time_series entry ...
    
    properties
        index
        %feature_category
        is_time_series %TODO: This can be removed ...
        %resolution
        %is_zero_bin %This might not be important
        %is_signed   %I think this dictates having 4 or 16 events ...
        %        name
        %        short_name
        %        units
    end
    
    methods (Static)
        function objs = getSpecs()
            %seg_worm.stats.movement_specs.getSpecs();
            
            csv_path = fullfile(sl.stack.getMyBasePath,'docs','movement_features.csv');
            
            
            
            %TODO: These properties would be better if paired then split ...
            %i.e. group names and type, then split
            %
            %    info = {'feature_field' 1
            %            'index' 2
            %            'feature_category' 1, etc
            %
            %These are the property names that we will assign each column to
            prop_names = {'feature_field' 'index' 'feature_category' 'is_time_series' ...
                'resolution' 'is_zero_bin' 'is_signed' 'name' 'short_name' 'units'};
            
            %1 - strings
            %2 - numeric
            %3 - logical
            prop_types = [1 2 1 3 2 3 3 1 1 1];
            
            fh = @seg_worm.stats.movement_specs;
            
            objs = seg_worm.stats.specs.getObjectsHelper(csv_path,fh,prop_names,prop_types);
        end
    end
    methods
        %TODO: Make abstract in specs ...
        function data = getData(obj,feature_obj)
            
            data = sl.struct.getSubField(feature_obj,obj.feature_field);
            
            %data  = eval(['feature_obj.' obj.feature_field]);
            
            %NOTE: We can't filter data here because the data is filtered according
            %to the value of the data, not according to the velocity of the midbody
            
            if ~isnan(obj.index)
                %This is basically for eigenprojections
                %JAH: I really don't like the orientation: [Dim x n_frames]
                data = data(obj.index,:);
            end
            
        end
    end
    
end
"""


class MovementSpecs(Specs):
    """
    %
    %   Class:
    %   seg_worm.stats.movement_specs
    %
    %   This class specifies how to treat each movement related feature for
    %   histogram processing.
    %
    %
    %   Access via static method:
    %   seg_worm.stats.movement_specs.getSpecs()
    %
    %   See Also:
    %   seg_worm.stats.hist.createHistograms
    %
    %   TODO:
    %   - might need to incorporate seg_worm.w.stats.wormStatsInfo
    %   - remove is_time_series entry ...
    """

    def __init__(self):
        index = None
        is_time_series # TODO: This can be removed
        #%feature_category
        #%resolution
        #%is_zero_bin %This might not be important
        #%is_signed   %I think this dictates having 4 or 16 events ...
        #%        name
        #%        short_name
        #%        units

    # TODO: make this static
    def getSpecs(self):
        """
        Formerly objs = getSpecs()
        %seg_worm.stats.movement_specs.getSpecs();

        """
        
        csv_path = fullfile(sl.stack.getMyBasePath,'docs','movement_features.csv');
        
        
        """        
        %TODO: These properties would be better if paired then split ...
        %i.e. group names and type, then split
        %
        %    info = {'feature_field' 1
        %            'index' 2
        %            'feature_category' 1, etc
        %
        %These are the property names that we will assign each column to
        """
        prop_names = {'feature_field' 'index' 'feature_category' 'is_time_series' ...
            'resolution' 'is_zero_bin' 'is_signed' 'name' 'short_name' 'units'};
        """
        %1 - strings
        %2 - numeric
        %3 - logical
        """
        prop_types = [1 2 1 3 2 3 3 1 1 1];
        
        fh = @seg_worm.stats.movement_specs;
        
        objs = seg_worm.stats.specs.getObjectsHelper(csv_path,fh,prop_names,prop_types);
    


"""
classdef specs < handle
    %
    %   Class:
    %   seg_worm.stats.specs
    
    properties
       feature_field
       feature_category
       resolution
       is_zero_bin
       is_signed
       name
       short_name
       units
    end
    
    methods (Static,Hidden)
        function objs = getObjectsHelper(csv_path,class_function_handle,prop_names,prop_types)
           %
           %    The inherited objects can give relatively simple
           %    instructions on how their properties should be interpreted
           %    from their csv specification file.
           %
           %
           %    seg_worm.stats.specs.getObjectsHelper
           %
           %    INPUTS
           %    ===========================================================
           %
           %    TODO: Cleanup and finish documentation
           %
            %It would be nice to do the reading and object construction in 
           %here but Matlab is awkward for dynamic object creation 
           
           %TODO: It would be nice to improve this function to do the
           %casting inside this function ...
           output = sl.io.readDelimitedFile(csv_path,',',...
               'remove_empty_lines',true,'remove_lines_with_no_content',true);
           
           %Subtract header row ...
           n_objs = size(output,1) - 1;
           objs(n_objs) = class_function_handle();
           
           n_fields = length(prop_names);
           for iField = 1:n_fields
              %NOTE: we skip a header row :/
              cur_field_values = output(2:end,iField);
              cur_field_name   = prop_names{iField};
              cur_field_type   = prop_types(iField);
              switch cur_field_type
                  case 1
                      %strings, do nothing ...
                  case 2
                      cur_field_values = num2cell(str2double(cur_field_values));
                  case 3
                      cur_field_values = num2cell(cellfun(@(x) x == '1',cur_field_values));
              end
              [objs.(cur_field_name)] = deal(cur_field_values{:});
                      
           end
           
        end
    end
    methods
        function value = getLongField(obj)
           value = obj.feature_field;
        end
    end
    
end
"""