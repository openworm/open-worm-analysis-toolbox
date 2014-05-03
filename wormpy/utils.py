# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt

def plot(data):
  plt.plot(data)
  plt.show()

def imagesc(data):
  #http://matplotlib.org/api/pyplot_api.html?highlight=imshow#matplotlib.pyplot.imshow
  plt.imshow(data,aspect='auto')
  plt.show()
  

def print_object(obj):

    """ Goal is to eventually mimic Matlab's default display behavior for objects """

    #TODO - have some way of indicating nested function and not doing fancy
    #print for nested objects ...

    MAX_WIDTH = 70

    """
    Example output from Matlab    
    
    morphology: [1x1 seg_worm.features.morphology]
       posture: [1x1 seg_worm.features.posture]
    locomotion: [1x1 seg_worm.features.locomotion]
          path: [1x1 seg_worm.features.path]
          info: [1x1 seg_worm.info]
    """

    dict_local = obj.__dict__

    key_names      = [x for x in dict_local]    
    key_lengths    = [len(x) for x in key_names]
    max_key_length = max(key_lengths)
    key_padding    = [max_key_length - x for x in key_lengths]
    
    max_leadin_length = max_key_length + 2
    max_value_length  = MAX_WIDTH - max_leadin_length
 
 
    lead_strings   = [' '*x + y + ': ' for x,y in zip(key_padding,key_names)]    
    
    #TODO: Alphabatize the results ????
    #Could pass in as a option
    #TODO: It might be better to test for built in types
    #   Class::Bio.Entrez.Parser.DictionaryElement
    #   => show actual dictionary, not what is above
    
    
    value_strings = []
    for key in dict_local:
        value = dict_local[key]
        try: #Not sure how to test for classes :/
            class_name  = value.__class__.__name__
            module_name = inspect.getmodule(value).__name__
            temp_str    = 'Class::' + module_name + '.' + class_name
        except:
            temp_str    = repr(value)
            if len(temp_str) > max_value_length:
                #type_str = str(type(value))
                #type_str = type_str[7:-2]
                try:
                  len_value = len(value)
                except:
                  len_value = 1
                temp_str = str.format('Type::{}, Len: {}',type(value).__name__,len_value)                
  
        value_strings.append(temp_str)    
    
    final_str = ''
    for cur_lead_str, cur_value in zip(lead_strings,value_strings):
        final_str += (cur_lead_str + cur_value + '\n')


    return final_str
