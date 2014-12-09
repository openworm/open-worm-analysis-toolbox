# -*- coding: utf-8 -*-
"""
Plot the eigenworms file.

For more information see
https://github.com/openworm/movement_validation/issues/79

"""

import sys, os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
import movement_validation


def main():
    # Open the eigenworms file    
    features_path = os.path.dirname(movement_validation.features.__file__)
    eigenworm_path = os.path.join(features_path, 
                                  movement_validation.config.EIGENWORM_FILE)
    eigenworm_file = h5py.File(eigenworm_path, 'r')
    
    # Extract the data
    eigenworms = eigenworm_file["eigenWorms"].value

    eigenworm_file.close()

    # Print the shape of eigenworm matrix
    print(np.shape(eigenworms))

    # Plot the eigenworms
    for eigenworm_i in range(np.shape(eigenworms)[1]):
        plt.plot(eigenworms[eigenworm_i])
    plt.show()




if __name__ == '__main__':
    main()
