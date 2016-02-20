# -*- coding: utf-8 -*-
"""
Plot the eigenworms file.

For more information see
https://github.com/openworm/open-worm-analysis-toolbox/issues/79

"""

import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
#import mpld3

# We must add .. to the path so that we can perform the
# import of open-worm-analysis-toolbox while running this as
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..')
import open_worm_analysis_toolbox as mv


def main():
    # Open the eigenworms file
    features_path = os.path.dirname(mv.features.__file__)
    eigenworm_path = os.path.join(features_path, mv.config.EIGENWORM_FILE)
    eigenworm_file = h5py.File(eigenworm_path, 'r')

    # Extract the data
    eigenworms = eigenworm_file["eigenWorms"].value

    eigenworm_file.close()

    # Print the shape of eigenworm matrix
    print(np.shape(eigenworms))

    # Plot the eigenworms
    for eigenworm_i in range(np.shape(eigenworms)[1]):
        plt.plot(eigenworms[:, eigenworm_i])
    # mpld3.show()

    plt.show()

if __name__ == '__main__':
    main()
