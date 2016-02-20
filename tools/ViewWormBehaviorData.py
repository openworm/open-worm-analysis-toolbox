# -*- coding: utf-8 -*-
"""
Written in August 2013
Based on a gist by Peter McCluskey:
https://gist.github.com/PeterMcCluskey/6418155

This script takes a local filename of a file that was downloaded from the
C. elegans behavioural database and writes a wormbehavior.html file in
the current directory with links to .png files that plot the datasets
from the 'worm' Group. It does nothing with the 'info' Group, which
contains some info about the type of worm and environment. Note that the
.png filenames start with a '.', so you may need ls -a to see them.

It takes a few minutes to run.

See https://github.com/openworm/OpenWorm/issues/82 for more info.

For an example file, try first downloading this file and placing it in
the same folder as this script:
    'ftp://anonymous@ftp.mrc-lmb.cam.ac.uk/pub/tjucikas/wormdatabase/' +
    'results-12-06-08/Laura%20Grundy/gene_NA/allele_NA/ED3049/on_food/' +
    'XX/30m_wait/L/tracker_7/2011-03-03___16_55_31/' +
    '764%20ED3049%20on%20food%20R_2011_03_03__16_55_31___7___11_features.mat'

"""

import sys
import re
import h5py
import matplotlib.pyplot as plotter


def view_group(group1, h5file, html_file, name=''):
    for fld in group1.keys():
        print("%s/%s %s" % (name, fld, len(group1[fld])))
        name1 = name + '/' + fld
        if isinstance(group1[fld], h5py.Group):
            # Recursive step!
            view_group(group1[fld], h5file, html_file, name=name1)
        else:
            try:
                is_ref = isinstance(group1[fld][0][0], h5py.h5r.Reference)
            except (IndexError, AttributeError):
                is_ref = False
            if is_ref:
                dataset = []
                i = 0
                for ref_set in group1[fld]:
                    i += 1
                    for ref1 in ref_set:
                        # I'm just guessing here about what to do
                        dataset.append(h5file[ref1][0][0])
                        print("deref %2d %.72s" % (i, str(dataset)))
            else:
                dataset = group1[fld]
            print("dataset %s" % dataset[0].__class__)
            print("%.72s" % dataset[:8])
            plotter.close()
            plotter.plot(dataset)
            # plotter.ylabel(?)
            plotter.title(name1)
            fname = re.sub("/", ".", name1) + '.png'
            plotter.savefig(fname)
            html_file.write('<li><a href="%s">%s</a></li>\n' % (fname, name1))


def view_file(filename):
    try:
        h5file = h5py.File(filename, 'r')
    except IOError:
        print("Problem with file %s" % filename)
        raise

    html_file = open('wormbehavior.html', 'w')
    html_file.write("<ol>\n")
    view_group(h5file['worm'], h5file, html_file)
    html_file.write("</ol>\n")

if __name__ == "__main__":
    view_file(sys.argv[1])
    #view_file('764 ED3049 on food R_2011_03_03__16_55_31___7___11_features.mat')


# I didn't use pytables because I got this:
#/worm/locomotion/motion/backward/frames/distance (UnImplemented(29, 1)) ''
#  NOTE: <The UnImplemented object represents a PyTables unimplemented
#         dataset present
