#!/bin/env python3
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       Author: Francesco Grigoli
import timeit
import loki_ae
import shutil, glob, numpy, os

#regular expression for file reading
extension='*.npy'   #for instance '*.[she][hn][zne].gz'

#precision of the travel-time database
precision='double'

# Path of the traveltime database
db_path='/media/peter/7739547D1B1A00DA/LOKI_grids/aspo_HF2_neg'

# name of the file containing information about the grid parameters
hdr_filename='aspo_HF2_neg.hdr'

catalog_name = 'HF2_catalog_acf1500_jose_test_LOKI_20'
#catalog_name = 'HF2_catalog_complete_mul'
#catalog_name = 'HF2_catalog_magnitude_2.0_lassie'
#catalog_name = 'Test_dataset'

# Path of of the data folder or event folder (for single event location)
data_path='/media/peter/OS/Catalog_events_ASPO_128_HF2/npy/HF2_catalog_acf1500_jose_test_LOKI_20_1MHz'


ntrial=1
npr=8		           # number of parallel cores
epsilon=0.001		      # to avoid numerical issues
slrat=2.6  			       # short to long time window ration
nshortmax=200			     # maximum length of the short time window
nshortmin=200	      # minimum length of the short time window

normalization = 'ampl'

###########tuning########
#for normalization in ['area','ampl']:
#      for slrat in numpy.arange(2.0,3.8,0.2):
#          for nshortmax in [300,350]:

#              nshortmin=nshortmax	      # minimum length of the short time window


# Path where store the outputs (catalogues, coherence matrices etc.)
             #output_path='/home/peter/working_dir/LOKI/output_%s_%s_%s' % (normalization, nshortmax, slrat)
output_path='/home/peter/Desktop/LOKI/output_%s_%s_%s_%s' % (normalization, slrat, nshortmin, catalog_name)

if os.path.isdir(output_path) == False:
    os.makedirs(output_path) 

loc=loki_ae.loki(data_path, output_path, db_path, hdr_filename)
loc.location_process(extension, precision, nshortmin, nshortmax, slrat, npr, ntrial, normalization)
# for remove in glob.glob('/home/pniemz/working_dir/LOKI/out_test/2015-06-04T08:50:50.*'):
#     shutil.rmtree(remove)
