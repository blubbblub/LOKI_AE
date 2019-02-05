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


import os, sys
import numpy as num
#import LatLongUTMconversion
import scipy.io as sio
import location
import C_STALTA
import tt_processing
import pylab as plt
from obspy.core import read
from matplotlib.colors import Normalize

class traveltimes:

    def __init__(self, db_path, hdr_filename):
        if not os.path.isdir(db_path):
           print('Error: data or database path do not exist')
           sys.exit()
        self.db_path = db_path
        if not os.path.isfile(db_path+'/'+hdr_filename):
           print('Error: header file does not exist')
           sys.exit()
        self.hdr_filename = hdr_filename
        self.load_header()


    def load_header(self):

        f = open(os.path.join(self.db_path, self.hdr_filename))
        lines = f.readlines()
        f.close()
        self.nx, self.ny, self.nz = [ int(x)   for x in lines[0].split()]
        self.x0, self.y0, self.z0 = [ float(x) for x in lines[1].split()]
        self.dx, self.dy, self.dz = [ float(x) for x in lines[2].split()]
        self.lat0, self.lon0 = [ float(x) for x in lines[3].split()]
        self.x = self.x0+(num.arange(0,self.nx)*self.dx)
        self.y = self.y0+(num.arange(0,self.ny)*self.dy)
        self.z = self.z0+(num.arange(0,self.nz)*self.dz)
        self.nxyz=self.nx*self.ny*self.nz
        db_stalist=[]
        if len(lines[4].split())>1:
            stations_coordinates={}
        else:
            stations_coordinates=None

        for line in lines[4:]:
            toks=line.split()
            db_stalist.append(toks[0])
            if len(toks)>1:
                stations_coordinates[toks[0]]=[eval(toks[1]), eval(toks[2]), eval(toks[3])]
            else:
                print('Missing coordinates for the station : '+ toks[0]+' \n')

        self.db_stations=set(db_stalist)
        self.stations_coordinates=stations_coordinates

    def load_traveltimes(self, phase, precision='double'):
        t={}
        for sta in self.db_stations:
            try:
               fn = os.path.join(self.db_path, 'layer.%(phase)s.%(station)s.time.buf' %{"phase":phase, "station":sta} )
            except:
               print('Error: reading file for station' + sta)
            if (precision=='single'):
                t[sta]= num.fromfile(fn, dtype=num.float32)
            elif (precision=='double'):
                t[sta]= num.fromfile(fn, dtype=num.float64)
            else:
                print('Error: precision must be set to "single" or "double"!!!')
                sys.exit()
        return t

    def ttdb_generator(self, velocity, phase='P'):

        if self.stations_coordinates is not None:
            for sta in self.stations_coordinates:
                print('Starting to calculate traveltimes for the station : ' + sta +' \n')
                xsta,ysta,zsta=self.stations_coordinates[sta][0:3]
                fname='homo.%(phase)s.%(station)s.time.buf' %{"phase":phase, "station":sta}
                fout=open(self.db_path+'/'+fname,'wb')
                tt=num.zeros(self.nxyz)
                for k in range(self.nxyz):
                    ix=k//(self.ny*self.nz)
                    iy=k//self.nz-(ix*self.ny)
                    iz=k-(iy*self.nz)-(ix*self.ny*self.nz)
                    dist=num.sqrt((self.x[ix]-xsta)**2+(self.y[iy]-ysta)**2+(self.z[iz]-zsta)**2)
                    tt[k]=dist/velocity
                tt.tofile(fout)
                fout.close()
                print('Traveltimes computation for the station : ' + sta + 'completed! \n')
        return None


tt=traveltimes('/home/peter/opt/LOKI/aspo_HF2_neg', 'aspo_HF2_neg.hdr')
tt.ttdb_generator(5800.,'P')
tt.ttdb_generator(3200.,'S')

