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


import os, sys, glob
import numpy as num
import numpy as np
#import LatLongUTMconversion
import scipy.io as sio
import location
import location_t0
import C_STALTA
import tt_processing
import datetime
import matplotlib.pyplot as plt
from obspy.core import read
from matplotlib.colors import Normalize
# from pympler.tracker import SummaryTracker  # PN
from pyrocko import trace, io, marker, util #PN

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

    def load_traveltimes(self, phase, precision='single'):
        t={}
        for sta in self.db_stations:
            try:
               fn = os.path.join(self.db_path, 'homo.%(phase)s.%(station)s.time.buf' %{"phase":phase, "station":sta} )
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
                fout=open(fname,'wb')
                tt=num.zeros(self.nxyz)
                print(self.nxyz,self.nx,self.ny,self.nz)
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

class waveforms:

    def __init__(self, event_path, extension='*'):
        if not os.path.isdir(event_path):
           print('Error: data path do not exist')
        self.data_path = event_path
        self.extension = extension
        try:
            self.load_waveforms(event_path)
        except:
            print('Error: data not read for the event: ', event_path)

    def load_waveforms(self, event_path):
        files=glob.glob(os.path.join(event_path,self.extension))[0]
        traces=num.load(files)
        stream={}
        for tr in traces:
            stream[tr['station'].item()]=tr['data']
        self.station_list(stream)
        self.stream=stream
        self.deltat=traces[0]['deltat'].item()
        self.evid=traces[0]['starttime'].item()

    def station_list(self, stream):
        data_stalist=[]
        self.ns=0
        for sta in stream.keys():
            self.ns=max(num.size(stream[sta]),self.ns)
            if sta not in data_stalist:
                data_stalist.append(sta)
        self.data_stations=set(data_stalist)

    def charfunc(self, ztr):
        obs_dataV=ztr**2
        for j in range(self.nstation):
            obs_dataV[j,:]=obs_dataV[j,:]/num.max(obs_dataV[j,:])
        return obs_dataV

    def process_data(self, db_stations, epsilon=0.001):
        self.stations=self.data_stations & db_stations
        self.nstation=len(self.stations)
        self.ztr=num.zeros([self.nstation,self.ns,])
        ztr=num.zeros([self.nstation,self.ns,])
        for i,sta in zip(range(self.nstation),self.stations):
            trace_z=self.stream[sta]
            nsz=num.size(trace_z)
            ztr[i,0:nsz]=trace_z
            ztr[i,1:]=(ztr[i,1:]-ztr[i,0:-1])/self.deltat
            ztr[i,0]=0.
            ztr[i,:]=ztr[i,:]/num.max(num.abs(ztr[i,:]))
        obs_dataV=self.charfunc(ztr)
        self.obs_dataV=obs_dataV

    def recstalta(self, nshort, nlong, norm=0):
        tshort=nshort*self.deltat
        tlong=nlong*self.deltat
        ks=1./nshort
        kl=1./nlong
        obs_data=C_STALTA.recursive_stalta(tshort, tlong, self.deltat, self.obs_dataV, kl, ks, norm)
        return obs_data

class loki:

    def __init__(self, data_path, output_path, db_path, hdr_filename):
        self.data_path=data_path
        self.output_path=output_path
        self.db_path=db_path
        self.hdr_filename=hdr_filename
        self.data_tree, self.events=self.data_struct(self.data_path, self.output_path)

    def data_struct(self, data_path, output_path):
        events=[]
        data_tree=[]
        for root,dirs,files in os.walk(data_path):
           if not dirs:
              data_tree.append(root)
              events.append(root.split('/')[-1])
        for event in events:
           if not os.path.isdir(output_path):
              os.mkdir(output_path)
        return data_tree, events

    def time_extractor(self, tp, ts, data_stations, db_stations):
        stations=data_stations & db_stations
        nsta=len(stations)
        nxyz= num.size(tp[list(stations)[0]])
        tp_mod=num.zeros([nxyz,nsta])
        ts_mod=num.zeros([nxyz,nsta])
        for i,sta in enumerate(stations): #use enumerate instead
            tp_mod[:,i]=tp[sta]
            ts_mod[:,i]=ts[sta]
        return tp_mod, ts_mod

    def catalogue_creation(self, event, event_time, ntrial):
        '''
        PN mod: original in catalogue_creation_loki
        '''
        ev_file=self.output_path+'/'+event+'/'+event+'.loc'
        data=num.loadtxt(ev_file)
        if (ntrial>1):
           xb= data[0,1]
           yb= data[0,2]
           zb= data[0,3]
           cb= data[0,4]
           cmax= data[0,4]
           merr=num.vstack((data[:,1],data[:,2],data[:,3]))
           err=num.cov(merr)
           errmax= num.sqrt(num.max(num.linalg.eigvals(err)))
        else:
           xb=data[1]; yb=data[2]; zb= data[3]; errmax='NA'; cb=data[4]; cmax=data[4];
        f=open(self.output_path+'/'+'catalogue','a')
        f.write(event_time+'    '+str(xb)+'   '+str(yb)+'   '+str(zb)+'   '+str(errmax)+'   '+str(cb)+'   '+str(cmax)+'\n')
        f.close()

    def catalogue_creation_loki(self, event, event_time, ntrial):
        ev_file=self.output_path+'/'+event+'/'+event+'.loc'
        data=num.loadtxt(ev_file)
        if (ntrial>1):
           w=num.sum(data[:,4])
           xb= ((num.dot(data[:,1],data[:,4])/w)*1000)
           yb= ((num.dot(data[:,2],data[:,4])/w)*1000)
           zb= num.dot(data[:,3],data[:,4])/w
           cb= num.mean(data[:,4])
           cmax=num.max(data[:,4])
           merr=num.vstack((data[:,1],data[:,2],data[:,3]))
           err=num.cov(merr)
           errmax= num.sqrt(num.max(num.linalg.eigvals(err)))
        else:
           xb=data[1]; yb=data[2]; zb= data[3]; errmax='NA'; cb=data[4]; cmax=data[4];
        f=open(self.output_path+'/'+'catalogue','a')
        f.write(event_time+'    '+str(xb)+'   '+str(yb)+'   '+str(zb)+'   '+str(errmax)+'   '+str(cb)+'   '+str(cmax)+'\n')
        f.close()

    def location_process(self, extension='*', precision='single', *input):
        terr = 0.00005
        ntrial_parameter = 1
        nshortmin=input[0]; nshortmax=input[1]; slrat=input[2]
        npr=input[3]
        n_trial=input[4]
        normalization=input[5]
        traveldb=traveltimes(self.db_path, self.hdr_filename)
        tp=traveldb.load_traveltimes('P', precision)
        ts=traveldb.load_traveltimes('S', precision)

        traces = [] #PN

        for event_path in self.data_tree:
            #tracker = SummaryTracker()   #PN
            print(event_path, extension)
            loc=waveforms(event_path, extension)
            event=loc.evid
            print('accessing to the event folder: ', event_path, event)
            if os.path.isdir(self.output_path+'/'+event):
               continue
            else:
               os.mkdir(self.output_path+'/'+event)
            loc.process_data(traveldb.db_stations, epsilon=0.001)
            #PN
            # tp_mod0/tsmod0... min arrival times
            #   --> differ for diff runs with same parameters and same event

            tp_mod0, ts_mod0=self.time_extractor(tp, ts, loc.data_stations, traveldb.db_stations)

            for ntrial in range(n_trial):

                tp_mod0_var = tp_mod0
                ts_mod0_var = ts_mod0
                
                if ntrial == 0 or ntrial == 1:
                    pass
                else:
                    for i_sta in range(loc.nstation):
                        tp_mod0_var[:,i_sta] += np.random.normal(0, terr)
                        ts_mod0_var[:,i_sta] += np.random.normal(0, terr)


                # also tp_mod, minimal arrival index, differs             
                tp_mod, ts_mod=tt_processing.tt_f2i(loc.deltat,tp_mod0_var,ts_mod0_var, npr)
                #print(np.shape(tp_mod),np.shape(ts_mod))
    
                for i in range(ntrial_parameter):
                    dn=(nshortmax-nshortmin)/(ntrial_parameter)
                    nshort=nshortmin+i*dn
                    nlong=nshort*slrat
                    obs_dataP=loc.recstalta(nshort, nlong)
                    obs_dataS=loc.recstalta(slrat*nshort, slrat*nlong)
                    #print(np.shape(obs_dataP), np.shape(obs_dataS))
                    
                    print(vars(loc))
                    pstalta=num.zeros([loc.nstation,loc.ns])
                    sstalta=num.zeros([loc.nstation,loc.ns])
                    for i,sta_name in enumerate(loc.stations):
                        print(sta_name)
                        if normalization=='area':
                           normfactorP=num.trapz(obs_dataP[i,:], dx=loc.deltat)
                           normfactorS=num.trapz(obs_dataS[i,:], dx=loc.deltat)
                        else:
                           normfactorP=num.max(obs_dataP[i,:])
                           normfactorS=num.max(obs_dataS[i,:])
                        # pstalta/sstalta as long as given raw data
                        pstalta[i,:]=(obs_dataP[i,:]/normfactorP)
                        sstalta[i,:]=(obs_dataS[i,:]/normfactorS)
                        
                        tr = trace.Trace(deltat = 1e-6,
                                         station = sta_name,
                                         network = 'ASPO',
                                         location = 'TASN',
                                         channel = 'SHZ',
                                         ydata= pstalta[i,:],
                                         tmin = util.stt(loc.evid, format='%Y-%m-%dT%H:%M:%S.OPTFRAC'))
                        
                        io.save(tr,'/media/peter/OS/LOKI/pstalta/pstalta_%s.yaff' % sta_name, format = 'yaff')

                        print (n_trial, 'max pstalta', np.argmax(pstalta[i,:]))
                        print (n_trial, 'pstalta', pstalta[i,:])
                        print (n_trial, 'max tp_mod0_var', np.argmax(tp_mod0_var[:,i]))
                        print (n_trial, 'tp_mod0_var', tp_mod0_var[:,i])

                    
                        # plt.plot(pstalta[i,:])
                    #plt.show()
    
                    #corrmatrix=location.stacking(tp_mod, tp_mod, pstalta, pstalta, npr)
                    #raw_input('KILL ME!!!')
                    iloctime,corrmatrix=location_t0.stacking(tp_mod, tp_mod, pstalta, pstalta, npr)
                    
                    np.save('loc_test/loc_times_%s_%s.npy' % (loc.evid,ntrial), np.array(iloctime))
                    np.save('loc_test/loc_times_%s_%s.npy' % (loc.evid,ntrial), np.array(tp_mod0[iloctime[0],:]))
                    np.save('loc_test/loc_stations_%s_%s.npy' % (loc.evid,ntrial), np.array(loc.stations))

                    torig=(iloctime[1]*loc.deltat-num.min(tp_mod0[iloctime[0],:]))
                    event_time=str(datetime.datetime.strptime(loc.evid,"%Y-%m-%dT%H:%M:%S.%f")+datetime.timedelta(microseconds=torig*1e6))
                    print(event_time)

                    cmax=num.max(corrmatrix)
                    #icmax=num.argmax(corrmatrix)
                    corrmatrix=num.reshape(corrmatrix,(traveldb.nx,traveldb.ny,traveldb.nz))
                    (ixloc,iyloc,izloc)=num.unravel_index(num.argmax(corrmatrix),(traveldb.nx,traveldb.ny,traveldb.nz))
                    #tloc=(loc.tstart+kloc*loc.deltat)-num.min(tp_mod0[icmax])
                    #xloc=ixloc*traveldb.dx; yloc=iyloc*traveldb.dy; zloc=izloc*traveldb.dz
                    xloc=traveldb.x[ixloc]; yloc=traveldb.y[iyloc]; zloc=traveldb.z[izloc]
                    out_file = open(self.output_path+'/'+event+'/'+event+'.loc','a')
                    out_file.write(str(ntrial)+' '+str(xloc)+' '+str(yloc)+' '+str(zloc)+' '+str(cmax)+' '+str(nshort)+' '+str(nlong)+'\n')
                    out_file.close()
                    num.save(self.output_path+'/'+event+'/'+'corrmatrix_trial_'+str(ntrial),corrmatrix)
                    #self.coherence_plot(self.output_path+'/'+event, corrmatrix, traveldb.x, traveldb.y, traveldb.z, ntrial)

            self.catalogue_creation(event, event_time, ntrial)
                # tracker.print_diff()
        print('Ho finito!!!')


    def coherence_plot(self, event_path, corrmatrix, xax, yax, zax, ntrial, normalization=False):
        nx,ny,nz=num.shape(corrmatrix)
        CXY=num.zeros([ny, nx])
        for i in range(ny):
            for j in range(nx):
                CXY[i,j]=num.max(corrmatrix[j,i,:])

        CXZ=num.zeros([nz, nx])
        for i in range(nz):
            for j in range(nx):
			             CXZ[i,j]=num.max(corrmatrix[j,:,i])

        CYZ=num.zeros([nz, ny])
        for i in range(nz):
            for j in range(ny):
                CYZ[i,j]=num.max(corrmatrix[:,j,i])

        if normalization:
           nrm=Normalize(vmin=0., vmax=1.)
        else:
           nrm=None


        fig = plt.figure()
        fig.suptitle('Coherence matrix X-Y', fontsize=14, fontweight='bold')
        ax = fig.gca()
        cmap = plt.cm.get_cmap('jet', 100)
        cs = plt.contourf(xax, yax, CXY, 20, cmap=cmap, interpolation='bilinear', norm=nrm)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        cbar = plt.colorbar(cs)
        plt.axes().set_aspect('equal')
        plt.savefig(event_path+'/'+'Coherence_matrix_xy'+str(ntrial)+'.eps')

        # fig = plt.figure()
        # fig.suptitle('Histogram X-Y', fontsize=14, fontweight='bold')
        # ax = fig.gca()
        # cmap = plt.cm.get_cmap('jet', 100)
        # cs = plt.hist(CXY, bins=256, fc='k', ec='k')
        # plt.axes().set_aspect('equal')
        # plt.savefig(event_path+'/'+'Histogram_xy'+str(ntrial)+'.eps')

        fig = plt.figure()
        fig.suptitle('Coherence matrix X-Z', fontsize=14, fontweight='bold')
        ax = fig.gca()
        cmap = plt.cm.get_cmap('jet', 100)
        cs = plt.contourf(xax, zax, CXZ, 20, cmap=cmap, interpolation='bilinear', norm=nrm)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Z (km)')
        cbar = plt.colorbar(cs)
        ax.invert_yaxis()
        plt.axes().set_aspect('equal')
        plt.savefig(event_path+'/'+'Coherence_matrix_xz'+str(ntrial)+'.eps')


        fig = plt.figure()
        fig.suptitle('Coherence matrix Y-Z', fontsize=14, fontweight='bold')
        ax = fig.gca()
        cmap = plt.cm.get_cmap('jet', 100)
        cs = plt.contourf(yax, zax, CYZ, 20, cmap=cmap, interpolation='bilinear', norm=nrm)
        ax.set_xlabel('Y(km)')
        ax.set_ylabel('Z (km)')
        ax.invert_yaxis()
        cbar = plt.colorbar(cs)
        plt.axes().set_aspect('equal')
        plt.savefig(event_path+'/'+'Coherence_matrix_yz'+str(ntrial)+'.eps')
        plt.close("all")
