
## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import scipy.signal as sp
import scipy.optimize as opt
from scipy.fft import fft, ifft, rfft, irfft, fftfreq
import cPickle as pickle
import h5py
import time
import datetime

path = r"F:\data\20230930\3um_SiO2\1\discharge\2"
path_save = r"F:\data\20230930\3um_SiO2\1\discharge\2\summary"
isExist = os.path.exists(path_save)
print(isExist)
if isExist ==0:
    os.makedirs(path_save)
ts = 1.

fdrive = 35. #31.
make_plot = True

data_columns = [0, 0] # column to calculate the correlation against
drive_column = 8 
def gain_fac( val ):
    ### Return the gain factor corresponding to a given voltage divider
    ### setting.  These numbers are from the calibration of the voltage
    volt_div_vals = {0.:  1.,
                     1.:  1.,
                     20.0: 100./5.07,
                     40.0: 100./2.67,
                     80.0: 100./1.38,
                     200.0: 100./0.464}
    if val in volt_div_vals:
        return volt_div_vals[val]
    else:
        print ("Warning, could not find volt_div value")
        return 1.
   
def get_data(fname):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    _, fext = os.path.splitext( fname )
    if( fext == ".h5"):
        try:
            f = h5py.File(fname,'r')
            dset = f['beads/data/pos_data']
            dat = np.transpose(dset)
            #max_volt = dset.attrs['max_volt']
            #nbit = dset.attrs['nbit']
            max_volt = 10.
            nbit = 32767.
            dat = 1.0*dat*max_volt/nbit
            attribs = dset.attrs

            ## correct the drive amplitude for the voltage divider. 
            ## this assumes the drive is the last column in the dset
            vd = 1. #vd = attribs['volt_div'] if 'volt_div' in attribs else 1.0
            curr_gain = gain_fac(vd)
            dat[:,-1] *= curr_gain

            ## now double check that the rescaled drive amp seems reasonable
            ## and warn the user if not
            offset_frac = 0.#offset_frac = np.abs(np.sqrt(2)*np.std( dat[:,-1] )/(200.0 * attribs['drive_amplitude'] )-1.0)
            if( curr_gain != 1.0 and offset_frac > 0.1):
                print ("Warning, voltage_div setting doesn't appear to match the expected gain for ", fname)

        except (KeyError, IOError):
            print ("Warning, got no keys for: ", fname)
            dat = []
            attribs = {}
            f = []
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}
        f = []

    return dat, attribs, f

def getdata(fname):
    print ("Processing ", fname)
    dat, attribs, cf = get_data(os.path.join(path, fname))

    if( len(attribs) > 0 ):
        fsamp = attribs["Fsamp"]
    print ("Getting data from: ", fname) 
    dat, attribs, cf = get_data(os.path.join(path, fname))
    fsamp = attribs["Fsamp"]
    xdat = dat[:,data_columns[1]]# data of the displacement
    xdat = xdat - np.mean(xdat)
    fourier_x=rfft(xdat)#Do the Fourier transform of the x
    f = fftfreq(len(xdat),1/fsamp)
    freq_x = np.abs(f[0:int(len(xdat)/2)+1])
        
    data_drive=dat[:,drive_column] - np.mean(dat[:,drive_column]) # Electric field without offset
    fourier_drive=rfft(data_drive) # Do the Fourier transform of the drive signal
    f = fftfreq(len(data_drive),1/fsamp)
    freq_drive = np.abs(f[0:int(len(data_drive)/2)+1])#frequency domain axis
    E_freq=freq_drive[np.where(np.abs(fourier_drive)==max(np.abs(fourier_drive)))[0]]

    #plt.plot(data_drive) ## check if the Fourier transform is done as assumed
    #plt.show()
    #plt.plot(freq_drive,np.abs(fourier_drive)) ## check if the Fourier transform is done as assumed
    #plt.show()
    print("Electric field's frequency is:", E_freq[0])
    Conv=fourier_drive*np.conjugate(fourier_x)/ max(np.abs(fourier_drive))
    Corr=irfft(Conv)
    print(len(Corr))
    n=np.where(np.abs(freq_x-E_freq)<2)
    print(n,np.where(np.abs(fourier_drive)==max(np.abs(fourier_drive)))[0][0])
    maxv = max(np.real(Conv[n]),key=abs)
    print("Displacement peak is:",maxv)
    print("drive peak is:", max(np.abs(fourier_drive)))
    Phase=np.angle(fourier_x[np.where(np.abs(fourier_drive)==max(np.abs(fourier_drive)))[0][0]])-np.angle(fourier_drive[np.where(np.abs(fourier_drive)==max(np.abs(fourier_drive)))[0]])
    print("the phase is:", Phase[0])
    cf.close()
    return Phase[0], maxv

corr_data=[]
time_data=[]
pres=[]
if make_plot:
    fig0 = plt.figure()
    # plt.hold(False)

last_file = ""
filelist = glob.glob(os.path.join(path, "*.h5"))
filelist = sorted(filelist, key=os.path.getmtime)
n=0

for cfile in filelist:
    n=n+1
    print(n,'s')
    m_time = os.path.getmtime(cfile)
    dt_m = datetime.datetime.fromtimestamp(m_time)
    if n==1:
        t0=dt_m 
    delta=dt_m-t0
    time_data.append(delta.total_seconds())
    corr = getdata(cfile)
    corr_data.append(corr)
hf = h5py.File(os.path.join(path_save, "discharge.h5"), "w")
hf.create_dataset("discharge", data=corr_data)
hf.close()
if make_plot:
    plt.clf()
    plt.plot(time_data,np.array(corr_data))
    plt.grid()
    plt.text(20,15,"Var=%f"%(np.std(np.array(corr_data)[:,1])/np.mean(np.array(corr_data)[:,1])))
    plt.xlabel('t(s)', {'size': 20})
    plt.ylabel('Correlation value', {'size': 20})
    plt.show()
  
    filtered=sp.savgol_filter(np.array(corr_data)[:,1]/30,2,1)

    plt.clf()
    plt.plot(time_data,-filtered)
    plt.grid()
    plt.xlabel('t(s)', {'size': 20})
    plt.ylabel('Charge[e]', {'size': 20})
    plt.show()

    time_data_new = [x/ 3600 for x in time_data]
    plt.clf()
    plt.plot(time_data_new,-filtered)
    plt.grid()
    plt.xlabel('t(h)', {'size': 20})
    plt.ylabel('Charge[e]', {'size': 20})
    plt.show()