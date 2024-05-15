
## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time
## The difference between charge_new and charge_new_1 is if the convolution is calculated
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

path = r"F:\data\20240515\3um_SiO2\1\discharge\test"
ts = 1.

fdrive = 85. 
make_plot = True

data_columns = [0, 0] # column to calculate the correlation against
drive_column = 3 
def get_data(fname):
    _, fext = os.path.splitext( fname )
    if( fext == ".h5"):
        try:
            f = h5py.File(fname,'r')
            dset = f['beads/data/pos_data']
            dat = np.transpose(dset)
            max_volt = 10.
            nbit = 32767.
            dat = 1.0*dat*max_volt/nbit
            attribs = dset.attrs

            ## correct the drive amplitude for the voltage divider. 
            ## this assumes the drive is the last column in the dset
            curr_gain = 1
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
    print("Electric field's frequency is:", E_freq[0])
    Conv=fourier_drive*np.conjugate(fourier_x) / max(np.abs(fourier_drive))
    n=np.where(np.abs(freq_x-E_freq)<2)
    print(n,np.where(np.abs(fourier_drive)==max(np.abs(fourier_drive)))[0][0])
    maxv = max(np.real(Conv[n]),key=abs)
    print("Displacement peak is:",maxv)
    print("drive peak is:", max(np.abs(fourier_drive)))
    Phase=np.angle(fourier_x[np.where(np.abs(fourier_drive)==max(np.abs(fourier_drive)))[0][0]])-np.angle(fourier_drive[np.where(np.abs(fourier_drive)==max(np.abs(fourier_drive)))[0]])
    print("the phase is:", Phase[0])
    cf.close()
    return Phase[0], maxv

def get_most_recent_file(p):

    ## only consider single frequency files
    filelist = glob.glob(os.path.join(p,"*.h5")) 
    mtime = 0
    mrf = ""
    for fin in filelist:
        if( fin[-3:] != ".h5" ):
            continue
        f = os.path.join(path, fin) 
        if os.path.getmtime(f)>mtime:
            mrf = f
            mtime = os.path.getmtime(f)

    fnum = re.findall('\d+.h5', mrf)[0][:-3]
    return mrf


corr_data=[]
if make_plot:
    fig0 = plt.figure()
    # plt.hold(False)

last_file = ""
while( True ):
    ## get the most recent file in the directory and calculate the correlation
    cfile = get_most_recent_file( path )   
    ## wait a sufficient amount of time to ensure the file is closed
    print (cfile)
    time.sleep(ts)
    if( cfile == last_file ): 
        continue
    else:
        last_file = cfile

    ## this ensures that the file is closed before we try to read it
    time.sleep( 1.2 )
    corr = getdata( cfile )
    corr_data.append(corr )
    np.savetxt( os.path.join(path, "current_corr.txt"), [corr,] )
    if make_plot:
        plt.clf()
        plt.plot(np.array(corr_data),label=["Phase","Charge"])
        plt.grid()
        plt.legend()
        plt.draw()
        plt.pause(0.01)