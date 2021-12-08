# config the parameters for ArPLS-SF
# the sample time for FAST pulsar backend is about 49us 
# the frequency resolution is 122.07KHz

debug = False

nchan = int(4096)       #freqency channels' number
subint= int(256)        #subint number
nsblk = int(1024)       #the number of samples per subint
beam = 19               #beam number to process, for FAST it's 19

t_sample = int(128)     #number of time intervals to construct the filter, about 0.4s
f_sample = int(4096)    #number of freq channels to construct the filter
pol_num = 0             #0,1,2,3 which pol to process
t_shape = int(1024) # number of time intervals to use the filter, about 50ms
f_shape = int(4096)     # number of freq channels to use the filter
factor = 1.5            #setting threshold
#path2save = 'D:\\pynote\\ArPLS-SF-results\\J1855+0455\\'
path2save = 'D:\\pynote\\ArPLS-SF-results\\J0528+2200\\' 

