# config the parameters for ArPLS-SF, the sample time for FAST pulsar backend is about 49us and the frequency resolution is 122.07KHz
debug = False
t_sample = int(32)     #number of time intervals to construct the filter, about 0.4s
f_sample = int(4096)   #number of freq channels
subint= int(256)         #subint number, depends on the PSRFITS data
beam = 19                #beam number to process
path2save = 'D:\\ArPLS-SF-results\\'
pol_num = 0               #0,1,2,3,4 which pol to process
t_shape = int(256)     # parameters to construct the filter, about 50ms
f_shape = int(4096)
factor = 1.5             #setting threshold 1,2,3 sigma
