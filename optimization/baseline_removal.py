import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from multiprocessing import Pool
import sys
import pywt

import config



def ArPLS(y, lam=1e5, ratio=0.05, itermax=15):
    '''
    copy from https://irfpy.irf.se/projects/ica/_modules/irfpy/ica/baseline.html
    
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Inputs:
        y:
            input data (i.e. SED curve)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
    '''

    N = len(y)
    #  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    D = D.T
    w = np.ones(N)
    lam = lam * np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        LAM = sparse.diags(lam, 0, shape=(N, N))
        Z = W + LAM * D.dot(D.T)
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        #lam = lam * (1-wt)
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt        
    return z    

## Gaussian slide window method
def gaussian_filter(V, mask, M=64, N=20, sigma_m=40, sigma_n=0.5):
    """
    Applies a gaussian filter (smoothing) to the given array taking into account masked values
    copy from https://github.com/cosmo-ethz/seek/blob/master/seek/utils/filter.py

    :param V: the value array to be smoothed
    :param mask: boolean array defining masked values
    :param M: kernel window size in axis=1
    :param N: kernel window size in axis=0
    :param sigma_m: kernel sigma in axis=1
    :param sigma_n: kernel sigma in axis=0
    
    :returns vs: the filtered array 
    """
    
    def wd(n, m, sigma_n, sigma_m):
        return np.exp(-n**2/(2*sigma_n**2) - m**2/(2*sigma_m**2))
    
    Vp = np.zeros((V.shape[0]+N, V.shape[1]+M))
    Vp[N//2:-N//2,M//2:-M//2] = V[:]

    Wfp = np.zeros((V.shape[0]+N, V.shape[1]+M))
    Wfp[N//2:-N//2,M//2:-M//2] = ~mask[:]
    #plt.imshow(Wfp,cmap='hot', origin='lower', aspect='auto', interpolation='nearest')
    Vh = np.zeros((V.shape[0]+N, V.shape[1]+M))
    Vh2 = np.zeros((V.shape[0]+N, V.shape[1]+M))
    
    n = np.arange(-N/2, N/2+1)
    m = np.arange(-M/2, M/2+1)
    kernel_0 = wd(n, 0, sigma_n=sigma_n, sigma_m=sigma_m).T
    kernel_1 = wd(0, m, sigma_n=sigma_n, sigma_m=sigma_m).T
    
    Vh = _gaussian_filter(Vp, V.shape[0], V.shape[1], Wfp, mask, Vh, Vh2, kernel_0, kernel_1, M, N)
    Vh = Vh[N//2:-N//2,M//2:-M//2]
    Vh[mask] = V[mask]
    return Vh


def _gaussian_filter(Vp, vs0, vs1, Wfp, mask, Vh, Vh2, kernel_0, kernel_1, M, N):
    '''
    copy from https://github.com/cosmo-ethz/seek/blob/master/seek/utils/filter.py
    '''
    n2 = int(N/2)
    m2 = int(M/2)
    for i in range((N//2), vs0+(N//2)):
        for j in range((M//2), vs1+(M//2)):
            if mask[i-n2, j-m2]:
                Vh[i, j] = 0 #V[i-n2, j-m2]
            else:
                val = np.sum((Wfp[i-n2:i+n2+1, j] * Vp[i-n2:i+n2+1, j] * kernel_0))
                Vh[i, j] = val / np.sum(Wfp[i-n2:i+n2+1, j] * kernel_0)
    
    for j2 in range((M//2), vs1+(M//2)):
        for i2 in range((N//2), vs0+(N//2)):
            if mask[i2-n2, j2-m2]:
                Vh2[i2, j2] = 0 #V[i2-n2, j2-m2]
            else:
                val = np.sum((Wfp[i2, j2-m2:j2+m2+1] * Vh[i2, j2-m2:j2+m2+1] * kernel_1))
                Vh2[i2, j2] = val / np.sum(Wfp[i2, j2-m2:j2+m2+1] * kernel_1)
    return Vh2

def smooth(sig, level=2, threshold=3, wavelet='db8'):
    sigma = sig.std()
    dwtmatr = pywt.wavedec(data=sig, wavelet=wavelet, level=level)
    denoised = dwtmatr[:]
    denoised[1:] = [pywt.threshold(i, value=threshold*sigma, mode='soft') for i in dwtmatr[1:]]
    smoothed_sig = pywt.waverec(denoised, wavelet, mode='sp1')[:sig.size]
    return smoothed_sig

def _baseline_ArPLS(data):
    l,m = data.shape
    for i in range(l):
        data[i,:] -= ArPLS(data[i,:]) 
    data[data<0] = 0
    return data

### fitting and removing the baseline by ArPLS (parallel computing)
def baseline_removal_ArPLS(data):
    output=[]
    l,m,n = data.shape
    Processes = int(8)
    data = data.reshape(Processes,-1,n)
    p = Pool(Processes)
    process_list = []
    for i in range(Processes):
        process_list.append(p.apply_async(_baseline_ArPLS, args=(data[i,:,:],)))
    p.close()
    p.join()

    for i in process_list:
        output.append(i.get())
    output = np.array(output)
    output = output.reshape(l,m,n)
    return output

def _baseline_gaussian(data):
    l,m = data.shape
    surface = data
    mask = np.zeros((l,m)).astype('bool')
    for i in range(5):
        surface = gaussian_filter(surface, mask, M=64, N=20, sigma_m=20, sigma_n=0.5)
    data = data - surface
    data[data<0]=0
    return data

### fitting and removing the baseline by GF
def baseline_removal_gaussian(data):
    output=[]
    l,m,n = data.shape
    Processes = int(8)
    data = data.reshape(Processes,-1,n)
    p = Pool(Processes)
    process_list = []
    for i in range(Processes):
        process_list.append(p.apply_async(_baseline_gaussian, args=(data[i,:,:],)))
    p.close()
    p.join()

    for i in process_list:
        output.append(i.get())
    output = np.array(output)
    output = output.reshape(l,m,n)
    return output

def _baseline_wavelet(data):
    l,m = data.shape
    for i in range(l):
        data[i,:] -= smooth(data[i,:])
    data[data<0] = 0
    return data

### fitting and removing the baseline by Wavelet
def baseline_removal_wavelet(data):
    output=[]
    l,m,n = data.shape
    Processes = int(8)
    data = data.reshape(Processes,-1,n)
    p = Pool(Processes)
    process_list = []
    for i in range(Processes):
        process_list.append(p.apply_async(_baseline_wavelet, args=(data[i,:,:],)))
    p.close()
    p.join()
    for i in process_list:
        output.append(i.get())
    output = np.array(output)
    output = output.reshape(l,m,n)
    return output

    