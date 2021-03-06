#projection interference for 1/4 fits file and PCP. using matrix instead of loop ,t_sample,f_sample
import numpy as np
import config
import torch
import matplotlib.pyplot as plt
import sys


t_sample = config.t_sample
f_sample = config.f_sample
beam = config.beam
t_shape = config.t_shape
f_shape = config.f_shape


def plot_covariance(data):
    covariance = data.mean(axis=0).squeeze()
    l,m,n = covariance.shape
    for i in range(m):
        for j in range(n):
            figure_name = config.path2save + 'covariance_' + str(i+1) + '-' + str(j+1) + '.png'
            plt.plot(covariance[:,i,j])
            plt.xlabel('Channel')
            plt.ylabel('Intensity')
            plt.title('covariance_'+str(i+1)+'-'+str(j+1))
            plt.savefig(figure_name,dpi=400)
            plt.close()

def make_covariance_matrix(data):
    t_step = int(t_shape/t_sample)
    f_step = int(f_shape/f_sample)    
    D=np.zeros((beam,beam,t_shape,f_shape))
    for i in range(beam):
        for j in range(beam):
            a = np.sqrt(data[i]*data[j])
            D[i][j] = a.reshape(t_shape,f_shape)
    D_ms = np.transpose(D,(2,3,0,1))
    D = D.reshape(beam,beam,t_sample,t_step,f_shape).mean(axis=3)
    D = D.reshape(beam,beam,t_sample,f_shape,f_step).mean(axis=-1).squeeze()
    D = np.transpose(D,(2,3,0,1))
    print('filter shape:',D.shape)
    #plot_covariance(D_ms)
    return D_ms,D


def make_matrix(D_ms,correlation):
    t_step = int(t_shape/t_sample)
    f_step = int(f_shape/f_sample)
    D=D_ms
    data_clean = np.zeros((beam,t_shape,f_shape))
    ### batch SVD by CPU
    correlation_cpu = torch.tensor(correlation.reshape(-1,beam,beam))
    u_cpu, s_cpu, vh_cpu = torch.svd(correlation_cpu)
    u_cpu = u_cpu.reshape(t_sample,f_sample,beam,beam)
    s_cpu = s_cpu.reshape(t_sample,f_sample,beam)

    '''
    u_cpu = np.array(u_cpu)
    s_cpu = np.array(s_cpu)
    vh_cpu = np.array(vh_cpu)
    u_cpu = u_cpu.reshape(t_sample,f_sample,beam,beam)
    s_cpu = s_cpu.reshape(t_sample,f_sample,beam)
    vh_cpu = vh_cpu.reshape(t_sample,f_sample,beam,beam)
    s_cpu = baseline_removal.ArPLS_eigen(s_cpu)
    spectrum = np.zeros((t_shape,f_shape,beam))
    for i in range(t_shape):
        for j in range(f_shape):
            u_sample = u_cpu[int(i/t_step)][int(j/f_step)].squeeze()
            s_sample = s_cpu[int(i/t_step)][int(j/f_step)].squeeze()
            vh_sample = vh_cpu[int(i/t_step)][int(j/f_step)].squeeze()
            matrix_clean = np.dot(u_sample,np.dot(np.diag(s_sample),vh_sample))
            spectrum[i][j]=np.diag(matrix_clean)
    spectrum = np.transpose(spectrum,(2,0,1))
    '''
    #np.save(config.path2save+'eigenvalue.npy',s_cpu)
    #sys.exit(0)
    '''
    ### batch SVD by GPU
    cuda0 = torch.device('cuda:0')
    correlation_gpu = torch.tensor(correlation_cpu,device=cuda0)
    u_gpu, s_gpu, vh_gpu = torch.svd(correlation_gpu)
    u_gpu = u_gpu.reshape(t_sample,f_sample,beam,beam)
    u_cpu = u_gpu.cpu()
    u_cpu = u_cpu.reshape(t_sample,f_sample,beam,beam)
    torch.cuda.empty_cache()
    '''
    u = np.array(u_cpu)
    spectrum = np.zeros((t_shape,f_shape,beam))
    
    ### subject the RFI subspace ###
    rfi_components = 1
    for i in range(t_shape):
        for j in range(f_shape): 
            u_sample = u[int(i/t_step)][int(j/f_step)].squeeze()
            u_rfi = u_sample[:,:rfi_components]
            P = np.dot(u_rfi,u_rfi.T)
            c = D[i][j]
            matrix_clean = c - np.dot(P,np.dot(c,P))
            spectrum[i][j] = np.diag(matrix_clean)
    '''
    ### projection on the noise subspace ###
    noise_component_start = 1
    noise_component_stop = 5
    for i in range(t_shape):
        for j in range(f_shape):
            u_sample = u[int(i/t_step)][int(j/f_step)].squeeze()
            u_noise = u_sample[:,noise_component_start:noise_component_stop]
            P = np.dot(u_noise,u_noise.T)
            c = D[i][j]
            matrix_clean = np.dot(P,np.dot(c,P))
            spectrum[i][j] = np.diag(matrix_clean)
    '''
    spectrum = np.transpose(spectrum,(2,0,1))
    for k in range(beam):
        data_clean[k]=spectrum[k]
    data_clean[data_clean<0]=0
    return data_clean          

