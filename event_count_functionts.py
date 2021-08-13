import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
import pandas as pd
import collections
import sys
from scipy.sparse import diags
import os
from scipy.interpolate import interp1d

### Find cross test ###

def flatten(x,y,limit):

    k = np.ones_like(y)
    dx = np.concatenate([np.diff(x),[x[-2]-x[-1]]])
    Dx = diags([-k,k],[0,1]).todense()
    dy = Dx.dot(y.T)/dx
    dy[0,-1] = dy[0,-2]
    dy = np.asarray(dy)[0]
    
    dy = np.nan_to_num(dy)

    for i in range(len(dy)-1):
                if dy[i+1] == 0.:
                        if dy[i] < 0.:
                                dy[i+1] += -10e-9
                        if dy[i] > 0.:
                                dy[i+1] += 10e-9

    # extremas
    ind = [0]
    for i in range(len(dy)-1):
        if dy[i]*dy[i+1] <= 0:
            ind.append(i+1)
    ind.append(len(dy)-1)
    

    dmag = []
    

    for i in range(len(ind)-1):
        i0 = ind[i]
        i1 = ind[i+1]
        dm = y[i1]-y[i0]
        dmag.append(dm)
        
    dmag_matrix = np.zeros((len(ind),len(ind)))
    
    
    
    ind_up = []
    ind_dn = []

    dni_up = []
    dni_dn = []
    
    dmag_up = []
    dmag_dn = []
    
    
        
    for i in range(len(ind)):
        for j in range(len(ind)):
            if np.abs(y[ind[j]] - y[ind[i]]) > limit and i < j:
                dmag_matrix[i,j] = y[ind[j]] - y[ind[i]]



    aux1 = True
    aux2 = True
    
    mask = np.zeros(dmag_matrix.shape[0])
    i1,j1 = 0,1
    while aux2:
        if np.all(dmag_matrix[:,j1] == mask):
            if j1+1 >= dmag_matrix.shape[1]:
                return ind_up,ind_dn,dni_up,dni_dn,dmag_up,dmag_dn
            j1 += 1
            continue
        if not np.all(dmag_matrix[:,j1] == mask):
            i1 = np.min(np.where(dmag_matrix[:,j1] != 0))
            j2 = j1+1
            i2 = j2-1
            break
        
    while aux1:
        if i1+1 == dmag_matrix.shape[0] or j1+1 == dmag_matrix.shape[1]:
            if dmag_matrix[i1,j1] > 0:
                ind_up.append(ind[i1])
                dni_up.append(ind[j1])
                dmag_up.append(dmag_matrix[i1,j1])
            if dmag_matrix[i1,j1] < 0:
                ind_dn.append(ind[i1])
                dni_dn.append(ind[j1])
                dmag_dn.append(dmag_matrix[i1,j1])
            break
        if dmag_matrix[i2,j2] == 0:
            j1 = np.where(np.max(np.abs([dmag_matrix[i1,j1],dmag_matrix[i1,j2]])) == np.abs(dmag_matrix[i1,:]))[0]
            while True:
                if type(j1) == np.int64 or type(j1) == int:
                    break
                j1 = j1[0]
            if j2+1 == dmag_matrix.shape[1]:
                #stop and save
                if dmag_matrix[i1,j1] > 0:
                    ind_up.append(ind[i1])
                    dni_up.append(ind[j1])
                    dmag_up.append(dmag_matrix[i1,j1])
                    break
                if dmag_matrix[i1,j1] < 0:
                    ind_dn.append(ind[i1])
                    dni_dn.append(ind[j1])
                    dmag_dn.append(dmag_matrix[i1,j1])
                    break
            i2 = j2 - np.abs(j2-j1)
            j2 += 1
            continue
        if dmag_matrix[i1,j1] * dmag_matrix[i2,j2] < 0:
            if dmag_matrix[i1,j1] > 0:
                ind_up.append(ind[i1])
                dni_up.append(ind[j1])
                dmag_up.append(dmag_matrix[i1,j1])
                i1 = i2
                j1 = j2
                j2 = j1 + 1
                i2 = j2 - np.abs(j2-j1)
                continue
            if dmag_matrix[i1,j1] < 0:
                ind_dn.append(ind[i1])
                dni_dn.append(ind[j1])
                dmag_dn.append(dmag_matrix[i1,j1])
                i1 = i2
                j1 = j2
                j2 = j1 + 1
                i2 = j2 - np.abs(j2-j1)
                continue
        if dmag_matrix[i1,j1] * dmag_matrix[i2,j2] > 0:
            j1 = j2
            j2 += 1
            i2 = j2 - 1
            continue
            

    return ind_up,ind_dn,dni_up,dni_dn,dmag_up,dmag_dn

def cross(path_to_curves,band,nlc,limit=1.0):
    #path_to_curves = '/home/favio/HE0230/s95/output/'
    band_aux = np.where(np.array(['u','g','r','i','z','y'])==band)[0][0]
    data = np.loadtxt(path_to_curves+'tablet_'+str(nlc)+'.dat').T
    aux1 = path_to_curves+'tablet_'+str(nlc)+'.dat'
    if not np.any(data):
        return [],[],[],[],[],[]
    t = data[0]
    if isinstance(t,np.float) or os.stat(aux1).st_size == 0:
        return [],[],[],[],[],[]
    mag = data[band_aux+1]
    min_event_index, max_event_index, min_event_xedni, max_event_xedni, dmag_dn, dmag_up = flatten(t,mag,limit)
    index_start = min_event_index+max_event_index
    index_stop  = min_event_xedni+max_event_xedni
    t_10_f = []
    mag_10_f = []
    for i,f in zip(index_start,index_stop):
        h = interp1d(10**(mag[range(i,f+1)]/-2.5),t[range(i,f+1)])
        mag_90 = np.abs(10**(mag[i]/-2.5) - 10**(mag[f]/-2.5))*0.9
        if mag[i] < mag[f]:
            mag_10 = 10**(mag[i]/-2.5) - mag_90
            t_10 = h(mag_10)
            t_10_f.append(t_10)
            mag_10_f.append(-2.5*np.log10(mag_10))
        if mag[i] > mag[f]:
            mag_10 = 10**(mag[f]/-2.5) - mag_90
            t_10 = h(mag_10)
            t_10_f.append(t_10)
            mag_10_f.append(-2.5*np.log10(mag_10))
    return t[index_start],t[index_stop],mag[index_start],mag[index_stop],t_10_f,mag_10_f

filters = ['u','g','r','i','z','y']
path_to_curves = "/home/favio/HE0230/s60/output/"
path_to_events = "/home/favio/HE0230/s60/events/"

for i in filters:
	table = []
	for j in range(10000):
		print(j,i)
		jd_start,jd_stop,m_start,m_stop,jd_10,m_10 = cross(path_to_curves,i,j)
		toste = np.array([jd_start,jd_stop,m_start,m_stop,jd_10,m_10]).T
		dataf = pd.DataFrame(toste,columns=['pix_start','pix_stop','mag_start','mag_stop','pix_10','mag_10'])
		if not os.path.exists(path_to_events):
			os.makedirs(path_to_events)
		dataf.to_csv(path_to_events+i+'_'+str(int(j)).zfill(4)+'.csv')
		





