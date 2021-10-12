import numpy as np
from scipy.sparse import diags
from scipy.interpolate import interp1d

## This function finds the indices of the array where an event starts and end given a threshold
def flatten(x,y,limit):
    """Return the indices of x where the variation in y is greater than limit."""
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
            #if i < j:
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
    
    ## ind denotes starting point of event
    ## dni denotes ending point of event
    ## up means it was an increase in magnitude (demagnification)
    ## dn means it was an decrease in magnitude (magnification)
    ## dmag is the variation of the magnitude of the event
    return ind_up, ind_dn, dni_up, dni_dn, dmag_up, dmag_dn

## Interpolate to get the x value where 90% of the variation on the y axis respect to the "peak" occurs
def events_90(x, y, limit, int_frac=0.9):
    up_start, dn_start, up_end, dn_end, up_dmag, dn_dmag = flatten(x, y, limit)
    x_starts = []
    x_ends   = []
    y_starts = []
    y_ends   = []
    flag     = []
    ## For the "up" events the ending point is modified
    for i in range(len(up_start)):
        f = interp1d(y[up_start[i]:up_end[i]+1], x[up_start[i]:up_end[i]+1], kind='zero')
        x_starts.append(x[up_start[i]])
        x_ends.append(f(y[up_start[i]] + int_frac * up_dmag[i]))
        y_starts.append(y[up_start[i]])
        y_ends.append(y[up_start[i]] + int_frac * up_dmag[i])
        flag.append("demag")
    ## For the "dn" events the starting point is modified
    for i in range(len(dn_start)):
        f = interp1d(y[dn_start[i]:dn_end[i]+1], x[dn_start[i]:dn_end[i]+1], kind='zero')
        x_starts.append(f(y[dn_end[i]] - int_frac * dn_dmag[i]))
        x_ends.append(x[dn_end[i]])
        y_starts.append(y[dn_end[i]] - int_frac * dn_dmag[i])
        y_ends.append(y[dn_end[i]])
        flag.append("demag")

    return x_starts, x_ends, y_starts, y_ends, flag

## Interpolate to get the x value where 90% of the variation on the y axis respect to the "peak" occurs
def events_90_flux(x, y, limit, int_frac=0.9):
    up_start, dn_start, up_end, dn_end, up_dmag, dn_dmag = flatten(x, y, limit)
    x_starts = []
    x_ends   = []
    y_starts = []
    y_ends   = []
    flag     = []
    
    y = 10**(-0.4*y)
    
    ## For the "up" events the ending point is modified
    for i in range(len(up_start)):
        f = interp1d(y[up_start[i]:up_end[i]+1], x[up_start[i]:up_end[i]+1], kind='zero')
        x_starts.append(x[up_start[i]])
        x_ends.append(f(y[up_start[i]] + int_frac * (y[up_end[i]]-y[up_start[i]])))
        y_starts.append(y[up_start[i]])
        y_ends.append(y[up_start[i]] + int_frac * (y[up_end[i]]-y[up_start[i]]))
        flag.append("demag")
    ## For the "dn" events the starting point is modified
    for i in range(len(dn_start)):
        f = interp1d(y[dn_start[i]:dn_end[i]+1], x[dn_start[i]:dn_end[i]+1], kind='zero')
        x_starts.append(f(y[dn_end[i]] - int_frac * (y[dn_end[i]]-y[dn_start[i]])))
        x_ends.append(x[dn_end[i]])
        y_starts.append(y[dn_end[i]] - int_frac * (y[dn_end[i]]-y[dn_start[i]]))
        y_ends.append(y[dn_end[i]])
        flag.append("demag")
    
    y_starts = -2.5*np.log10(y_starts)
    y_ends = -2.5*np.log10(y_ends)

    return x_starts, x_ends, y_starts, y_ends, flag

def cc_count(curve):
    aux = np.where(curve==1)[0]
    red_ind = []
    if len(aux) > 1:
        red_ind.append(aux[0])
        for i in range(len(aux)-1):
            if aux[i+1] - aux[i] == 1:
                continue
            if aux[i+1] - aux[i] > 1:
                red_ind.append(aux[i+1])
    return len(red_ind)