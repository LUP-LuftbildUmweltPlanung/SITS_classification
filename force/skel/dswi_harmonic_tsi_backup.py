import numpy as np
import time
from datetime import datetime, timedelta, date
from scipy.optimize import curve_fit
# some global config variables
start = date.fromisoformat('1970-01-01')

start_date_pred = '2018-01-01'  # days since epoch (1970-01-01)
end_date_pred = '2023-03-12'  # days since epoch (1970-01-01)
step = 16  # days
start_date_pred_iso = (date.fromisoformat(start_date_pred) - start).days
end_date_pred_iso = (date.fromisoformat(end_date_pred) - start).days


start_date_ref = '2010-01-01'  # days since epoch (1970-01-01)
end_date_ref = '2018-01-01'  # days since epoch (1970-01-01)
start_date_ref_iso = (date.fromisoformat(start_date_ref) - start).days
end_date_ref_iso = (date.fromisoformat(end_date_ref) - start).days
dates_ref = range(start_date_ref_iso, end_date_ref_iso, 1)



def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """
    bandnames = [(datetime(1970, 1, 1) + timedelta(days=days)).strftime('%Y%m%d') for days in range(start_date_pred_iso, end_date_pred_iso, step)]
    bandnames.append("std")
    return bandnames

# regressor
# define all three models from the paper
def objective_simple(x, a0, a1, b1, c1):
    return a0 + a1 * np.cos(2 * np.pi / 365 * x) + b1 * np.sin(2 * np.pi / 365 * x) + c1 * x


def objective_advanced(x, a0, a1, b1, c1, a2, b2):
    return objective_simple(x, a0, a1, b1, c1) + a2 * np.cos(4 * np.pi / 365 * x) + b2 * np.sin(4 * np.pi / 365 * x)


def objective_full(x, a0, a1, b1, c1, a2, b2, a3, b3):
    return objective_advanced(x, a0, a1, b1, c1, a2, b2) + a3 * np.cos(6 * np.pi / 365 * x) + b3 * np.sin(
        6 * np.pi / 365 * x)


# - choose which model to use
objective = objective_full

def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """
    inarray:   numpy.ndarray[nDates, nBands, nrows, ncols](Int16)
    outarray:  numpy.ndarray[nOutBands](Int16) initialized with no data values
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    nodata:    int
    nproc:     number of allowed processes/threads
    Write results into outarray.
    """
    idx = np.argwhere(np.isin(dates, np.array(dates_ref)))
    inarray = inarray.astype(np.float32)
    inarray = inarray[:, :, 0, 0]
    invalid = inarray == nodata
    valid = np.where(inarray[:, 0] != nodata)[0]  # skip no data; just check first band
    # print(valid)
    if len(valid) == 0:
        return
    inarray[invalid] = np.nan
    #print(len(dates))


    # prepare data
    #inarray = inarray[:, :, 0, 0]
    #inarray[inarray != nodata] = 0
    #print(inarray[valid])
    # band indices
    green = np.argwhere(bandnames == b'GREEN')[0][0]
    red = np.argwhere(bandnames == b'RED')[0][0]
    nir = np.argwhere(bandnames == b'NIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]
    #print(bandnames)
    # calculate DSWI ((Band 8 (NIR) + Band 3 (Green)) / (Band 11 (SWIR1) + Band 4 (Red)))
    dswi = (inarray[:,nir] + inarray[:,green]) / (inarray[:,swir1] + inarray[:,red])
    #dswi = np.sum(inarray[:, [green, nir]], axis=1) / np.sum(inarray[:, [red, swir1]], axis=1)
    #dswi = np.subtract(inarray[:, nir],inarray[:, swir1]) / np.sum(inarray[:, [nir, swir1]], axis=1)
    #dswi = inarray[:,green]
    #print(dswi)
    #print(dates)
    #print(len(dswi))

    #print(np.array(len(dates)))

    #print(idx)
    #time.sleep(1000)
    #valid = valid[idx]
    #dswi = dswi[idx]


    dswi = dswi[idx]
    dates = dates[idx]
    #print(idx)
    #print(valid)
    #print(dswi.shape)
    #ytrain = dswi[valid][np.isfinite(dswi[valid])]
    #xtrain = dates[valid][np.isfinite(dswi[valid])]
    ytrain = dswi[valid][np.logical_and(dswi[valid]<5,dswi[valid]>-5,np.isfinite(dswi[valid]))]
    xtrain = dates[valid][np.logical_and(dswi[valid]<5,dswi[valid]>-5,np.isfinite(dswi[valid]))]

    #print(len(ytrain))
    # fit
    #try:
    popt, _ = curve_fit(objective, xtrain, ytrain)
    #except:
        #print(xtrain)
        #print(ytrain)

    # predict
    xtest = np.array(range(start_date_pred_iso, end_date_pred_iso, step))
    ytest = objective(xtest, *popt)
    #
    # # store results
    #print(xtest)
    #print(ytest)

    #valid = np.isfinite(dswi)
    #outarray[valid] = dswi[valid]
    #outarray[:, :, 0, 0]=dswi[:, 0]
    #outarray[:] = ytest
    outarray[:-1]= ytest*100
    outarray[-1:] = np.nanstd(ytrain)*100