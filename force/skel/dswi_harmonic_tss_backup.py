import numpy as np
import time
from datetime import datetime, timedelta, date
from scipy.optimize import curve_fit
# some global config variables
start = date.fromisoformat('1970-01-01')

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    return [f'{(start+timedelta(days=int(dat))).year}{(start+timedelta(days=int(dat))).month:02d}{(start+timedelta(days=int(dat))).day:02d}_{sens.decode("utf-8")}' for dat, sens in zip(dates, sensors)]


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
    inarray = inarray.astype(np.float32)
    inarray = inarray[:, :, 0, 0]
    invalid = inarray == nodata ## bool mask values
    ## bool qai masks


    valid = np.where(inarray[:, 0] != nodata)[0]  # skip no data; just check first band

    if len(valid) == 0:
        return
    inarray[invalid] = np.nan
    #print(len(dates))

    # prepare data
    #inarray = inarray[:, :, 0, 0]
    #inarray[inarray != nodata] = 0
    #print(inarray[valid])
    # band indices
    #print(start+timedelta(days=int(dates[0])))
    #print(dates[0])
    #time.sleep(1000)
    green = np.argwhere(bandnames == b'GREEN')[0][0]
    red = np.argwhere(bandnames == b'RED')[0][0]
    #nir = np.argwhere(bandnames == b'BROADNIR')[0][0]
    nir = np.argwhere(bandnames == b'NIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]

    vals = inarray[valid,:]
    # calculate DSWI ((Band 8 (NIR) + Band 3 (Green)) / (Band 11 (SWIR1) + Band 4 (Red)))
    dswi = (vals[:,nir] + vals[:,green]) / (vals[:,swir1] + vals[:,red])
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


    #dswi = dswi[idx]
    #dates = dates[idx]
    #print(idx)
    #print(valid)
    #print(dswi.shape)
    #ytrain = dswi[valid][np.isfinite(dswi[valid])]
    #xtrain = dates[valid][np.isfinite(dswi[valid])]

    #print(len(ytrain))
    # fit
    #try:
    #popt, _ = curve_fit(objective, xtrain, ytrain)
    #except:
        #print(xtrain)
        #print(ytrain)
        #time.sleep(1000)
    # predict
    #xtest = np.array(range(start_date_pred_iso, end_date_pred_iso, step))
    #ytest = objective(xtest, *popt)
    #
    # # store results
    #print(xtest)
    #print(ytest)

    #dswi = dswi[np.isfinite(dswi)]
    #valid = np.isfinite(dswi)
    #outarray[valid] = dswi[valid]
    #outarray[:, :, 0, 0]=dswi[:, 0]
    #outarray[:] = ytest
    #print(inarray[:, green])
    #print(inarray[:, nir])
    #print(inarray[:, swir1])
    #print(inarray[:, red])
    #print(dswi)
    #dswi = dswi *100
    #print(dswi)
    #print(dswi * 100)
    #time.sleep(10000)

    outarray[valid]= dswi*100
