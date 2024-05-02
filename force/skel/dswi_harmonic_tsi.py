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

def objective_simple_notrend(x, a0, a1, b1):
    return a0 + a1 * np.cos(2 * np.pi / 365 * x) + b1 * np.sin(2 * np.pi / 365 * x)
def objective_advanced_notrend(x, a0, a1, b1, a2, b2):
    return objective_simple_notrend(x, a0, a1, b1) + a2 * np.cos(4 * np.pi / 365 * x) + b2 * np.sin(4 * np.pi / 365 * x)
def objective_full_notrend(x, a0, a1, b1, a2, b2, a3, b3):
    return objective_advanced_notrend(x, a0, a1, b1, a2, b2) + a3 * np.cos(6 * np.pi / 365 * x) + b3 * np.sin(
        6 * np.pi / 365 * x)

# - choose which model to use
objective = objective_full_notrend

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
    blue = np.argwhere(bandnames == b'BLUE')[0][0]
    red = np.argwhere(bandnames == b'RED')[0][0]
    #re1 = np.argwhere(bandnames == b'REDEDGE1')[0][0]
    #re2 = np.argwhere(bandnames == b'REDEDGE2')[0][0]
    nir = np.argwhere(bandnames == b'NIR')[0][0]
    bnir = np.argwhere(bandnames == b'BROADNIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]
    #swir2 = np.argwhere(bandnames == b'SWIR2')[0][0]
    #print(bandnames)
    # NBR = (BNIR - SWIR2) / (BNIR + SWIR2)
    # nbr = (inarray[:, bnir] - inarray[:, swir2]) / (inarray[:, bnir] + inarray[:, swir2])
    # # NDVI = (BNIR - RED) / (BNIR + RED)
    # ndvi = (inarray[:, bnir] - inarray[:, red]) / (inarray[:, bnir] + inarray[:, red])
    # # ARI = BNIR * ((1 / GREEN) - (1 / RE1))
    # ari = inarray[:, bnir] * ((1 / inarray[:, green]) - (1 / inarray[:, re1]))
    # # CRI = (1 / BLUE) - (1 / GREEN)
    # cri = (1 / inarray[:, blue]) - (1 / inarray[:, green])
    # # RENDVI1 = (RE1 - RED) / (RE1 + RED)
    # rendvi1 = (inarray[:, re1] - inarray[:, red]) / (inarray[:, re1] + inarray[:, red])
    # # RENDVI2 = (RE2 - RED) / (RE2 + RED)
    # rendvi2 = (inarray[:, re2] - inarray[:, red]) / (inarray[:, re2] + inarray[:, red])
    # # DSWI = (BNIR + GREEN) / (SWIR1 + RED)
    dswi = (inarray[:, bnir] + inarray[:, green]) / (inarray[:, swir1] + inarray[:, red])
    # # MSI = SWIR1 / BNIR
    # msi = inarray[:, swir1] / inarray[:, bnir]
    # # NDWI = (BNIR - SWIR1) / (BNIR + SWIR1)
    # ndwi = (inarray[:, bnir] - inarray[:, swir1]) / (inarray[:, bnir] + inarray[:, swir1])
    # # VMI = ((BNIR + 0.1) - (SWIR2 + 0.02)) / ((BNIR + 0.1) + (SWIR2 + 0.02))
    # vmi = ((inarray[:, bnir] + 0.1) - (inarray[:, swir2] + 0.02)) / ((inarray[:, bnir] + 0.1) + (inarray[:, swir2] + 0.02))
    # # CCCI = ((BNIR - RE1) / (BNIR + RE1)) / ((BNIR - RED) / (BNIR + RED))
    # ccci = ((inarray[:, bnir] - inarray[:, re1]) / (inarray[:, bnir] + inarray[:, re1])) / (
    #         (inarray[:, bnir] - inarray[:, red]) / (inarray[:, bnir] + inarray[:, red]))

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