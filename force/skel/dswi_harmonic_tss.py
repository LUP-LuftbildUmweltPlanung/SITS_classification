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
    blue = np.argwhere(bandnames == b'BLUE')[0][0]
    red = np.argwhere(bandnames == b'RED')[0][0]
    #re1 = np.argwhere(bandnames == b'REDEDGE1')[0][0]
    #re2 = np.argwhere(bandnames == b'REDEDGE2')[0][0]
    nir = np.argwhere(bandnames == b'NIR')[0][0]
    bnir = np.argwhere(bandnames == b'BROADNIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]
    #swir2 = np.argwhere(bandnames == b'SWIR2')[0][0]

    vals = inarray[valid,:]
    # # calculate DSWI ((Band 8 (NIR) + Band 3 (Green)) / (Band 11 (SWIR1) + Band 4 (Red)))
    # dswi = (vals[:,nir] + vals[:,green]) / (vals[:,swir1] + vals[:,red])
    # # NBR = (BNIR - SWIR2) / (BNIR + SWIR2)
    # nbr = (vals[:, bnir] - vals[:, swir2]) / (vals[:, bnir] + vals[:, swir2])
    # # NDVI = (BNIR - RED) / (BNIR + RED)
    # ndvi = (vals[:, bnir] - vals[:, red]) / (vals[:, bnir] + vals[:, red])
    # # ARI = BNIR * ((1 / GREEN) - (1 / RE1))
    # ari = vals[:, bnir] * ((1 / vals[:, green]) - (1 / vals[:, re1]))
    # # CRI = (1 / BLUE) - (1 / GREEN)
    # cri = (1 / vals[:, blue]) - (1 / vals[:, green])
    # # RENDVI1 = (RE1 - RED) / (RE1 + RED)
    # rendvi1 = (vals[:, re1] - vals[:, red]) / (vals[:, re1] + vals[:, red])
    # # RENDVI2 = (RE2 - RED) / (RE2 + RED)
    # rendvi2 = (vals[:, re2] - vals[:, red]) / (vals[:, re2] + vals[:, red])
    # # DSWI = (BNIR + GREEN) / (SWIR1 + RED)
    dswi = (vals[:, bnir] + vals[:, green]) / (vals[:, swir1] + vals[:, red])
    # # MSI = SWIR1 / BNIR
    # msi = vals[:, swir1] / vals[:, bnir]
    # # NDWI = (BNIR - SWIR1) / (BNIR + SWIR1)
    # ndwi = (vals[:, bnir] - vals[:, swir1]) / (vals[:, bnir] + vals[:, swir1])
    # # VMI = ((BNIR + 0.1) - (SWIR2 + 0.02)) / ((BNIR + 0.1) + (SWIR2 + 0.02))
    # vmi = ((vals[:, bnir] + 0.1) - (vals[:, swir2] + 0.02)) / ((vals[:, bnir] + 0.1) + (vals[:, swir2] + 0.02))
    # # CCCI = ((BNIR - RE1) / (BNIR + RE1)) / ((BNIR - RED) / (BNIR + RED))
    # ccci = ((vals[:, bnir] - vals[:, re1]) / (vals[:, bnir] + vals[:, re1])) / (
    #             (vals[:, bnir] - vals[:, red]) / (vals[:, bnir] + vals[:, red]))


    outarray[valid]= dswi*100
