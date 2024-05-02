import numpy as np
from datetime import timedelta, date

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """
    out_bands = []
    for d,s in zip(dates,sensors):
        d = date.fromisoformat('1970-01-01') + timedelta(days=int(d))
        ds = str(d).replace('_','') + "_" + s.decode('UTF-8')
        out_bands.append(ds)
    return out_bands


def forcepy_block(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
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

    # prepare data
    inarray = inarray.astype(np.float32)  # cast to float ...
    invalid = inarray == nodata
    if np.all(invalid):
        return
    inarray[invalid] = np.nan        # ... and inject NaN to enable np.nan*-functions
    green = np.argwhere(bandnames == b'GREEN')[0][0]
    red = np.argwhere(bandnames == b'RED')[0][0]
    nir = np.argwhere(bandnames == b'NIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]

    # calculate DSWI ((Band 8 (NIR) + Band 3 (Green)) / (Band 11 (SWIR1) + Band 4 (Red)))
    #dswi = np.nansum(inarray[:, [0, 2]], axis=1) / np.nansum(inarray[:, [1, 3]], axis=1)
    #dswi = np.sum(inarray[:, [green, nir]], axis=1) / np.sum(inarray[:, [red, swir1]], axis=1)
    #dswi = (inarray[:, nir,:,:] + inarray[:, green,:,:]) / (inarray[:, swir1,:,:] + inarray[:, red,:,:])

    #print(len(dswi))
    #print(inarray)

    # store results


    value = np.sum(inarray[:,[green, nir],:,:], axis=1) / np.sum(inarray[:,[red, swir1],:,:], axis=1)
    valid = np.isfinite(value)
    outarray[valid] = np.round(value * 1000)[valid]

    #print(np.shape(outarray))
    #print(outarray[~np.isnan(outarray)])

