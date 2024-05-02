import numpy as np


def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """
    print(dates)
    return dates


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

    inarray = inarray[:, :, 0, 0]
    valid = np.where(inarray[:, 0] != nodata)[0]  # skip no data; just check first band
    #print(valid)
    if len(valid) == 0:
        return

    print(len(dates))
    import time
    time.sleep(1000)
    # prepare data
    #inarray = inarray[:, :, 0, 0]
    #inarray[inarray != nodata] = 0
    #print(inarray[valid])
    # band indices
    green = np.argwhere(bandnames == b'GREEN')[0][0]
    red = np.argwhere(bandnames == b'RED')[0][0]
    nir = np.argwhere(bandnames == b'NIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]

    # calculate DSWI ((Band 8 (NIR) + Band 3 (Green)) / (Band 11 (SWIR1) + Band 4 (Red)))
    dswi = (inarray[:,nir] + inarray[:,green]) / (inarray[:,swir1] + inarray[:,red])
    print(dswi)
    print(dates)
    #
    # # store results
    #valid = np.isfinite(dswi)
    #outarray[valid] = dswi[valid]
    #outarray[:, :, 0, 0]=dswi[:, 0]
    outarray=dswi*100
