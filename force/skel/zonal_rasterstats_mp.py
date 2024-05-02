# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:52:04 2021

@author: Administrator
"""

import itertools
import multiprocessing
import geopandas
from rasterstats import point_query
import fiona
import time


shp = r"s"
tif = r"t"
output = r"o"

def chunks(data, n):
    """Yield successive n-sized chunks from a slice-able iterable."""
    for i in range(0, len(data), n):
        yield data[i:i+n]


def zonal_stats_partial(feats):
    """Wrapper for zonal stats, takes a list of features"""
    return point_query(feats, tif)


if __name__ == '__main__':
    
    startzeit = time.time()
    with fiona.open(shp) as src:
        features = list(src)

    # Create a process pool using all cores
    #cores = multiprocessing.cpu_count()
    cores=15
    p = multiprocessing.Pool(cores)

    # parallel map
    stats_lists = p.map(zonal_stats_partial, chunks(features, cores))

    # flatten to a single list
    values = list(itertools.chain(*stats_lists))
    
    shape=geopandas.read_file(shp)
    shape["value"] = values

    shape.to_file(output)
    #endzeit = time.time()
    #print("process finished in "+str((endzeit-startzeit)/60)+" minutes")
    #print(len(stats))
    #print(len(features))
    assert len(values) == len(features)
    


