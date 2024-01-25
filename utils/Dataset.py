import torch
import torch.utils.data
import pandas as pd
import os
import numpy as np
from numpy import genfromtxt
import tqdm
import glob

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = -1

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root, partition, classes, cache=True, seed=0, response = None):
        assert partition in ["test","train","valid","reference"]

        self.seed = seed

        # ensure that different seeds are set per partition
        seed += sum([ord(ch) for ch in partition])
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        self.root = root
        self.response = response
        self.trainids = os.path.join(self.root, "csv", partition)
        self.validids = os.path.join(self.root, "csv", partition)
        self.partition = partition

        classes = np.array(classes)
        self.classes = np.unique(classes)
        self.nclasses = len(self.classes)

        self.data_folder = "{root}/csv".format(root=self.root)
        #all_csv_files
        #self.csvfiles = [ for f in os.listdir(root)]
        print("Initializing BavarianCropsDataset {} partition".format(self.partition))

        self.cache = os.path.join(self.root,"npy", partition)
        self.cache = self.cache.replace("\\", "/")
        print("read {} classes".format(self.nclasses))

        if cache and self.cache_exists() and not self.mapping_consistent_with_cache():
            self.clean_cache()

        if cache and self.cache_exists() and self.mapping_consistent_with_cache():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print("no cached dataset found. iterating through csv folders in " + str(self.data_folder))
            self.cache_dataset()

        self.hist, _ = np.histogram(self.y, bins=self.nclasses)

        print("loaded {} samples".format(len(self.ids)))
        #print("class frequencies " + ", ".join(["{c}:{h}".format(h=h, c=c) for h, c in zip(self.hist, self.classes)]))

        print(self)

    def __str__(self):
        return "Dataset {}. partition {}. X:{}, y:{} with {} classes".format(self.root, self.partition,str(len(self.X)) +"x"+ str(self.X[0].shape), self.y.shape, self.nclasses)

    def cache_dataset(self):
        """
        Iterates though the data folders and stores y, ids, classweights, and sequencelengths
        X is loaded at with getitem
        """
        #ids = self.split(self.partition)

        ids = glob.glob(f"{self.trainids}/*.csv")
        assert len(ids) > 0

        self.X = list()
        self.nutzcodes = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        #i = 0
        for id in tqdm.tqdm(ids):
            X,nutzcode = self.load(id)
            if len(nutzcode) > 0:
                if self.response == "classification":
                    nutzcode = int(nutzcode[0])
                else:
                    nutzcode = float(nutzcode[0])

                self.X.append(X)
                self.nutzcodes.append(nutzcode)
                self.ids.append(id)
        self.y = np.array([nutzcode for nutzcode in self.nutzcodes])

        self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
        assert len(self.sequencelengths) > 0
        self.sequencelength = self.sequencelengths.max()
        self.ndims = np.array(X).shape[1]

        self.hist,_ = np.histogram(self.y, bins=self.nclasses)
        self.classweights = 1 / self.hist
        #if 0 in self.hist:
        #    classid_ = np.argmin(self.hist)
        #    nutzid_ = self.mapping.iloc[classid_].name
        #    raise ValueError("Class {id} (nutzcode {nutzcode}) has 0 occurences in the dataset! "
        #                     "Check dataset or mapping table".format(id=classid_, nutzcode=nutzid_))


        #self.dataweights = np.array([self.classweights[y] for y in self.y])
        self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, self.X, self.classweights)

    def mapping_consistent_with_cache(self):
        # cached y must have the same number of classes than the mapping
        return True
        #return len(np.unique(np.load(os.path.join(self.cache, "y.npy")))) == self.nclasses

    def cache_variables(self, y, sequencelengths, ids, ndims, X, classweights):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "classweights.npy"), classweights)
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        #np.save(os.path.join(self.cache, "dataweights.npy"), dataweights)
        #print(y)
        #print(X)
        np.save(os.path.join(self.cache, "X.npy"), np.array(X))

    def load_cached_dataset(self):
        # load
        self.classweights = np.load(os.path.join(self.cache, "classweights.npy"))
        self.y = np.load(os.path.join(self.cache, "y.npy"))
        self.ndims = int(np.load(os.path.join(self.cache, "ndims.npy")))
        self.sequencelengths = np.load(os.path.join(self.cache, "sequencelengths.npy"))
        self.sequencelength = self.sequencelengths.max()
        self.ids = np.load(os.path.join(self.cache, "ids.npy"))
        self.X = np.load(os.path.join(self.cache, "X.npy"), allow_pickle=True)

    def cache_exists(self):
        weightsexist = os.path.exists(os.path.join(self.cache, "classweights.npy"))
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.npy"))
        return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists and weightsexist

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "classweights.npy"))
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        #os.remove(os.path.join(self.cache, "dataweights.npy"))
        os.remove(os.path.join(self.cache, "X.npy"))
        os.removedirs(self.cache)

    def load(self, csv_file):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""

        data = genfromtxt(csv_file, delimiter=',', skip_header=1,filling_values=9999)
        X = data[:, 3:] * NORMALIZING_FACTOR
        nutzcodes = data[:, 2]


        return X, nutzcodes


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        load_file = False
        if load_file:
            id = self.ids[idx]
            csvfile = os.path.join(self.data_folder, "{}.csv".format(id))
            X,nutzcodes = self.load(csvfile)
            y = self.nutzcodes
        else:

            X = self.X[idx]
            y = np.array([self.y[idx]] * X.shape[0]) # repeat y for each entry in x
        # pad up to maximum sequence length
        t = X.shape[0]


        npad = self.sequencelengths.max() - t
        X = np.pad(X,[(0,npad), (0,0)],'constant', constant_values=PADDING_VALUE)
        y = np.pad(y, (0, npad), 'constant', constant_values=PADDING_VALUE)


        X = torch.from_numpy(X).type(torch.FloatTensor)
        if self.response == "classification":
            y = torch.from_numpy(y).type(torch.LongTensor)
        else:
            y = torch.from_numpy(y).type(torch.FloatTensor)

        return X, y, self.ids[idx]
