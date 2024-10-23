import torch
import torch.utils.data
import pandas as pd
import os
import numpy as np
from numpy import genfromtxt
import tqdm
import glob
from datetime import datetime
import pickle

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root, classes, cache=True, seed=0, response = None, norm = None, norm_response = None, thermal = None):

        self.seed = seed

        # ensure that different seeds are set per partition
        #seed += sum([ord(ch) for ch in partition])
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self.norm = norm
        self.norm_r = norm_response
        self.root = root
        self.response = response
        self.trainids = os.path.join(self.root, "csv")
        self.validids = os.path.join(self.root, "csv")
        self.thermal = thermal

        classes = np.array(classes)
        self.classes = np.unique(classes)
        self.nclasses = len(self.classes)

        self.data_folder = "{root}/csv".format(root=self.root)

        self.cache = os.path.join(self.root,"npy")
        self.cache = self.cache.replace("\\", "/")

        if cache and self.cache_exists():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print("no cached dataset found. iterating through csv folders in " + str(self.data_folder))
            self.cache_dataset()

        self.hist, _ = np.histogram(self.y, bins=self.nclasses)

        print("Loaded {} Reference Samples".format(len(self.ids)))
        #print("class frequencies " + ", ".join(["{c}:{h}".format(h=h, c=c) for h, c in zip(self.hist, self.classes)]))

        #print(self)

    def __str__(self):
        return "Dataset {}. X:{}, y:{} with {} classes and example doy range: {} - {}".format(self.root,str(len(self.X)) +"x"+ str(self.X[0].shape), self.y.shape, self.nclasses, str(min(self.doy[0])), str(max(self.doy[0])))

    def cache_dataset(self):
        ids = glob.glob(f"{self.trainids}/*.csv")
        assert len(ids) > 0

        self.X = list()
        self.doy = list()  # Add a list to store DOY information
        if self.thermal != None:
            self.thermal_time = list()  # Add a list to store DOY information
        self.nutzcodes = list()
        self.ids = list()

        for id in tqdm.tqdm(ids):
            if self.thermal != None:
                X, nutzcode, doy, thermal_time = self.load(id)  # Adjusted to unpack three values
            else:
                X, nutzcode, doy = self.load(id)  # Adjusted to unpack three values

            if len(nutzcode) > 0:
                if self.response == "classification":
                    nutzcode = int(nutzcode[0])
                else:
                    nutzcode = float(nutzcode[0])

                self.X.append(X)
                self.doy.append(doy)  # Store DOY information
                if self.thermal != None:
                    self.thermal_time.append(thermal_time)
                self.nutzcodes.append(nutzcode)
                self.ids.append(id)

        self.y = np.array([nutzcode for nutzcode in self.nutzcodes])
        self.sequencelengths = np.array([len(X) for X in self.X])  # Simplified
        self.sequencelength = max(self.sequencelengths)
        self.ndims = np.array(self.X[0]).shape[1]  # Assuming all X have the same number of dimensions

        if self.thermal != None:
            self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, self.X, self.doy, self.thermal_time)  # Adjusted to also cache DOY
        else:
            self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, self.X,self.doy, None)  # Adjusted to also cache DOY

    def cache_variables(self, y, sequencelengths, ids, ndims, X, doy, thermal_time):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        #doy_array = np.array(doy, dtype=object)
        #np.save(os.path.join(self.cache, "doy.npy"), doy_array)
        #X_array = np.array(X, dtype=object)
        #np.save(os.path.join(self.cache, "X.npy"), X_array)
        #np.savez(os.path.join(self.cache, "doy.npz"), doy)
        #np.savez(os.path.join(self.cache, "X.npz"), X)
        with open(os.path.join(self.cache, "doy.pkl"), 'wb') as f:
            pickle.dump(doy,f)
        with open(os.path.join(self.cache, "X.pkl"), 'wb') as f:
            pickle.dump(X,f)
        if self.thermal != None:
            with open(os.path.join(self.cache, "thermal_time.pkl"), 'wb') as f:
                pickle.dump(thermal_time,f)

    # When saving:


    # When loading:
    def load_cached_dataset(self):
        self.y = np.load(os.path.join(self.cache, "y.npy"))
        self.ndims = int(np.load(os.path.join(self.cache, "ndims.npy")))
        self.sequencelengths = np.load(os.path.join(self.cache, "sequencelengths.npy"))
        self.sequencelength = self.sequencelengths.max()
        self.ids = np.load(os.path.join(self.cache, "ids.npy"))
        #self.doy = np.load(os.path.join(self.cache, "doy.npy"), allow_pickle=True)
        #self.X = np.load(os.path.join(self.cache, "X.npy"), allow_pickle=True)
        #self.doy = np.load(os.path.join(self.cache, "doy.npz"), allow_pickle=True)
        #self.X = np.load(os.path.join(self.cache, "X.npz"), allow_pickle=True)
        with open(os.path.join(self.cache, "doy.pkl"), 'rb') as f:
            self.doy = pickle.load(f)
        with open(os.path.join(self.cache, "X.pkl"), 'rb') as f:
            self.X = pickle.load(f)
        if self.thermal != None:
            with open(os.path.join(self.cache, "thermal_time.pkl"), 'rb') as f:
                self.thermal_time = pickle.load(f)


    def cache_exists(self):
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.pkl"))
        doyxists = os.path.exists(os.path.join(self.cache, "doy.pkl"))
        if self.thermal != None:
            thermaltimexists = os.path.exists(os.path.join(self.cache, "thermal_time.pkl"))
            return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists and doyxists and thermaltimexists
        else:
            return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists and doyxists

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        #os.remove(os.path.join(self.cache, "dataweights.npy"))
        os.remove(os.path.join(self.cache, "X.npz"))
        os.remove(os.path.join(self.cache, "doy.npz"))
        if self.thermal != None:
            os.remove(os.path.join(self.cache, "thermal_time.npz"))
        os.removedirs(self.cache)

    def load(self, csv_file):

        # data = genfromtxt(csv_file, delimiter=',', skip_header=1,filling_values=0) ###was 9999 before!!
        # X = data[:, 3:] * self.norm
        # if self.norm_r == None:
        #     nutzcodes = data[:, 2]
        # elif self.norm_r == "log10":
        #     nutzcodes = np.log10(data[:, 2] + 1)
        # else:
        #     nutzcodes = data[:, 2] * self.norm_r
        #
        # doy = data[:, 1]

        # Read the CSV file with numpy
        data = genfromtxt(csv_file, delimiter=',', skip_header=1, filling_values=0)  # Fill missing values with 0
        if self.thermal != None:
            # Extract data without applying normalization
            X = data[:, 4:]  # Features without normalization
            thermal_time = data[:, 3]  # Target values without normalization
            nutzcodes = data[:, 2]  # Target values without normalization
            doy = data[:, 1]  # Day of year
            return X, nutzcodes, doy, thermal_time

        else:
            # Extract data without applying normalization
            X = data[:, 3:]  # Features without normalization
            nutzcodes = data[:, 2]  # Target values without normalization
            doy = data[:, 1]  # Day of year
            return X, nutzcodes, doy




    def __len__(self):
        return len(self.ids)


    # def __getitem__(self, idx):
    #     X, y, doy = self.X[idx], self.y[idx], self.doy[idx]
    #     X_tensor = torch.from_numpy(X).float()
    #     doy_tensor = torch.from_numpy(doy).float()  # Assuming doy is already a numpy array
    #     y_tensor = torch.tensor(y, dtype=torch.long if self.response == "classification" else torch.float)
    #
    #     return X_tensor, y_tensor, doy_tensor

    def __getitem__(self, idx):
        # Retrieve the raw data
        X_raw = self.X[idx]
        nutzcodes_raw = self.y[idx]
        doy = self.doy[idx]


        # Apply normalization to X
        X = X_raw * self.norm if self.norm is not None else X_raw

        # Apply normalization to nutzcodes
        if self.norm_r is None:
            nutzcodes = nutzcodes_raw
        elif self.norm_r == "log10":
            nutzcodes = np.log10(nutzcodes_raw + 1)
        else:
            nutzcodes = nutzcodes_raw * self.norm_r
        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X).float()
        doy_tensor = torch.from_numpy(doy).float()  # Assuming doy is already a numpy array
        y_tensor = torch.tensor(nutzcodes, dtype=torch.long if self.response == "classification" else torch.float)

        # Check for NaN values in y_tensor
        if torch.isnan(X_tensor).any():
            print(f'NaN values detected in y_tensor at index {idx}')
        if torch.isnan(doy_tensor).any():
            print(f'NaN values detected in X_tensor at index {idx}')
        if torch.isnan(y_tensor).any():
            print(f'NaN values detected in doy_tensor at index {idx}')

        if self.thermal != None:
            thermal_time = self.thermal_time[idx]
            thermal_tensor = torch.from_numpy(thermal_time).float()
            return X_tensor, y_tensor, doy_tensor, thermal_tensor
        else:
            return X_tensor, y_tensor, doy_tensor, None