import os
import glob
import tqdm
import torch
import numpy as np
import pandas as pd

from functools import partial
from torch.utils.data import Dataset

def collate_fn(data):
    id_, video_tensor, labels = zip(*data)
    video_tensor = torch.stack(video_tensor, 0)
    labels = torch.stack(labels, dim=0).unsqueeze(1)
    return id_, video_tensor, labels

def getDatasets(params):

    path_volumes_train = params['data']['path_volumes_train']
    path_volumes_valid = params['data']['path_volumes_train']
    path_volumes_test  = params['data']['path_volumes_test']
    
    path_labels_train = params['data']['path_labels_train']
    path_labels_valid = params['data']['path_labels_val']
    path_labels_test  = params['data']['path_labels_test']
    
    path_reports  = params['data']['path_reports']
    path_metadata = params['data']['path_metadata']

    train = CTVolumeDataset(
        path_volumes_train, 
        path_labels_train, 
        path_reports, 
        path_metadata, 
        params, 
        split="train"
    )
    valid = CTVolumeDataset(
        path_volumes_valid, 
        path_labels_valid, 
        path_reports, 
        path_metadata, 
        params, 
        split="valid"
    )
    test  = CTVolumeDataset(
        path_volumes_test, 
        path_labels_test, 
        path_reports, 
        path_metadata, 
        params, 
        split='test'
    )

    return train, valid, test


class CTVolumeDataset(Dataset):
    def __init__(self, 
                 path_volumes, 
                 path_labels, 
                 path_reports, 
                 path_metadata, 
                 params, 
                 split
                ):

        # device
        self.device = params['data']['device']

        # important features
        self.params        = params
        self.split         = split
        self.data_folder   = path_volumes

        # folder with volumes
        self.path_labels = path_labels

        # file with reports
        self.path_reports = path_reports

        # file with metadata
        self.path_metadata = path_metadata

        # read metadata
        self.metadata = pd.read_csv(self.path_metadata)

        # paths of nii files
        self.paths   = []
        self.samples = self.prepare_samples_ctrate()

        # read labels
        self.df_labels = pd.read_csv(self.path_labels)
        self.df_labels = self.intToFloat(self.df_labels)
        self.samples_names = list(self.df_labels['VolumeName'])
        self.labels_names = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly', 
            'Pericardial effusion', 'Coronary artery wall calcification', 
            'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis', 
            'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 
            'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 
            'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
        ]

        # transform
        self.nii_to_tensor = partial(self.nii_img_to_tensor)

        # target shape
        self.target_shape = (480, 480, 240)

    def prepare_samples_ctrate(self):
        samples = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(accession_folder, '*.npz')):
                    samples.append(nii_file)
                    self.paths.append(nii_file)
        return samples

    def intToFloat(self, df):
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col] = df[col].astype(float)
        return df

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, dd=240, dh=480, dw=480):

        # Warning: To adjust 
        # Assuming that the CT scan is already formatted with SLP orientation with HU values
        array = np.load(path)['arr_0']
    
        # Array to tensor
        tensor = torch.tensor(array)
    
        # Clip Hounsfield Units to [-1000, +200]
        tensor = torch.clip(tensor, -1000., +200.)
    
        # Shift to [0, +1200]
        tensor = tensor + torch.tensor(+1000., dtype=torch.float32)
    
        # Map [0, +1200] to [0, 1]
        tensor = tensor / torch.tensor(+1200., dtype=torch.float32)
    
        # ImageNet Normalization
        tensor = tensor + torch.tensor(-0.449, dtype=torch.float32)

        # extract dimensions
        d, h, w = tensor.shape

        # calculate cropping values for height, width, and depth
        dd, dh, dw = (240, 480, 480)
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # crop
        tensor = tensor[d_start:d_end, h_start:h_end, w_start:w_end]

        # # calculate padding values for height, width, and depth
        pad_h_before = (dh - tensor.size(1)) // 2
        pad_h_after  = dh - tensor.size(1) - pad_h_before
        pad_w_before = (dw - tensor.size(2)) // 2
        pad_w_after  = dw - tensor.size(2) - pad_w_before
        pad_d_before = (dd - tensor.size(0)) // 2
        pad_d_after  = dd - tensor.size(0) - pad_d_before

        # pad
        tensor = torch.nn.functional.pad(tensor, (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after), value=-0.449)

        # unsqueeze
        tensor = tensor.unsqueeze(0)

        return tensor

    def get_labels(self, nii_file):
        
        # extract id of patient
        id_ = os.path.basename(nii_file)[:-4]

        # extract age 
        age = self.metadata[self.metadata.VolumeName == id_]["PatientAgeNorm"].values[0]

        # extract sex
        sex = self.metadata[self.metadata.VolumeName == id_]["PatientSex"].values[0]

        # extract label
        labels = self.df_labels[self.df_labels.VolumeName == id_][self.labels_names].values[0]

        return id_, torch.from_numpy(labels), torch.tensor(age), torch.tensor(sex)

        
    def __getitem__(self, index):

        # get path of nii_file
        nii_file = self.samples[index]

        # load volume and perform transformation
        video_tensor = self.nii_to_tensor(nii_file)

        # load labels
        id_, labels = self.get_labels(nii_file)

        return id_, video_tensor, labels

