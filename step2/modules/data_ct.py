import os
import re
import glob
import tqdm
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from functools import partial
from torch.utils.data import Dataset

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]

def getDatasets(args, tokenizer):
    
    # path with abnormality binary labels
    path_labels_train     = args.path_labels_train
    path_labels_valid     = args.path_labels_valid
    path_labels_test      = args.path_labels_test

    # path with anomaly binary labels with corresponding sentence
    path_sentences_train = args.path_sentences_train
    path_sentences_valid = args.path_sentences_valid
    path_sentences_test  = args.path_sentences_test
    
    # datasets
    train_ds = CTReportDataset(
        args, 
        path_volumes   = args.trainfolder,  
        path_sentences = path_sentences_train, 
        path_labels    = path_labels_train, 
        tokenizer      = tokenizer, 
    )
    valid_ds = CTReportDataset(
        args, 
        path_volumes   = args.validfolder, 
        path_sentences = path_sentences_valid, 
        path_labels    = path_labels_valid, 
        tokenizer      = tokenizer
    )
    test_ds  = CTReportDataset(
        args, 
        path_volumes   = args.testfolder, 
        path_sentences = path_sentences_test, 
        path_labels    = path_labels_test, 
        tokenizer      = tokenizer
    )

    return train_ds, valid_ds, test_ds


class CTReportDataset(Dataset):
    def __init__(
        self, 
        args, 
        path_volumes, 
        path_sentences, 
        path_labels, 
        tokenizer
    ):
        """
        Initialize CT-RATE dataset.
        args            : config.
        path_volumes    : folder with 3D CT scans.
        path_sentences  : csv file with per-abnormality labels extracted corresponding sentences.
        path_labels     : csv files with binary labels.
        tokenizer       : GPT-2 tokenizer.
        """
        
        # names of labels
        self.labels_names = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly', 
            'Pericardial effusion', 'Coronary artery wall calcification', 
            'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis', 
            'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 
            'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 
            'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
        ]

        # args and paths
        self.args              = args
        self.data_folder       = path_volumes
        self.min_slices        = min_slices

        # tokenizer
        self.tokenizer         = tokenizer
        self.max_seq_length    = args.max_seq_length

        # prepare reports, labels
        self.accession_to_text = self.load_accession_text(path_sentences, path_labels)
        self.labels            = pd.read_csv(path_labels)
        self.paths             = []
        self.normalization     = args.normalization
        self.samples           = self.prepare_samples(path_labels)

        # device to use
        self.device            = args.device_dataset

        self.transform          = transforms.Compose([transforms.Resize((resize_dim,resize_dim)), transforms.ToTensor()])
        self.nii_to_tensor      = partial(self.nii_img_to_tensor, transform = self.transform)
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.target_shape = (480, 480, 240)

    def clean_report(self, report):
        """
        Clean report if needed.
        """
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace('"', '') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def load_accession_text(self, path_sentences, path_labels):
        """
        To load reports.
        """

        # read file with reports
        df               = pd.read_csv(path_sentences)
        df['VolumeName'] = ['_'.join(x.split('_')[:-1]) for x in list(df['VolumeName'])]

        # add special tokens
        bos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        # list with ids with at least one sentence
        list_ids_anomaly = df.VolumeName.unique()

        # list with all ids
        df_anomalies = pd.read_csv(path_labels)
        list_ids_all = ['_'.join(x.split('.')[0].split('_')[:-1]) for x in df_anomalies.VolumeName.unique()]
        list_ids_all = list(np.unique(list_ids_all))

        # list of ids with no anomalies (hence no sentences)
        list_ids_healthy = list(set(list_ids_all) - set(list_ids_anomaly))

        # concatenate 2 lists and sort it
        list_ids = list(list_ids_anomaly) + list_ids_healthy
        list_ids = sorted(list_ids, key=lambda x: int(x.split('_')[1]))

        # dict that will stock volume name and reports.
        accession_to_text = {label: {} for label in list_ids}
        # iterate over label and ids with empty sentence
        for key in accession_to_text:
            accession_to_text[key] = {label: "" for label in self.labels_names}
        # iterate over sentences and add it
        for _, row in df.iterrows():
            accession_to_text[row["VolumeName"]][row['AbnormalityName']] = bos_token + self.clean_report(row["Findings_EN"]) + eos_token

        return accession_to_text

    def prepare_samples(self, labels_file):
        """
        Prepare samples from read reports.
        """

        # load labels
        labels = pd.read_csv(labels_file)

        # extract accession number of patients of interest
        accession_numbers = ['_'.join(x.split('_')[:3]) for x in list(labels['VolumeName'])]        

        samples = []

        # iterate over patients
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):

            # iterate over series
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                accession_number = os.path.basename(accession_folder)
                if accession_number not in self.accession_to_text:
                    continue
                if(accession_number in accession_numbers):
                    
                    #impression_text = self.accession_to_text[accession_number]
                    impression_text = list(self.accession_to_text[accession_number].values())

                    # iterate over volumes
                    for nii_file in glob.glob(os.path.join(accession_folder, '*.npz')):

                        # construct the input text with the included metadata
                        if impression_text == "Not given.":
                            impression_text = ""

                        input_text_concat = str(impression_text)

                        input_text = f'{input_text_concat}'
                        samples.append((nii_file, impression_text))
                        self.paths.append(nii_file)

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform, dd=240, dh=480, dw=480):
        """
        Read the volume and pre-process it.
        """

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
        h_start = max((h - dh) // 2, 0)
        h_end   = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end   = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end   = min(d_start + dd, d)

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
        """
        Read binary labels.
        """
        
        # extract id of patient
        id_ = os.path.basename(nii_file)[:-4]

        # extract label
        labels = self.labels[self.labels.VolumeName == id_][self.labels_names].values[0]
        
        return torch.from_numpy(labels).float()
    
    def __getitem__(self, index):
        """
        Extract all data for a given patients.
        """

        # image, sentences from report and ID
        img, text_list = self.samples[index]
        img_id         = img.split("/")[-1]

        # volume
        tensor = self.nii_to_tensor(img)

        # labels
        labels = self.get_labels(img)

        # text
        encodings   = self.tokenizer(text_list, truncation=True, max_length=self.args.max_seq_length)
        ids         = encodings['input_ids']
        mask        = encodings['attention_mask']
        seq_lengths = [len(x) for x in ids]
        
        # sample to return
        sample = (img_id, labels, tensor, ids, mask, seq_lengths)

        return sample
