from importlib import import_module
import xlrd
import pandas
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join, dirname
import sys

np.set_printoptions(threshold=np.inf)
sys.path.append('..')


def check_overflow(frame_start, frame_end):
    if frame_start < 4:
        frame_end = frame_end - frame_start + 4
        frame_start = 4
    elif frame_end > 6000:
        frame_start = 6000 - (frame_end - frame_start)
        frame_end = 6000
    return frame_start, frame_end


class TransitionDataset(Dataset):
    def __init__(self, set='train'):
        self.dataset_path = '/home/zhong/Dataset/C-MHAD'
        self.dataset_group = 'TransitionMovementsApplication'
        self.subject_index = list(range(1, 13))
        self.label_info = {}
        self.num_channel = 6
        self.freq = 50
        self.seq_length = 10 * self.freq  # 10 seconds
        self.seq_skip = 2 * self.freq  # 2 seconds
        if set == 'train':
            self.trial_num = list(range(1, 10))
        else:
            self.trial_num = list(range(10, 11))
        for subject in self.subject_index:
            sheet = self.load_label_sheet(subject)
            self.label_info[subject] = {}
            for trial in self.trial_num:
                self.label_info[subject][trial] = None
            for row in range(1, sheet.nrows):
                trial = int(sheet.cell_value(row, 0))
                if not trial in self.trial_num:
                    continue
                df = self.load_data_frame(subject, trial)
                miss_frames = 6005 - len(df.index)
                action = int(sheet.cell_value(row, 1))
                start_frame = int(self.freq * sheet.cell_value(row, 2)) - miss_frames
                end_frame = int(self.freq * sheet.cell_value(row, 3)) - miss_frames
                start_frame, end_frame = check_overflow(start_frame, end_frame)
                if self.label_info[subject][trial] is None:
                    self.label_info[subject][trial] = torch.zeros(len(df.index) - 2)
                self.label_info[subject][trial][start_frame:end_frame] = action
        self.samples = self.generate_samples()

    def load_full_trial(self, subject, trial):
        sheet = self.load_label_sheet(subject)
        df = self.load_data_frame(subject, trial)
        y = torch.zeros(len(df.index) - 2)
        for row in range(1, sheet.nrows):
            trial_temp = int(sheet.cell_value(row, 0))
            if trial_temp != trial:
                continue
            miss_frames = 6005 - len(df.index)
            action = int(sheet.cell_value(row, 1))
            start_frame = int(self.freq * sheet.cell_value(row, 2)) - miss_frames
            end_frame = int(self.freq * sheet.cell_value(row, 3)) - miss_frames
            start_frame, end_frame = check_overflow(start_frame, end_frame)
            y[start_frame:end_frame] = action
        y = y.type(torch.LongTensor)
        x = df.iloc[2:, 1:].astype(float)
        x = x.to_numpy()
        x = torch.from_numpy(x).type(torch.FloatTensor)
        assert x.shape[0] == y.shape[0]
        return x, y

    def get_label_path(self, subject):
        label_path = join(self.dataset_path,
                          self.dataset_group,
                          'Subject{}'.format(subject),
                          'ActionOfInterestTraSubject{}.xlsx'.format(subject)
                          )
        return label_path

    def load_label_sheet(self, subject):
        label_path = self.get_label_path(subject)
        workbook = xlrd.open_workbook(label_path)
        sheet = workbook.sheet_by_index(0)
        return sheet

    def load_data_frame(self, subject, trial):
        label_path = self.get_label_path(subject)
        data_path = join(dirname(label_path),
                         'InertialData',
                         'inertial_sub{}_tr{}.csv'.format(subject, trial)
                         )
        df = pandas.read_csv(data_path)
        return df

    def generate_samples(self):
        samples = []
        for subject in self.subject_index:
            for trial in self.trial_num:
                label_seq = self.label_info[subject][trial]
                end_flag = False
                start_frame = 0
                while end_flag is False:
                    sample = {'subject': subject, 'trial': trial, 'start_frame': start_frame}
                    samples.append(sample)
                    start_frame += self.seq_skip
                    if start_frame + self.seq_length >= len(label_seq):
                        start_frame -= start_frame + self.seq_length - len(label_seq)
                        sample = {'subject': subject, 'trial': trial, 'start_frame': start_frame}
                        samples.append(sample)
                        end_flag = True
        return samples

    def load_sample_data(self, sample):
        subject, trial, start_frame = sample['subject'], sample['trial'], sample['start_frame']
        df = self.load_data_frame(subject, trial)
        df = df.iloc[2:, 1:].astype(float)
        df = df.to_numpy()
        label_seq = self.label_info[subject][trial]
        assert df.shape == (len(label_seq), self.num_channel)
        x = df[start_frame:start_frame + self.seq_length]
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = label_seq[start_frame:start_frame + self.seq_length].type(torch.LongTensor)
        return x, y

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x, y = self.load_sample_data(sample)
        assert x.shape == (self.seq_length, self.num_channel)
        assert y.shape == (self.seq_length,)
        return x, y

    def __len__(self):
        return len(self.samples)


class Dataloader(object):
    def __init__(self, para, ds_type='train'):
        # loader: x -> (n,s,f), y -> (n,s)
        self.para = para
        dataset_name = para.dataset
        module = import_module('data.' + dataset_name)
        bs = para.batch_size
        self.dataset = module.TransitionDataset(set=ds_type)
        ds_len = len(self.dataset)
        if para.test_state:
            self.loader = DataLoader(
                dataset=self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True
            )
            self.loader_len = ds_len
        else:
            self.loader = DataLoader(
                dataset=self.dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len
