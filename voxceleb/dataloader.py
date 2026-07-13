#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchaudio.transforms import MelSpectrogram, Resample
import numpy as np
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import time, itertools

import bisect
import h5py

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats,axis=0).astype(float)

    return feat;

class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4)

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)

        rir, fs     = soundfile.read(rir_file)
        rir         = np.expand_dims(rir.astype(float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


class train_dataset_loader(Dataset):
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, **kwargs):

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        # old wrong code - right?
        # with open(train_list, 'r') as f:self.train_list = [line for line in f]
        self.train_list = train_list
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment

        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines();

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split();

            speaker_label = dictkeys[data[0]];
            filename = os.path.join(train_path,data[1]);

            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indices):

       # If indices is an integer, convert to a list to allow for iteration
        if type(indices) is int:
            indices = [indices]

        feat = []

        for index in indices:

            audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)

            if self.augment:
                augtype = random.randint(0,4)
                if augtype == 1:
                    audio   = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio   = self.augment_wav.additive_noise('music',audio)
                elif augtype == 3:
                    audio   = self.augment_wav.additive_noise('speech',audio)
                elif augtype == 4:
                    audio   = self.augment_wav.additive_noise('noise',audio)

            feat.append(audio);

        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)



class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval   = num_eval
        self.test_path  = test_path

        lines = []
        files = []
        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        self.test_list = list(set(files))
        self.test_list.sort()


    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path,self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label         = data_source.data_label
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk
        self.batch_size         = batch_size
        self.epoch              = 0
        self.seed               = seed
        self.distributed        = distributed

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)


        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data    = data_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)

            rp      = lol(np.arange(numSeg),self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size())
            start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
            end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])


    def __len__(self) -> int:
        return self.num_samples

    def info(self) -> None:
        print("No info available")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def get_relative_path(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)

class VoxCelebDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_rate=16000):
        """
        Args:
            root_dir (string): Directory with all the speaker folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sample_rate (int): Desired sample rate for the audio files.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.audio_files = []
        self.speaker_ids = []

        # Walk through each speaker's directory and collect audio file paths
        for speaker_id in os.listdir(root_dir):
            speaker_folder = os.path.join(root_dir, speaker_id)
            if os.path.isdir(speaker_folder):
                for file_name in os.listdir(speaker_folder):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(speaker_folder, file_name)
                        self.audio_files.append(file_path)
                        self.speaker_ids.append(speaker_id)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        speaker_id = self.speaker_ids[idx]

        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Apply MelSpectrogram transform if specified
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, int(speaker_id)

def load_voxceleb_dataset(datapath='voxceleb/data', batch_size=128, threads=2, data_split=1, split_idx=0):
    """
    Setup dataloader for the VoxCeleb dataset.

    Args:
        datapath: Path to the VoxCeleb dataset folder.
        batch_size: Number of samples per batch.
        threads: Number of subprocesses to use for data loading.
        data_split: Number of splits for the training dataloader.
        split_idx: The index for the split of the dataloader.

    Returns:
        train_loader, test_loader
    """
    assert split_idx < data_split, 'The index of data partition should be smaller than the total number of splits'

    # Define transformation
    transform = MelSpectrogram(sample_rate=16000, n_mels=128)

    # Load dataset
    dataset = VoxCelebDataset(root_dir=datapath, transform=transform, sample_rate=16000)

    # Split the dataset if needed
    if data_split > 1:
        indices = torch.tensor(np.arange(len(dataset)))
        data_num = len(dataset) // data_split
        ind_start = data_num * split_idx
        ind_end = min(data_num * (split_idx + 1), len(dataset))
        train_indices = indices[ind_start:ind_end]
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=threads)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=threads)

    # Set up test loader (using the entire dataset here for simplicity; usually you’d have a separate test split)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=threads)

    return train_loader, test_loader


class train_dataset_loader_h5py(Dataset):
    """
    Dataset loader for h5py files
    """
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, **kwargs):
        if not h5py.is_hdf5(train_list):
            raise ValueError('The path to the train_list is not a valid h5py file.\nTry setting train_list to "data/SpeakerDataVox2.hdf5"')

        self.max_frames = max_frames
        self.data = h5py.File(train_list, 'r')
        self.utterance_per_speaker = []
        self.num_speakers = len(self.data.keys())
        for i in range(self.num_speakers):
            self.utterance_per_speaker.append(self.data["speaker_{}".format(i)].attrs['num_utterances'])

    def __getitem__(self, index):
        speaker, utterance = index
        # single int to list so the next for loop works
        if type(utterance) is int:
            utterance = [utterance]

        max_audio = self.max_frames * 160 + 240

        feat = []
        for u in utterance:
            audios = []
            audio = self.data["speaker_{}".format(speaker)]["utterance_{}".format(u)]
            audiosize = audio.shape[0]
            if audiosize <= max_audio:
                shortage    = max_audio - audiosize + 1
                audio       = np.pad(audio, (0, shortage), 'wrap')
                audiosize   = audio.shape[0]
            startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
            for asf in startframe:
                audios.append(audio[int(asf):int(asf)+max_audio])
            feat.append(np.stack(audios,axis=0))

        feat = np.concatenate(feat, axis=0)
        return torch.FloatTensor(feat), speaker

    def __len__(self):
        return sum(self.utterance_per_speaker)

def sample_noreplace(arr, n, k):
    assert k <= len(arr)
    idx = np.random.randint(len(arr) - np.arange(k), size=[n, k])
    for i in range(k-1, 0, -1):
        idx[:,i:] += idx[:,i:] >= idx[:,i-1,None]
    return np.array(arr)[idx]

class train_dataset_sampler_h5py(torch.utils.data.Sampler):
    '''
    creates a list of tuples (speaker, [utterances]) which is than batched by the dataloader in tensors (batch_size, utterances, audio_length)
    the number of utterances = nPerSpeaker value in the config file
    '''
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):
        self.utterance_per_speaker   = data_source.utterance_per_speaker
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk
        self.batch_size         = batch_size
        self.epoch              = 0
        self.seed               = seed
        self.distributed        = distributed
        self.num_speakers = data_source.num_speakers
        self.datasource = data_source

    def __iter__(self):
        speaker_available = [] # stores which speakers still have utterances left
        speaker_batches = [] # stores the utterances of each speaker, randomly shuffled and divided into nPerSpeaker segments
        num_speaker_batches = [] # stores the number(amount) of segments for each speaker

        start_time = time.time()

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        rng = np.random.default_rng(self.seed + self.epoch)

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        for i in range(self.num_speakers):
            numSeg  = round_down(min(self.utterance_per_speaker[i],self.max_seg_per_spk),self.nPerSpeaker)
            # shuffle indices of all utterances with torch.randperm but then cut down the list to only numSeg samples
            # such that they can be cleanly partitioned with lol-function
            speaker_batch = lol(torch.randperm(self.utterance_per_speaker[i], generator=g).tolist()[:numSeg],self.nPerSpeaker)
            speaker_batches.append(speaker_batch)
            num_speaker_batches.append(len(speaker_batch))
            if numSeg > 0:
                speaker_available.append(i)

        # for logging
        self.available_speakers = len(speaker_available)

        # holds the samples in a continues list, which is mixed correctly
        mixed_list = []
        assert len(speaker_available) >= self.batch_size, "Not enough speakers available to create a batch of size {} ({} Speakers available -> max batch size for this dataset is {})".format(self.batch_size, len(speaker_available), len(speaker_available))
        while len(speaker_available) >= self.batch_size:
            # choose batch_size number of speakers without replacement, meaning no speaker is repeated in the batch
            speakers = rng.choice(speaker_available,size=self.batch_size ,replace=False)
            for speaker in speakers:
                # if the speaker has no more utterances left, remove it from the list of available speakers
                num_speaker_batches[speaker] -= 1
                if num_speaker_batches[speaker] == 0:
                    del speaker_available[bisect.bisect_left(speaker_available, speaker)]

                # add the utterances of the speaker to the mixed_list
                try:
                    mixed_list.append((speaker, speaker_batches[speaker].pop()))
                except:
                    print("Error: Speaker {} has no more utterances left".format(speaker))
                    exit()

        self.num_samples = len(mixed_list)
        end_time = time.time()
        # print("Time setting up Dataloader: ", end_time - start_time)
        # print("Dataloader contains {} samples with {} utterances per sample which amount to {} batches with a batch-size of {}".format(self.num_samples, self.nPerSpeaker, self.num_samples//self.batch_size, self.batch_size))
        return iter(mixed_list)

    def __len__(self) -> int:
        return self.num_samples

    def info(self) -> None:
        print("Total Number of audio samples in dataset: ", len(self.datasource))
        print("Number of used samples (considering needing nPerSpeaker samples per batch and limiting samples for one speaker to max_seg_per_spk in an epoch): ", self.num_samples*self.nPerSpeaker)
        print("    => {} % of the dataset is used".format(round(( self.num_samples*self.nPerSpeaker / len(self.datasource) * 100), 2)))
        print("Number of distinct speakers: ", self.num_speakers, "  ", round(self.available_speakers / self.num_speakers * 100,2), "% of speakers used (", self.available_speakers, ")")
        print("Number of batches of size ",self.batch_size," with ",self.nPerSpeaker," utterances per speaker: ",self.num_samples//self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch