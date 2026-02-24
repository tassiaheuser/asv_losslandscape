"""
    Dataset Statistics Vox2:
    mean:  0.0  std:  0.084  max:  1.0  min:  -1.0
    
    Min number of utterances per speaker :
        Speaker 911, 21 utterances.
        In total 35 Speakers have only 21 utterances.
        The 500 speakers have all less then 35 utterances
    Max number of utterances per speaker :
        Speaker 1689, 500 utterances
    Total number of speakers:
        5994
    Total number of audio files:
        1092009
    The longest audio file is 3523584 samples long
    The shortest audio file is 63488 samples long
    Sample rate: 
        16000

    Loading the whole dataset into Memory takes 62GB



    Vox1:
    Total number of audio files: 
        75222
"""

"""
Findings: 
Voxceleb2_hdf5 is correct.
the voxceleb2_test data has the right amount of utterances: 36237

Voxceleb2_test.hdf is correct.

Voxceleb1_hdf5 is uses the vox2/test_list_refactored.txt list which is the test_list.txt just on one column instead of two.
It has 40 speakers with total 75222 utterances ... which is neighter vox2 or vox1 test
The paths in the test_list_refactored.txt file are only availabe in vox1/voxceleb1 
vox 1           dev	    test:
# of speakers	1,211	40
# of videos	    21,819	677
# of utterances	148,642	4,874

The vox1/voxceleb1 dir has 153516 wav files which is a little more than the 148,642 listed in the webpage for vox1 dev. 
"""


import os
# of speakers	1,211	40
# of videos	21,819	677
# of utterances	148,642	4,874
import numpy as np
import torch
import soundfile
import time
import psutil
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import glob

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    return audio.astype(np.float32)
    

def create_list(path,out_file='train_list.txt'):
    """
    Creates  a list of ids and paths and stores them in a text file.
    It iterates through all .wav fils it finds in path/*/*/*.wav.
    The first folder after path is the id. The path stored in the text file a relative path to the wav file.
    """
    if not os.path.exists(path):
        raise ValueError('Path does not exist.')
    
    files = glob.glob(os.path.join(path, '*/*/*.wav'))
    with open(out_file, 'w') as f:
        for file in files:
            parts = file.split('/')
            id = parts[-3]
            f.write(f"{id} {os.path.relpath(file, path)}\n")

class GetAudio():
    def __init__(self, train_list, max_frames, train_path, **kwargs):


        self.train_list = train_list
        self.max_frames = max_frames 
        # self.non_working_indices = []
        
        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines() 

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split() 

            speaker_label = dictkeys[data[0]] 
            filename = os.path.join(train_path,data[1]) 
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    # @profileit
    def __getitem__(self, index):
        # start = time.time()
        try:
            audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"\nSkipping file {self.data_list[index]} due to error: {e}")

        # end = time.time()
        # print(f"Time taken for loading {len(indices)} files: {end-start}")
        return audio, self.data_label[index]

    def __len__(self):
        return len(self.data_list)


def getAudioVox2():               
    Audio = GetAudio(
        train_list = "/mnt/Games/Tassia_Daten/vox2/train_list.txt", 
        max_frames = 200,
        train_path = "/mnt/Games/Tassia_Daten/vox2/voxceleb2")   
    return Audio

def getAudioVox2_test():               
    Audio = GetAudio(
        train_list = "/mnt/Games/Tassia_Daten/vox2/vox2_test_list_new.txt", #! confirm refactored list
        max_frames = 200,
        train_path = "/mnt/Games/Tassia_Daten/vox2/voxceleb2_test")   
    return Audio

# only execute 
def getAudioVox1():   ###! what
    Audio = GetAudio(
        train_list = "/mnt/Games/Tassia_Daten/vox2/test_list_refactored.txt", 
        max_frames = 200,
        train_path = "/mnt/Games/Tassia_Daten/vox1/voxceleb1")
    return Audio


# Sort into dictionary of file indices for each ID
def create_speaker_dict(Audio):
    indices = list(range(len(Audio)))
    data_dict = {}
    for index in indices:
        speaker_label = Audio.data_label[index]
        if not (speaker_label in data_dict):
            data_dict[speaker_label] = [] 
        data_dict[speaker_label].append(index) 
    return data_dict

def saveAudioLengths(save_path):
    Audio = getAudioVox1()
    audio_lengths = []
    for i in tqdm(range(len(Audio))):
        audio, label = Audio[i]
        audio_lengths.append(audio.size)
    np.save(save_path, np.array(audio_lengths))

def save_to_hdf5(Audio, save_path):
    double_speakers = False
    shorten_audio_index=63488
    print("save dataset as HDF5 to ",save_path)
    data_dict = create_speaker_dict(Audio)
    with h5py.File(save_path, "w") as f:
        p_bar = tqdm(range(len(Audio)))
        for key, value in data_dict.items():
            grp = f.create_group("speaker_" + str(key))
            j = 0
            for i in range(len(value)):
                audio, label = Audio[value[i]]
                if label != key:
                    print(f"Error: key={key} but label={label}")
                if double_speakers:
                    if audio.size < shorten_audio_index*2+1:
                        grp.create_dataset('utterance_' + str(j), data=audio[:shorten_audio_index])
                        j += 1
                    else:
                        grp.create_dataset('utterance_' + str(j) , data=audio[:shorten_audio_index])
                        j += 1
                        grp.create_dataset('utterance_' + str(j) , data=audio[shorten_audio_index:2*shorten_audio_index])
                        j += 1
                else:
                    grp.create_dataset('utterance_' + str(j), data=audio[:shorten_audio_index])
                    j += 1
                p_bar.update(1)
                p_bar.refresh()
            grp.attrs['num_utterances'] = j -1
    print(" ----------------- finished ----------------- ")

def dual_save(Audio,save_path):
    print("save dataset as HDF5 and numpy npz to ",save_path)
    data_dict = create_speaker_dict(Audio)
    label_list = []
    audio_list = []
    with h5py.File(os.path.join(save_path,"SpeakerData.hdf5"), "w") as f:
        p_bar = tqdm(range(len(Audio)))
        for key, value in data_dict.items():
            grp = f.create_group("speaker_" + str(key))
            for i in range(len(value)):
                label_list.append(key)
                audio, label = Audio[value[i]]
                if label != key:
                    print(f"Error: key={key} but label={label}")
                audio_list.append(audio[:63488])
                grp.create_dataset('utterance_' + str(i), data=audio[:63488])
                p_bar.update(1)
                p_bar.refresh()
            grp.attrs['num_utterances'] = len(value)
        np.save(os.path.join(save_path,"labels.npy"), np.array(label_list))
        np.save(os.path.join(save_path,"audio_data.npy"), np.array(audio_list))
    print(" ----------------- finished ----------------- ")

def save_to_numpy(data_dict):
    label_list = []
    audio_list = []
    for key, value in data_dict.items():
        for i in range(len(value)):
            label_list.append(key)
            audio, label = Audio[value[i]]
            if label != key:
                print(f"Error: key={key} but label={label}")
            audio_list.append(audio[:63488])
    np.save("/mnt/ssd2/Tassi/TassiMA/Voxceleb_data/label_list.npy", np.array(label_list))
    np.save("/mnt/ssd2/Tassi/TassiMA/Voxceleb_data/audio_list.npy", np.array(audio_list))
    print(" ----------------- finished ----------------- ")

def find_min_max(Audio):
    total_audio = []
    total_label = []
    current_max = 0
    current_min = float('inf')
    for i in tqdm(range(len(Audio))):
        audio, label = Audio[i]
        length = audio.shape[1]
        if length > current_max:
            current_max = length
            print(f"Current min: {current_min}")
            print(f"Current max: {current_max}")
            print("   ----   ")
        if length < current_min:
            current_min = length
            print(f"Current min: {current_min}")
            print(f"Current max: {current_max}")
            print("   ----   ")
        total_audio.append(total_audio[:63488])
        total_label.append(label)

    print(" ----------------- finished ----------------- ")
    print(f"Current min: {current_min}")
    print(f"Current max: {current_max}")
    print("   ----   ")
    print("Memory usage: ",)
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    print(" ----------------- finished ----------------- ")

def create_utterance_Hist_from_hdf5(hdf5_path,save_path):
    num_utterances = []
    with h5py.File(hdf5_path, "r") as f:
        for idx in range(len(f.keys())):
            num_utterances.append(f['speaker_'+str(idx)].attrs['num_utterances'].item())
    plt.figure()
    plt.title("Vox1: Number of utterances per speaker (500 Bins)")
    plt.hist(num_utterances, bins=500)
    plt.xlabel("Speaker Index")
    plt.ylabel("Number of Utterances")
    plt.savefig(os.path.join(save_path,"Vox1_utterances_per_speaker_hist(bin=500).pdf"))
    # plt.show()
    plt.figure()
    plt.title("Vox1: Number of utterances per speaker (50 Bins)")
    plt.hist(num_utterances, bins=50)
    plt.xlabel("Speaker Index")
    plt.ylabel("Number of Utterances")
    plt.savefig(os.path.join(save_path,"Vox1_utterances_per_speaker_hist(bin=50).pdf"))
    # plt.show()
    # print(num_utterances)

# iterate_hdf5("/mnt/ssd2/Tassi/TassiMA/Voxceleb_data/SpeakerDataVox2.hdf5","/mnt/ssd2/Tassi/TassiMA/Figures")

# saveAudioLengths("/mnt/ssd2/Tassi/TassiMA/Figures/Vox1_audio_lengths.npy")

if __name__ == "__main__":
    # Audio = getAudioVox1()
    # save_to_hdf5(Audio,"/mnt/ssd2/Tassi/TassiMA/Voxceleb_data/SpeakerDataVox1.hdf5")
    # save_to_hdf5(Audio,"/mnt/Games/Tassia_Daten/SpeakerDataVox1.hdf5")
    # saveAudioLengths("/mnt/ssd2/Tassi/TassiMA/Figures/Vox1_audio_lengths.npy")


    Audio = getAudioVox2_test()
    save_to_hdf5(Audio,"/mnt/ssd2/Tassi/TassiMA/Voxceleb_data/SpeakerDataVox2_test.hdf5")

    # create_utterance_Hist_from_hdf5("/mnt/ssd2/Tassi/TassiMA/Voxceleb_data/SpeakerDataVox1.hdf5","/mnt/ssd2/Tassi/TassiMA/Figures")
# iterate_hdf5("/mnt/ssd2/Tassi/TassiMA/Voxceleb_data/SpeakerDataVox1.hdf5")



    
# class train_dataset_sampler(torch.utils.data.Sampler):
#     def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

#         self.data_label         = data_source.data_label 
#         self.nPerSpeaker        = nPerSpeaker + 1
#         self.max_seg_per_spk    = max_seg_per_spk 
#         self.batch_size         = batch_size 
#         self.epoch              = 0 
#         self.seed               = seed 
#         self.distributed        = distributed 
        
    # def __iter__(self):

    #     start_time = time.time()

    #     indices = list(range(len(self.data_label)))
    #     data_dict = {}

    #     # Sort into dictionary of file indices for each ID
    #     for index in indices:
    #         speaker_label = self.data_label[index]
    #         if not (speaker_label in data_dict):
    #             data_dict[speaker_label] = [] 
    #         data_dict[speaker_label].append(index) 


    #     ## Group file indices for each class
    #     dictkeys = list(data_dict.keys()) 
    #     dictkeys.sort()

    #     lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

    #     flattened_list = []
    #     flattened_label = []
        
    #     for findex, key in enumerate(dictkeys):
    #         data    = data_dict[key]
    #         numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            
    #         rp      = lol(np.arange(numSeg),self.nPerSpeaker)
    #         flattened_label.extend([findex] * (len(rp)))
    #         for indices in rp:
    #             flattened_list.append([data[i] for i in indices])

    #     ## Mix data in random order
    #     mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
    #     mixlabel        = []
    #     mixmap          = []

    #     ## Prevent two pairs of the same speaker in the same batch
    #     for ii in mixid:
    #         startbatch = round_down(len(mixlabel), self.batch_size)
    #         if flattened_label[ii] not in mixlabel[startbatch:]:
    #             mixlabel.append(flattened_label[ii])
    #             mixmap.append(ii)

    #     mixed_list = [flattened_list[i] for i in mixmap]

    #     end_time = time.time()
    #     print("Data loading time: ", end_time - start_time)

    #     ## Divide data to each GPU
    #     if self.distributed:
    #         total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
    #         start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
    #         end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
    #         self.num_samples = end_index - start_index
    #         return iter(mixed_list[start_index:end_index])
    #     else:
    #         total_size = round_down(len(mixed_list), self.batch_size)
    #         self.num_samples = total_size
    #         return iter(mixed_list[:total_size])

    
    # def __len__(self) -> int:
    #     return self.num_samples

    # def set_epoch(self, epoch: int) -> None:
    #     self.epoch = epoch


