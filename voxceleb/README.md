# Evaluating ResNet-based Speaker Verification Systems through Loss Landscape Visualization
## VoxCeleb trainer

This repository contains the PyTorch code to train and evaluate ASV Neural Networks for the master thesis "Evaluating ResNet-based Speaker Verification Systems through Loss Landscape Visualization"
> Tassia Heuser

This code is based on the github repository [https://github.com/clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) which implements Chung et al. paper " _In defence of metric learning for speaker recognition_'[2] and '_Pushing the limits of raw waveform speaker recognition_ "[3]

To download the voxceleb dataset and the pretrained ResNetSE34V2 and ResNetSE34L models please follow the guide provided on the clovaai github.


## Strucutre
```   
├── configs                     stores configuration files that are \
│                               passed to trainSpeakerNet.py as argument for \
│                               --config instead of typing out all required command line arguments \
│
├── Create_h5_dataset
│   ├── create_new_Dataset.py   script to transform a directory containing .wav files from voxceleb to a hdf5 dataset
│   └── dataset_analysis.py     analyses performed on Voxceleb1 and 2
│
├── loss            
├── models                      the model python definitions
├── notebooks                   Notebook to generate t-SNE visualizations based on stored emmbedings from --eval
├── optimizer
├── scheduler
├── dataloader.py
├── SpeakerNet.py               implements training and evaluation loops
├──trainSpeakerNet.py           the main script to train and evaluate models, parses command line arguments
├── tuneThreshold.py            functions to calculate error rates and MinDcf
.
.
.
└── README.md
```


# Main features added

* Log training runs to weights and biases
* new efficient data loader
* calculate train set statistics like AuC scores an accuracy across the whole Voxceleb 2 dataset
* New Cosine and Constant LR-schedulers
* Seamles continuation of training by storing full optimizer and scheduler statistics



### Citation

Please cite [1] if you make use of Chungs code. Please see [here](References.md) for the full list of methods used in this trainer.

[1] _In defence of metric learning for speaker recognition_
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Proc. Interspeech},
  year={2020}
}
```

[2] _The ins and outs of speaker recognition: lessons from VoxSRC 2020_
```
@inproceedings{kwon2021ins,
  title={The ins and outs of speaker recognition: lessons from {VoxSRC} 2020},
  author={Kwon, Yoohwan and Heo, Hee Soo and Lee, Bong-Jin and Chung, Joon Son},
  booktitle={Proc. ICASSP},
  year={2021}
}
```

[3] _Pushing the limits of raw waveform speaker recognition_
```
@inproceedings{jung2022pushing,
  title={Pushing the limits of raw waveform speaker recognition},
  author={Jung, Jee-weon and Kim, You Jin and Heo, Hee-Soo and Lee, Bong-Jin and Kwon, Youngki and Chung, Joon Son},
  booktitle={Proc. Interspeech},
  year={2022}
}
```

### License
```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
