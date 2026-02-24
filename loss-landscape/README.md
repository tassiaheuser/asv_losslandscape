# Evaluating ResNet-based Speaker Verification Systems through Loss Landscape Visualization
## Loss Landscape


This repository contains the PyTorch code to generate loss landscapes for the master thesis "Evaluating ResNet-based Speaker Verification Systems through Loss Landscape Visualization"
> Tassia Heuser

The code is based on the github repository [https://github.com/tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape) which implements Li et al. paper.
> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.

To generate loss landscapes a network architecture and its pre-trained parameters are needed.
To train a new Voxceleb model use the Voxceleb repository
If you need more guidance on the basics on how to utilize this repository, check out tomgoldstein which has a more in depth documentation. This repository maintains the baseline functionallity of tomgoldstein but adds some more options relevant to the master thesis.

## Strucutre
```   
├── configs             stores configuration files that are \
│                       passed to plot_surface.py as argument for \
│                       --config instead of typing out all needed parameters \
│
├── voxceleb
│   ├── loss/           the python modules with loss functions
│   ├── model/          the model python definitions
│   ├── trained_net/    holds parameter files for trained voxceleb networks
│   ├── dataloader      old voxceleb dataloader and a new dataloader based on h5py, which is 30x faster
│   └── model_loader    loads the right model files from model/
├── utils.py            utilities like the config file loader
├── plot_surface.py     the main script to generate loss landscapes, parses command line arguments
.
.
.
└── README.md
```

# Main features added

* Increase the reesolution of a .h5 surface file and calculate the loss landscape for a larger range of parameters while keeping the old values
* Add a new dataloader based on h5py which is 30x faster than the old dataloader
* Extend a suface files dimension, e.g. from -1 to 1 -> -1.5 to 1.5
