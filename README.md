# L-CRP-TEMA

This repository contains the code for applying the L-CRP method for TEMA project. 

## Setting up

### Load the data and models

Data and models are available on Google Drive https://drive.google.com/drive/folders/1vmkyJzojacZUc2rz-VBw5T07KzoFslzB?usp=sharing. 

Path for data is `datasets/data`.

#### 1. U-Net model: 
- Checkpoint: Checkpoints/unet_flood.pt
- Dataset: Data/General_Flood_v3.zip
- Task: specifically for this model - flood detection

#### 2. YOLOv6s6 model:
- Checkpoint: Checkpoints/best_v6s6_ckpt.pt
- Dataset: Data/PersonCarDetectionData 
- Task: person and car detection

#### 3. PIDNet model: not yet available










### Suggested repo structure:

```
L-CRP-TEMA/
│── datasets/
│   ├── data/                  # Store raw and preprocessed datasets, added to gitignore
│   │   ├── General_Flood_v3/  
│   │   ├── PersonCarDetectionData/
│   ├── flood_dataset.py      # Script for loading flood dataset
│
│── experiments/               # Stores experiment configurations & logs
│
│── examples/                  # Stores example notebooks
│   ├── unet_example/  
│
│── models/                    # Store model architectures & checkpoints
│   ├── checkpoints/           # Stores trained models' weights, added to gitignore
│   │   ├── unet_flood.pt      # Checkpoint for flood detection model
│
│── scripts/                   
│
│── utils/                     
│
│── config.yaml                # Global configuration file
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
│── .gitignore                 # Ignore unnecessary files
```