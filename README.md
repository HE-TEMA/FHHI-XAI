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






