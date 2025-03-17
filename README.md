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


### Build the Docker image

To build for the TEMA cloud.
`docker build --platform linux/amd64 -t expalantion_tfa02 .`

For development on a mac build like this:
`docker build -t explanation_tfa02 .`




### Push the image to the registry

- Obtain Personal Access Token (PAT) by following [these instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

    It is likely that you already have a PAT stored in your git credential helper.
    To see it, do
    ``` bash
    echo "url=https://github.com" | git credential fill
    ```

- Set the PAT environment variable to your PAT value:
     ```bash
     export PAT=<your_git_personal_access_token>
     ```

- Test GHCR login:
     ```bash
     echo $PAT | docker login ghcr.io -u <your_git_username> --password-stdin
     ```

 - Push the TFA-02 container image to the Container registry:
     ```bash
     docker tag explanation_tfa02 ghcr.io/he-tema/explanation_tfa02:1.0
     docker push ghcr.io/he-tema/explanation_tfa02:1.0
     ```
- Write to Nicola Colosi on tema slack to deploy the pushed container on the TEMA cluster.

### Run the docker container

Edit the file if you want to change some environment variables
```bash
./run_docker.sh
```



