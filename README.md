# PCX-TEMA

This repository contains the code for applying the PCX method for TEMA project. 

## Setting up

### Load the data and models

Data and models are available on Google Drive https://drive.google.com/drive/folders/1vmkyJzojacZUc2rz-VBw5T07KzoFslzB?usp=sharing. 

Path for data is `datasets/data`.

#### 1.PIDNet model: 
- Checkpoint: Checkpoints/flood_model.pt
- Dataset: Data/flood_segmentation.zip
- Task: specifically for this model - flood segmentation

#### 2. YOLOv6s6 model:
- Checkpoint: Checkpoints/best_v6s6_ckpt.pt
- Dataset: Data/PersonCarDetectionData 
- Task: person and car detection



### Build the Docker image

To build for the TEMA cloud.
`docker build --platform linux/amd64 -t explanation_tfa02 .`

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

### Development

#### Install dependencies
```bash
conda create -n tema python=3.8
conda activate tema
pip install -r requirements.txt
```

During development, you can run the application components separately in different terminal tabs for easier debugging and log monitoring:

1. Start Redis server:
```bash
redis-server
```

2. Start the Flask application:
```bash
DEBUG=1 python app.py
```
DEBUG=1 will make the app reload on any code changes, remove if you don't want this.

3. Start the worker process:
```bash
python worker.py
```
For the worker to work properly on a Mac, before running it do 
```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

The application will be available at `http://localhost:8080/tfa02` by default.

Note: For production deployment, use the Docker container as described in the "Run the docker container" section above.

### Testing

You can test the application using the provided test script located in the `tests/` folder. 
```bash
cd tests
```

There are two ways to run the tests:
1. Local testing (sends notifications to local Redis):
```bash
python test_post_data.py ImageMetadata
```

2. Cloud testing (sends notifications to TEMA cloud):
```bash
python test_post_data.py ImageMetadata --cloud
```

The test script will send sample image metadata to the application and you should see the processing results in the logs of both the Flask application and the worker process.


### References


LRP (Layer-wise Relevance Propagation):

- original paper: (https://doi.org/10.1371/journal.pone.0130140)
- overview paper: (https://doi.org/10.1007/978-3-030-28954-6_10)
- zennit toolbox: (https://github.com/chr5tphr/zennit)
  
CRP (Concept Relevance Propagation):

- paper: (https://doi.org/10.1038/s42256-023-00711-8)
- zennit-crp toolbox: (https://github.com/rachtibat/zennit-crp)

  
L-CRP (Concept Relevance Propagation for Localization Models):

- paper: (https://arxiv.org/pdf/2211.11426)
- L-CRP code: (https://github.com/maxdreyer/L-CRP/tree/main)

  
PCX (Prototypical Concept-based Explanations):
- paper: (https://arxiv.org/pdf/2311.16681)
- PCX code: (https://github.com/maxdreyer/pcx/tree/main)


