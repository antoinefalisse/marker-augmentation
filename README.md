# CS-230: Human body marker augmentation

### Install requirements (see docker below too)
1. Install [Anaconda](https://www.anaconda.com/)
2. Clone the repository to your machine.
3. Open the Anaconda command prompt and create a conda environment: `conda create -n cs-230-project`
4. Activate the environment: `conda activate cs-230-project`
5. Install [tensorflow with GPU support](https://www.tensorflow.org/install/pip) (you can probably also use CPU only if you train on a small part of the dataset).
6. Install other dependencies. Navigate to the local directory where the repository is cloned, then: `python -m pip install -r requirements.txt`

### Dataset
- A subset of the dataset is available [here](https://drive.google.com/file/d/1zstU911Jc9_Y692pjhk8smBwRnOh5hr1/view?usp=sharing). Download it into augmenter-cs230/Data. The path of the first feature time sequence should be something like /augmenter-cs230/Data/data_CS230/feature_0.npy.
- The full dataset is available [here](https://drive.google.com/file/d/1oUlPlV58onFt6VGCOp01cfv6j-wfu60z/view?usp=sharing). Download it into augmenter-cs230/Data. The path of the first feature time sequence should be something like /augmenter-cs230/Data/data_CS230/feature_0.npy.

### Overview files
**Main files**:
- `trainLSTM.py`: script to train LSTM model.
- `evaluateLSTM.py`: script to evaluate LSTM model.
- `testLSTM.py`: script to test LSTM model on failure example.
- `tuneHyperParametersLSTM.py`: script to tune hyperparameters of LSTM model.
- `myModels.py`: script describing the model architecture.
- `myModelsHyperParameters.py`: script describing the model architecture with support for hyperparameters tuning.
- `myDataGenerator.py`: script describing the data generator.
- `mySettings.py`: script with some tunable model settings.

**Other files (you should not need to interact with these files)**:
- `splitData.py`: script to split the data into different sets.
- `infoDatasets.py`: some details about the dataset.
- `utilities.py`: various utilities.

### Docker
- You can also use docker. The following instructions worked for me on a machine where cuda and cudnn were installed, you should adjust the paths and names according to your working environment.
- Build image: eg `docker build -t antoine/tensorflow:latest-gpu-0.1 .`
- Train model: eg `sudo docker run --gpus all -it --rm -v /home/clarkadmin/Documents/MyRepositories/augmenter-cs230:/augmenter-cs230 antoine/tensorflow:latest-gpu-0.1 python augmenter-cs230/trainLSTM.py`
- Evaluate model: eg `sudo docker run --gpus all -it --rm -v /home/clarkadmin/Documents/MyRepositories/augmenter-cs230:/augmenter-cs230 antoine/tensorflow:latest-gpu-0.1 python augmenter-cs230/evaluateLSTM.py`
- Test model: eg `sudo docker run --gpus all -it --rm -v /home/clarkadmin/Documents/MyRepositories/augmenter-cs230:/augmenter-cs230 antoine/tensorflow:latest-gpu-0.1 python augmenter-cs230/testLSTM.py`
- Tune hyperparameters: eg `sudo docker run --gpus all -it --rm -v /home/clarkadmin/Documents/MyRepositories/augmenter-cs230:/augmenter-cs230 antoine/tensorflow:latest-gpu-0.1 python augmenter-cs230/tuneHyperParametersLSTM.py`
- The files might be locked after training, to unlock run `sudo chown -R $USER: $HOME`
