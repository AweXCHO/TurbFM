PBCL (Physically boosted cooperative learning framework)
=============
## Description
This is the implementation of paper"Learning 2D strength fields of atmospheric turbulence hidden in infrared imaging". PBCL analyzes the infrared imaging results captured in atmospheric turbulence environment to measure the 2D strength fields of atmospheric turbulence and generate clear imaging signal simultaneously.

## System requirements
#### Prerequisites
* Ubuntu 18.04
* NVIDIA GPU + CUDA (Geforce RTX 3090 with 24GB memory, CUDA 11.1 was tested)

#### Installation
* Python 3.7+
* Pytorch 1.7.0+

## Demo
#### Dataset
A small version of our turbulence dataset is deposited in ```../data/small_turbulence_dataset/```, which includes three parts of data (for training, valid, and test). The training data and valid data are from our algorithm simulated data. In addition, we provide simulated data and real-world data in the test dir for quantitative and qualitative evaluation.

#### Test and evaluation
* Run ```python simu_data_test.py``` to test the trained model deposited in ```../data/experiment/model_name/``` on the algorithm simulated data in ```../data/small_turbulence_dataset/test/simulated_data/``` . The TS quantity fields and sequence results will be stored in ```../results/model_name/results_on_simulated_data/```.
* After the running of ```simu_data_test.py```, run ```python evaluate.py``` to obtain the quantitative evaluation of results.
* For real-world data, run ```python real_data_test.py``` to generate the results.

#### Training
* Run ```python train.py``` to perform training with the default setting on the training data in ```../data/small_turbulence_dataset/train/```.

## Liscense
* This project is covered under the BSD-3-Clause License.