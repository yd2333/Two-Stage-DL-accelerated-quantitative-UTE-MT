Author: Yidan Wang

Short introduction: Current pipeline is only tested on OA data and Brain data. 

How to run knee_acq_final.ipynb:
1. For testing: directly run the jupyter notebook and change configure file accordingly. No need to edit config file. Config stage1 and stage2 info in config file. Only run correlation analysis and exporting code block after stage2
2. For training: Write a new configure file. Use existing config as template. Must edit model name so that old weights file are not overwritten.

Structure of this folder:
├───checkpoints
│   └───knee_acq
├───configs
│   └───.ipynb_checkpoints
├───data
│   └───knee
└───mt_result
    ├───OA_stage1.pthtest_display_original_loss
    │   ├───1
    │   ├───2
    │   ├───3
    │   └───4
    ├───OA_stage1_T1=02.pth
    └───OA_stage2_unet.pthtest_display_original_loss
        ├───1
        ├───2
        ├───3
        └───4

Folders walk thru:
./checkpoints: store model log and pth file according to the project
./configs: Configuration of model and datatype, input and output configuration. This is where you want to edit for most of the time if not debugging for a model.
./data: .m and .mat files used for generating data
./mt_result: visualizations for different models

Files:
./knee_acq_final.ipynb: jupyter notebook as the entry to run the code
./model.py: where the actual model is stored
./preprocessing.py: preprocessing code, including loading the mat file, concatenating MT and T1 code, train and test data selection
./utils.py: visualization code

Environment setup
Change environment name (1st line), environment folder (last line) accordingly. Then run the cmd below.
cmd: conda env create -f environment.yml
