# A3GC-IP
Attention-Oriented Adjacency Adaptive Recurrent Graph Convolutions for Human Pose Estimation from Sparse Inertial Measurements

The Code included in articulate is adapted from Transpose, with the original Github Repo here: https://github.com/Xinyu-Yi/TransPose
Here we describe the process to reproduce the results and retrain the models.

#### 1. Get the SMPL model and the data:

**1.1.** Download SMPL model from https://smpl.is.tue.mpg.de/ (we use version 1.1.0), and place it in the models folder.  

**1.2.** Prepare th test data:  
- Download the DIP-IMU data from https://dip.is.tue.mpg.de/ and place the unzipped files in data/raw_datasets/DIP-IMU  
- For Total Capture the data including ground truth is obtained from the DIP authors. Place them in data/raw_datasets/TotalCapture/DIP_recalculate  
- If you want to retrain the models, you need to obtain the AMASS data from https://amass.is.tue.mpg.de/ . (SMPL+HG versions were used, place them in data/raw_datasets/AMASS)  

#### 2. Prepare the data:

**2.1.** Depending on which tests/training you want to run, uncomment the corresonding lines in preprocess.py. The script is adapted again from Transpose with changes made to allow for a fair split of the AMASS data not including Total Capture, as well as for doing the contralateral data augmentation for the training/validation data there (We do it there as it imposes enough computations to slow down the training when done in the data loader).  

**2.2.** Postprocess the data generated with above script (later this will be one script alltogether.), by again uncommenting the corresponding lines in postprocess_data.py  

#### 3. Evaluation:

for evaluation, pick the evaluation script for the model you want to test, e.g. evaluate_a3gc.py  
(The evaluation from the original DIP is excluded from here, as it was trained using the provided code of the authors.)  

    python3 evaluate_a3gc_tp.py --path trained_models/A3GC --type finetuning --hu1 256 --hu2 64 --hu3 128 --norm --cda  
    python3 evaluate_tp.py --path trained_models/Transpose --type finetuning --rotsize 9 --norm  
    python3 evaluate_dip.py --path trained_models/DIP_global/checkpoint_finetuning_13.tar --type finetuning --rotsize 9 --hu 512 --norm  
    python3 evaluate_ggru_tp.py --path trained_models/G-GRU --type finetuning --hu1 256 --hu2 64 --hu3 128 --norm  

#### 4. Training:

For training, pick the training script for the model you want to train, e.g. train_a3gc.py  

    python3 train_a3gc_tp.py -b 16 --name A3GC_ --lr 1e-3 --hu1 256 --hu2 64 --hu3 128 --norm --cda --resume trained_models/run_62 --start_at 3 # pretrain  
    python3 train_a3gc_tp.py -b 16 --name A3GC_fine --lr 1e-4 --hu1 256 --hu2 64 --hu3 128 --norm --cda -f --resume trained_models/run_62 # fine  



**5.** The selection of best and worst frames as done in the paper can be done with get_worst_frames.py, but you need to change the paths for saved predictions. you can save predictions by calling evaluate_a3gc.py or evaluate_tp.py with --save_predictions 

