import numpy as np
import logging
from datetime import datetime
from cellpose import models, core, io, plot, train, utils
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted
import torch



def main():
    logging_save_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Logs'
    train_seg_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Train_Segmentations'
    test_seg_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Test_Segmentations'
    model_save_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Models'
    # Make sure the logging directory exists (create parents if needed)
    Path(logging_save_dir).mkdir(parents=True, exist_ok=True)
    Path(train_seg_dir).mkdir(parents=True, exist_ok=True)
    Path(test_seg_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True) 
    # Let cellpose set up logging and write the logfile into our desired dir
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logger, logfile = io.logger_setup(cp_path=logging_save_dir, logfile_name=f"train_cellpose_{timestamp}.log")

    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    gpu_ok = models.use_gpu()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Cellpose GPU enabled: {gpu_ok}")

    model = models.CellposeModel(gpu=gpu_ok)
    logger.info('Cellpose model created')


    masks_ext = "_seg.npy"
    output = io.load_train_test_data(train_seg_dir, test_seg_dir, mask_filter=masks_ext)
    train_data, train_labels, _, test_data, test_labels, _ = output
    # (not passing test data into function to speed up training)

    model_name = "cellpose_rgc_SPP1_101325"
    n_epochs = 100
    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 1

    new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                                train_data=train_data,
                                                                train_labels=train_labels,
                                                                batch_size=batch_size,
                                                                n_epochs=n_epochs,
                                                                learning_rate=learning_rate,
                                                                weight_decay=weight_decay,
                                                                nimg_per_epoch=max(2, len(train_data)), # can change this
                                                                model_name=model_name,
                                                                save_path=model_save_dir )
    
    # Save the model
    logger.info(f"Model saved to {new_model_path}")
    

if __name__ == '__main__':
    main()