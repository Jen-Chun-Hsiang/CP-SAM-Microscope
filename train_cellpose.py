import numpy as np
import logging
from datetime import datetime
from cellpose import models, core, io, plot, train, utils, metrics
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted
import torch
from utils.plot_utils import plot_losses
from utils.file_utils import cleanup_multiple_dirs



def main():
    logging_save_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Logs'
    train_seg_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Train_Segmentations'
    test_seg_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Test_Segmentations'
    model_save_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Models'
    test_result_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/Test_Results'
    # Make sure the logging directory exists (create parents if needed)
    Path(logging_save_dir).mkdir(parents=True, exist_ok=True)
    Path(train_seg_dir).mkdir(parents=True, exist_ok=True)
    Path(test_seg_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True) 
    Path(test_result_dir).mkdir(parents=True, exist_ok=True) 
    # Let cellpose set up logging and write the logfile into our desired dir
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logger, logfile = io.logger_setup(cp_path=logging_save_dir, logfile_name=f"train_cellpose_{timestamp}.log")

    # Clean up temporary files in train and test directories
    logger.info("Cleaning up temporary files...")
    cleanup_results = cleanup_multiple_dirs([train_seg_dir, test_seg_dir], logger=logger)
    logger.info(f"Cleanup completed: {sum(cleanup_results.values())} total files removed")

    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    gpu_ok = torch.cuda.is_available()  # or core.use_gpu() if preferred
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
    exp_tag = f"{model_name}_{timestamp}"
    n_epochs = 200
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
    
    # Plot and save losses
    plot_path = plot_losses(train_losses, test_losses, test_result_dir, exp_tag)
    logger.info(f"Loss plot saved to {plot_path}")
    
    # Generate predictions on test data using the trained model
    logger.info("Generating mask predictions on test data...")
    trained_model = models.CellposeModel(gpu=gpu_ok, pretrained_model=new_model_path)
    
    # Create directory for prediction outputs
    prediction_dir = Path(test_result_dir) / f"predictions_{exp_tag}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    
    # Run predictions on test data (following the tutorial pattern)
    masks = trained_model.eval(test_data, batch_size=32)[0]
    

    # Save individual masks and visualizations
    for idx in range(len(test_data)):
        logger.info(f"Saving results for test image {idx+1}/{len(test_data)}")
        
        # Save mask as numpy array
        mask_save_path = prediction_dir / f"test_mask_{idx:03d}.npy"
        np.save(mask_save_path, masks[idx])
        
        # Save visualization
        fig = plt.figure(figsize=(12, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(test_data[idx], cmap='gray')
        plt.title(f'Original Image {idx}')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(test_labels[idx], cmap='tab20')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(masks[idx], cmap='tab20')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.tight_layout()
        vis_save_path = prediction_dir / f"test_comparison_{idx:03d}.png"
        plt.savefig(vis_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved mask to {mask_save_path}")
        logger.info(f"Saved visualization to {vis_save_path}")
    
    # Save all metrics
    metrics_save_path = prediction_dir / f"metrics_{exp_tag}.npy"
    np.save(metrics_save_path, {'ap': ap, 'masks': masks})
    logger.info(f"Metrics saved to {metrics_save_path}")
    logger.info(f"All predictions saved to {prediction_dir}")
    

if __name__ == '__main__':
    main()