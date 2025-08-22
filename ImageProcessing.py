#!/usr/bin/env python3
"""
CellPose-SAM Image Processing Script

This script processes PNG images using CellPose-SAM for cell segmentation,
saves processed images, and exports quantified measurements.

Requirements:
- cellpose
- numpy
- matplotlib
- skimage
- opencv-python
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from skimage import measure, filters
from skimage.color import label2rgb
import json

try:
    from cellpose import models, io, plot
    print("CellPose imported successfully")
except ImportError:
    print("Error: CellPose not installed. Install with: pip install cellpose")
    exit(1)

class CellPoseProcessor:
    def __init__(self, input_folder, image_save_folder, result_save_folder, 
                 model_type='cyto2', channels=[0,0], diameter=30, 
                 flow_threshold=0.4, cellprob_threshold=-2.0, 
                 min_size=15, normalize=True,
                 filename_contains=None, image_type='png', do_3D=False):
        """
        Initialize the CellPose processor optimized for partial membrane labeling
        
        Args:
            input_folder (str): Path to folder containing PNG images
            image_save_folder (str): Path to save processed images
            result_save_folder (str): Path to save NPZ results
            model_type (str): CellPose model type - 'cyto2' best for membrane staining
            channels (list): Channel configuration [cytoplasm, nucleus]
            diameter (float): Expected cell diameter in pixels (estimate neuron size)
            flow_threshold (float): Flow error threshold (lower = more permissive)
            cellprob_threshold (float): Cell probability threshold (lower = more permissive)
            min_size (int): Minimum cell size in pixels
            normalize (bool): Normalize image intensities
            filename_contains (str|None): Optional substring to filter input filenames. None = no filter
            image_type (str): Image extension/type to load (e.g., 'png', 'tif'). Default: 'png'
            do_3D (bool): Enable 3D processing for Z-stacks (set True for volumetric TIF stacks)
        """
        self.input_folder = Path(input_folder)
        self.image_save_folder = Path(image_save_folder)
        self.result_save_folder = Path(result_save_folder)
        self.filename_contains = (filename_contains or None)
        self.image_type = (image_type or 'png')
        
        # Create output directories if they don't exist
        self.image_save_folder.mkdir(parents=True, exist_ok=True)
        self.result_save_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize CellPose model
        print(f"Initializing CellPose model: {model_type}")
        self.model = models.Cellpose(model_type=model_type)
        self.channels = channels
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.normalize = normalize
    self.do_3D = bool(do_3D)
        
        # Results storage
        self.processing_results = []
    
    def find_png_files(self):
        """Find input images by type with optional filename substring filter.

        Honors self.image_type (extension) and self.filename_contains.
        Default behavior: all PNG files when no substring is provided.
        """
        ext = (self.image_type or 'png').lstrip('.')
        patterns = [f"*.{ext.lower()}", f"*.{ext.upper()}"]

        files = []
        for pat in patterns:
            files.extend(self.input_folder.glob(pat))

        # Optional substring filter (case-insensitive)
        if self.filename_contains:
            needle = str(self.filename_contains).lower()
            files = [p for p in files if needle in p.name.lower()]

        files = sorted(set(files))
        filt_desc = f" type=.{ext}" + (f", contains='{self.filename_contains}'" if self.filename_contains else "")
        print(f"Found {len(files)} file(s) in '{self.input_folder}' matching{filt_desc}")
        return files
    
    def preprocess_image(self, image):
        """
        Preprocess image (2D or 3D) for better membrane detection with varying backgrounds
        
        Args:
            image (np.array): Input image
            
        Returns:
            np.array: Preprocessed image
        """
        def _enhance_2d(img2d):
            # Convert to grayscale if needed
            if img2d.ndim == 3:
                if img2d.shape[2] == 3:
                    gray2d = 0.299 * img2d[:,:,0] + 0.587 * img2d[:,:,1] + 0.114 * img2d[:,:,2]
                else:
                    gray2d = img2d[:,:,0]
            else:
                gray2d = img2d.copy()

            # Normalize to 0-255 range
            ptp = gray2d.max() - gray2d.min()
            if ptp == 0:
                gray2d = np.zeros_like(gray2d, dtype=np.uint8)
            else:
                gray2d = ((gray2d - gray2d.min()) / ptp * 255).astype(np.uint8)

            # CLAHE + light blur
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced2d = clahe.apply(gray2d)
            enhanced2d = cv2.GaussianBlur(enhanced2d, (3, 3), 0.5)
            return enhanced2d

        # 3D volume: process slice-by-slice
        if self.do_3D and image.ndim >= 3:
            # Handle (Z,Y,X) or (Z,Y,X,C)
            if image.ndim == 3 or (image.ndim == 4 and image.shape[-1] not in (3,4)):
                # Assume (Z,Y,X)
                enhanced_stack = np.stack([_enhance_2d(image[z]) for z in range(image.shape[0])], axis=0)
            else:
                # Assume (Z,Y,X,C)
                enhanced_stack = np.stack([_enhance_2d(image[z]) for z in range(image.shape[0])], axis=0)

            if self.channels == [0, 0]:
                return enhanced_stack  # (Z,Y,X)
            else:
                # Expand to pseudo-RGB per slice: (Z,Y,X,3)
                return np.stack([enhanced_stack, enhanced_stack, enhanced_stack], axis=-1)

        # 2D image path (original behavior)
        enhanced = _enhance_2d(image)
        if self.channels == [0, 0]:
            return enhanced
        else:
            return np.stack([enhanced, enhanced, enhanced], axis=-1)
    
    def load_image(self, image_path):
        """Load and prepare image for CellPose processing"""
        try:
            # Load image
            img = io.imread(str(image_path))
            
            # Preprocess for membrane detection
            preprocessed = self.preprocess_image(img)
            
            return img, preprocessed  # Return both original and preprocessed
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
    
    def segment_image(self, image):
        """
        Perform cell segmentation using CellPose with optimized parameters for membrane staining
        
        Args:
            image (np.array): Input image (preprocessed)
            
        Returns:
            tuple: (masks, flows, styles, diameters)
        """
        try:
            print(f"  Using parameters:")
            print(f"    Diameter: {self.diameter}")
            print(f"    Flow threshold: {self.flow_threshold}")
            print(f"    Cell probability threshold: {self.cellprob_threshold}")
            print(f"    Min size: {self.min_size}")
            
            masks, flows, styles, diameters = self.model.eval(
                image, 
                diameter=self.diameter,
                channels=self.channels,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                min_size=self.min_size,
                normalize=self.normalize,
                do_3D=self.do_3D,  # Enable 3D when requested
                net_avg=True,  # Average networks for better results
                augment=True,  # Use test-time augmentation
                tile=True,     # Use tiling for large images
                tile_overlap=0.1
            )
            
            print(f"  Detected {len(np.unique(masks))-1} potential cells before filtering")
            
            # Additional post-processing for partial membrane labeling
            masks = self.post_process_masks(masks, image)
            
            return masks, flows, styles, diameters
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None, None, None, None
    
    def post_process_masks(self, masks, image):
        """
        Post-process masks to handle partial membrane labeling
        
        Args:
            masks (np.array): Initial segmentation masks
            image (np.array): Original image
            
        Returns:
            np.array: Refined masks
        """
        if masks is None:
            return masks
            
        # Remove very small or very large objects
        props = measure.regionprops(masks)
        filtered_mask = np.zeros_like(masks)
        kept_labels = []
        
        # Calculate area statistics for filtering
        areas = [prop.area for prop in props]
        if areas:
            median_area = np.median(areas)
            min_area_threshold = max(self.min_size, median_area * 0.1)
            max_area_threshold = median_area * 10  # Remove very large objects
            
            for prop in props:
                # Keep objects within reasonable size range
                if min_area_threshold <= prop.area <= max_area_threshold:
                    # Additional shape filtering for partial membranes
                    # Keep objects that are not too circular (partial membranes should be elongated)
                    if prop.eccentricity > 0.3 or prop.area > median_area * 0.5:
                        filtered_mask[masks == prop.label] = len(kept_labels) + 1
                        kept_labels.append(prop.label)
            
            print(f"  Kept {len(kept_labels)} cells after post-processing")
        
        return filtered_mask
    
    def calculate_measurements(self, image, masks):
        """
        Calculate quantitative measurements from segmentation masks
        
        Args:
            image (np.array): Original image
            masks (np.array): Segmentation masks
            
        Returns:
            dict: Dictionary containing measurements
        """
        # Build an intensity image matching masks shape when possible
        intensity_image = None
        if masks is not None:
            if masks.ndim == 2:
                if image.ndim == 2:
                    intensity_image = image
                elif image.ndim == 3 and image.shape[2] >= 1:
                    intensity_image = image[:, :, 0]
            elif masks.ndim == 3:
                if image.ndim == 3 and image.shape == masks.shape:
                    intensity_image = image
                elif image.ndim == 4 and image.shape[:3] == masks.shape:
                    intensity_image = image[..., 0]

        # Get region properties
        props = measure.regionprops(masks, intensity_image=intensity_image)
        
        measurements = {
            'cell_count': len(props),
            'cell_ids': [],
            'areas': [],
            'perimeters': [],
            'centroids': [],
            'mean_intensities': [],
            'max_intensities': [],
            'min_intensities': [],
            'eccentricities': [],
            'solidity': [],
            'aspect_ratios': []
        }
        
        for prop in props:
            measurements['cell_ids'].append(prop.label)
            measurements['areas'].append(prop.area)
            # Perimeter may be undefined in 3D
            perim = getattr(prop, 'perimeter', np.nan)
            measurements['perimeters'].append(perim)
            measurements['centroids'].append(prop.centroid)
            # Intensity metrics are only valid if intensity_image is provided
            if intensity_image is not None:
                measurements['mean_intensities'].append(getattr(prop, 'mean_intensity', np.nan))
                measurements['max_intensities'].append(getattr(prop, 'max_intensity', np.nan))
                measurements['min_intensities'].append(getattr(prop, 'min_intensity', np.nan))
            else:
                measurements['mean_intensities'].append(np.nan)
                measurements['max_intensities'].append(np.nan)
                measurements['min_intensities'].append(np.nan)
            measurements['eccentricities'].append(getattr(prop, 'eccentricity', np.nan))
            measurements['solidity'].append(getattr(prop, 'solidity', np.nan))
            
            # Calculate aspect ratio
            maj = getattr(prop, 'major_axis_length', None)
            minr = getattr(prop, 'minor_axis_length', None)
            if maj is not None and minr is not None and minr > 0:
                aspect_ratio = maj / minr
            else:
                aspect_ratio = np.nan
            measurements['aspect_ratios'].append(aspect_ratio)
        
        return measurements
    
    def create_visualization(self, original_image, preprocessed_image, masks, flows):
        """
        Create comprehensive visualization of segmentation results for membrane labeling
        
        Args:
            original_image (np.array): Original image
            preprocessed_image (np.array): Preprocessed image
            masks (np.array): Segmentation masks
            flows (list): Flow outputs from CellPose
            
        Returns:
            np.array: Visualization image
        """
        # Create figure with more subplots for membrane labeling analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Prepare 2D views: for 3D, use maximum intensity projection (MIP)
        def _mip(img):
            if img is None:
                return None
            if img.ndim == 2:
                return img
            if img.ndim == 3:
                return np.max(img, axis=0)
            if img.ndim == 4:  # (Z,Y,X,C)
                return np.max(img, axis=0)
            return img

        orig_view = _mip(original_image)
        pre_view = _mip(preprocessed_image)

        # Original image
        if orig_view is not None:
            if orig_view.ndim == 3:
                axes[0,0].imshow(orig_view)
            else:
                axes[0,0].imshow(orig_view, cmap='gray')
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Preprocessed image
        if pre_view is not None:
            if pre_view.ndim == 3:
                axes[0,1].imshow(pre_view[:,:,0], cmap='gray')
            else:
                axes[0,1].imshow(pre_view, cmap='gray')
        axes[0,1].set_title('Preprocessed (Enhanced)')
        axes[0,1].axis('off')
        
        # Segmentation masks
        if masks is not None and masks.max() > 0:
            masks_view = masks
            if masks.ndim == 3:
                masks_view = np.max(masks, axis=0)  # 2D projection of labels
            # Color each cell differently
            axes[0,2].imshow(masks_view, cmap='tab20', vmax=20)
            axes[0,2].set_title(f'Segmentation Masks\n({len(np.unique(masks))-1} cells detected)')
            axes[0,2].axis('off')
            
            # Overlay on original
            if orig_view is not None:
                if orig_view.ndim == 3:
                    base_img = orig_view[:,:,0] if orig_view.shape[2] >= 1 else np.mean(orig_view, axis=2)
                else:
                    base_img = orig_view
            
            overlay_img = label2rgb(masks_view, image=base_img, alpha=0.4, bg_label=0, colors=plt.cm.Set1(np.linspace(0, 1, 12)))
            axes[1,0].imshow(overlay_img)
            axes[1,0].set_title('Overlay on Original')
            axes[1,0].axis('off')
            
            # Contours only
            contour_img = base_img.copy()
            if len(contour_img.shape) == 2:
                contour_img = np.stack([contour_img, contour_img, contour_img], axis=-1)
            
            # Draw contours for each cell
            for cell_id in np.unique(masks_view)[1:]:  # Skip background (0)
                cell_mask = (masks_view == cell_id).astype(np.uint8)
                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_img, contours, -1, (255, 255, 0), 2)
                
                # Add cell number
                M = cv2.moments(cell_mask)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(contour_img, str(cell_id), (cx-10, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            axes[1,1].imshow(contour_img)
            axes[1,1].set_title('Cell Contours with IDs')
            axes[1,1].axis('off')
            
            # Cell size distribution
            props = measure.regionprops(masks_view)
            areas = [prop.area for prop in props]
            if areas:
                axes[1,2].hist(areas, bins=min(20, len(areas)), alpha=0.7, edgecolor='black')
                axes[1,2].set_xlabel('Cell Area (pixels)')
                axes[1,2].set_ylabel('Count')
                axes[1,2].set_title(f'Cell Size Distribution\nMean: {np.mean(areas):.1f} pixels')
                axes[1,2].grid(True, alpha=0.3)
            else:
                axes[1,2].text(0.5, 0.5, 'No cells detected', ha='center', va='center')
                axes[1,2].set_title('Cell Size Distribution')
        
        else:
            for i in range(3):
                for j in range(2):
                    if i == 0 and j == 0:
                        continue
                    if i == 0 and j == 1:
                        continue
                    axes[j,i].text(0.5, 0.5, 'No cells detected', ha='center', va='center')
                    axes[j,i].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image array
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return plot_img
    
    def save_results(self, image_name, image, masks, flows, measurements, visualization):
        """
        Save processing results to files
        
        Args:
            image_name (str): Name of the original image
            image (np.array): Original image
            masks (np.array): Segmentation masks
            flows (list): CellPose flows
            measurements (dict): Quantified measurements
            visualization (np.array): Visualization image
        """
        base_name = Path(image_name).stem
        
        # Save visualization image
        vis_path = self.image_save_folder / f"{base_name}_segmentation.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization: {vis_path}")
        
        # Save masks as image
        if masks is not None:
            mask_path = self.image_save_folder / f"{base_name}_masks.png"
            # Use max projection for 3D mask preview
            if masks.ndim == 3:
                mask_view = np.max(masks, axis=0)
            else:
                mask_view = masks
            mask_normalized = (mask_view * 255 / mask_view.max()).astype(np.uint8) if mask_view.max() > 0 else mask_view.astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_normalized)
            print(f"Saved mask preview: {mask_path}")
        
        # Save NPZ file with all results
        npz_path = self.result_save_folder / f"{base_name}_results.npz"
        
        # Prepare data for NPZ
        save_data = {
            'original_image': image,
            'masks': masks if masks is not None else np.array([]),
            'measurements': measurements
        }
        
        # Add flows if available
        if flows is not None and len(flows) > 0:
            save_data['flows'] = flows[0] if flows[0] is not None else np.array([])
        
        # Convert lists to numpy arrays for measurements
        for key, value in measurements.items():
            if isinstance(value, list):
                measurements[key] = np.array(value)
        
        np.savez_compressed(str(npz_path), **save_data)
        print(f"Saved results: {npz_path}")
        
        # Also save measurements as JSON for easy reading
        json_path = self.result_save_folder / f"{base_name}_measurements.json"
        json_measurements = {}
        for key, value in measurements.items():
            if isinstance(value, np.ndarray):
                json_measurements[key] = value.tolist()
            else:
                json_measurements[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_measurements, f, indent=2)
        print(f"Saved measurements: {json_path}")
    
    def process_single_image(self, image_path):
        """
        Process a single image through the complete pipeline
        
        Args:
            image_path (Path): Path to the image file
        """
        print(f"\nProcessing: {image_path.name}")
        
        # Load image
        original_image, preprocessed_image = self.load_image(image_path)
        if original_image is None:
            return
        
        print(f"Image shape: {original_image.shape}")
        
        # Perform segmentation on preprocessed image
        masks, flows, styles, diameters = self.segment_image(preprocessed_image)
        
        if masks is not None:
            cell_count = len(np.unique(masks)) - 1
            print(f"Final count: {cell_count} cells detected")
            
            # Calculate measurements
            measurements = self.calculate_measurements(original_image, masks)
            
            # Create comprehensive visualization
            visualization = self.create_visualization(original_image, preprocessed_image, masks, flows)
            
            # Save results
            self.save_results(image_path.name, original_image, masks, flows, measurements, visualization)
            
            # Store summary results
            result_summary = {
                'filename': image_path.name,
                'cell_count': measurements['cell_count'],
                'mean_area': np.mean(measurements['areas']) if measurements['areas'] else 0,
                'mean_intensity': np.mean(measurements['mean_intensities']) if measurements['mean_intensities'] else 0,
                'detection_confidence': 'good' if cell_count > 0 else 'poor'
            }
            self.processing_results.append(result_summary)
            
        else:
            print("Segmentation failed")
            # Still create a visualization showing the failure
            visualization = self.create_visualization(original_image, preprocessed_image, None, None)
            measurements = {'cell_count': 0}
            self.save_results(image_path.name, original_image, None, None, measurements, visualization)
    
    def process_all_images(self):
        """Process all images in the input folder matching filters"""
        png_files = self.find_png_files()
        
        if not png_files:
            print("No images found in the input folder matching filters")
            return
        
        print(f"Starting processing of {len(png_files)} images...")
        
        for i, image_path in enumerate(png_files, 1):
            print(f"\n[{i}/{len(png_files)}] Processing {image_path.name}")
            try:
                self.process_single_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
        
        # Save summary results
        self.save_summary()
        print(f"\nProcessing complete! Results saved to:")
        print(f"  Images: {self.image_save_folder}")
        print(f"  Data: {self.result_save_folder}")
    
    def save_summary(self):
        """Save a summary of all processing results"""
        if not self.processing_results:
            return
        
        summary_path = self.result_save_folder / "processing_summary.json"
        
        # Calculate overall statistics
        total_cells = sum(r['cell_count'] for r in self.processing_results)
        successful_images = sum(1 for r in self.processing_results if r['cell_count'] > 0)
        
        summary = {
            'total_images_processed': len(self.processing_results),
            'successful_segmentations': successful_images,
            'total_cells_detected': total_cells,
            'average_cells_per_image': total_cells / len(self.processing_results) if self.processing_results else 0,
            'results_by_image': self.processing_results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved: {summary_path}")
        print(f"Total images: {len(self.processing_results)}")
        print(f"Successful segmentations: {successful_images}")
        print(f"Total cells detected: {total_cells}")

def main():
    """Main function to run the CellPose processing for membrane-labeled neurons"""
    # CLI arguments (with safe defaults matching previous behavior)
    parser = argparse.ArgumentParser(description="CellPose-SAM Image Processing")
    # Accept preferred name plus legacy aliases for input directory
    parser.add_argument("--input-image-folder", "--input", "-i", dest="input", default="/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/input_image",
                        help="Input image folder containing files to process")
    # Explicitly note these are output folders
    parser.add_argument("--out-images", "-oi", default="/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/output_images",
                        help="Output folder to save processed images")
    parser.add_argument("--out-results", "-or", default="/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/output_results", 
                        help="Output folder to save NPZ/JSON results")
    parser.add_argument("--type", "-t", default="png", help="Image type/extension to process (e.g., png, tif)")
    parser.add_argument("--contains", "-c", default=None, help="Optional substring filter for filenames")
    parser.add_argument("--three-d", "--3d", dest="do_3D", action="store_true", help="Enable 3D segmentation for Z-stacks (use with TIFF stacks)")

    args = parser.parse_args()

    # Configuration - can be overridden by CLI
    input_folder = args.input            # Folder containing images
    image_save_folder = args.out_images  # Folder to save processed images
    result_save_folder = args.out_results # Folder to save NPZ and JSON files
    image_type = args.type               # Image extension to search for
    filename_contains = args.contains    # Optional filename substring filter
    do_3D = bool(args.do_3D)
    
    # OPTIMIZED CellPose settings for partial membrane labeling
    model_type = "cyto2"      # Best for cytoplasmic/membrane staining
    channels = [0, 0]         # Grayscale processing (most antibody images are single channel)
    diameter = 30             # Estimate neuron diameter in pixels (ADJUST based on your images!)
    flow_threshold = 0.4      # Lower = more permissive (good for partial membranes)
    cellprob_threshold = -2.0 # Lower = more permissive (detects weaker signals)
    min_size = 15             # Minimum cell size in pixels
    normalize = True          # Normalize intensities (helps with varying backgrounds)
    
    print("=== CellPose Configuration for Membrane-Labeled Neurons ===")
    print(f"Model: {model_type} (optimized for cytoplasm/membrane)")
    print(f"Expected cell diameter: {diameter} pixels")
    print(f"Flow threshold: {flow_threshold} (lower = more permissive)")
    print(f"Cell probability threshold: {cellprob_threshold} (lower = more sensitive)")
    print(f"Minimum cell size: {min_size} pixels")
    print(f"Input folder: {input_folder}")
    print(f"Image type: .{image_type}")
    print(f"Filename contains: {filename_contains if filename_contains else 'None (all)'}")
    print(f"3D processing: {'ON' if do_3D else 'OFF'}")
    print("=" * 60)
    
    # Create processor and run
    processor = CellPoseProcessor(
        input_folder=input_folder,
        image_save_folder=image_save_folder,
        result_save_folder=result_save_folder,
        model_type=model_type,
        channels=channels,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        normalize=normalize,
        filename_contains=filename_contains,
    image_type=image_type,
    do_3D=do_3D
    )
    
    # Process all images
    processor.process_all_images()

if __name__ == "__main__":
    main()