import os
import time
import warnings
import argparse
import configparser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision # Not used directly, but models might.
from PIL import Image
from sklearn.metrics import f1_score # roc_auc_score, confusion_matrix not directly used
# from torch.nn import Softmax # Softmax not used explicitly
# from torchmetrics.classification import Dice # Not used
from torchmetrics.classification import BinaryAccuracy
# from torchmetrics import JaccardIndex # Not used
from torchvision import transforms
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Not used
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Not used

from utils import metrics # Import custom metrics

warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3 and DeepLabV3+ models.")
    parser.add_argument('--img_folder', type=str, default='/home/fisher/Peoples/hseung/Full/img',
                        help='Path to the main image folder.')
    parser.add_argument('--mask_folder', type=str, default='/home/fisher/Peoples/hseung/Full/mask',
                        help='Path to the main mask folder.')
    parser.add_argument('--model_paths_file', type=str, default='task1/model_paths.txt',
                        help='Path to the text file containing model paths.')
    parser.add_argument('--output_csv_file', type=str, 
                        default='/home/fisher/Peoples/hseung/NEW/deeplabv3plus_output_metric/deeplabplus_vs_deeplab_metric_new.csv',
                        help='Path for saving the output CSV metrics file.')
    parser.add_argument('--output_image_folder', type=str, 
                        default='/home/fisher/Peoples/hseung/NEW/segment outputs/deeplabv3plus_30_01/',
                        help='Path to save output images (if enabled).')
    parser.add_argument('--model_set_key', type=str, default='012', choices=['01', '012'], 
                        help="Model set key to use from the paths file (e.g., '01' or '012').")
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Device to load models on (e.g., "cuda:0", "cpu").')
    return parser.parse_args()

def load_model_paths_from_config(file_path):
    # Manual parsing based on the prompt's conceptual example for model_paths.txt
    paths_manual = {}
    current_section = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('(') and line.endswith(')'):
                    current_section = line[1:-1]
                    paths_manual[current_section] = {}
                elif current_section and ':' in line:
                    model_name_key, model_path_val = line.split(':', 1)
                    paths_manual[current_section][model_name_key.strip()] = model_path_val.strip().strip("'").strip('"')
    except FileNotFoundError:
        print(f"Error: Model paths file '{file_path}' not found.")
        return None
    return paths_manual

def main():
    args = parse_arguments()
    
    start_time = time.time()

    all_model_paths = load_model_paths_from_config(args.model_paths_file)
    if all_model_paths is None:
        return # Exit if model paths file not found

    model_section = all_model_paths.get(args.model_set_key) 
    if not model_section:
        print(f"Error: Model set key '{args.model_set_key}' not found in {args.model_paths_file}")
        print(f"Available sections: {list(all_model_paths.keys())}")
        return

    deeplab_path = model_section.get('deeplab')
    deeplabv3plus_path = model_section.get('deeplabv3plus')

    if not deeplab_path:
        print(f"Error: 'deeplab' path not found in section '{args.model_set_key}' of {args.model_paths_file}")
        return
    if not deeplabv3plus_path: 
        print(f"Error: 'deeplabv3plus' path not found in section '{args.model_set_key}' of {args.model_paths_file}")
        return
        
    device_to_load = torch.device(args.device)
    
    try:
        deeplab = torch.load(deeplab_path, map_location=device_to_load)
        deeplabv3plus = torch.load(deeplabv3plus_path, map_location=device_to_load)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}. Please check paths in {args.model_paths_file} for section '{args.model_set_key}'.")
        return
    except Exception as e: 
        print(f"Error loading model weights: {e}")
        return

    deeplabv3plus.eval()
    deeplab.eval()
    
    # Models are already on device_to_load due to map_location in torch.load
    # If not, they should be moved:
    # deeplab.to(device_to_load)
    # deeplabv3plus.to(device_to_load)


    try:
        all_img_files = [f for f in os.listdir(args.img_folder)]
        img_filenames = sorted([f for f in all_img_files if ('20220817' in f) or ('20220819' in f)])
    except FileNotFoundError:
        print(f"Error: Image folder '{args.img_folder}' not found.")
        return
    
    test_file = img_filenames

    metric_df = pd.DataFrame(columns=['File Name','Deep Lab Accuarcy', 'Deep Plus Accuarcy', 'Deep DIce', 'Plus Dice',
                                      'Deep Jaccard', 'Plus Jaccard', 'Deep f1', 'Plus F1'])
    
    for file_index in range(len(test_file)):
        metric_dict = {}
        img_filename = test_file[file_index]
        img_path = os.path.join(args.img_folder, img_filename)
        
        mask_filename_base = os.path.splitext(img_filename)[0]
        mask_filename = mask_filename_base + '.npy'

        mask_path = os.path.join(args.mask_folder, mask_filename)
        
        try:
            mask_npy = np.load(mask_path)
        except FileNotFoundError:
            print(f"Warning: Mask file '{mask_path}' not found for image '{img_filename}', skipping.")
            continue
        mask = mask_npy
        
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        img = Image.open(img_path).convert('RGB')
        image_array = np.array(img)

        for object_index in range(num_objs):
            # This inner loop processes each object within the image
            # Ensure models are on the correct device for each inference if they weren't moved before
            deeplab.to(device_to_load)
            deeplabv3plus.to(device_to_load)

            target_size = (img.size[1],img.size[0])
            rcnn_trial = np.zeros(target_size,dtype=int)
            rcnn_trial[boxes[object_index][1]:boxes[object_index][3],boxes[object_index][0]:boxes[object_index][2]]=1
            mask_trial = np.stack([rcnn_trial,rcnn_trial,rcnn_trial], axis=-1)
            masked_image = image_array * mask_trial
            masked_images = Image.fromarray(masked_image.astype('uint8'))
            
            # img_trans was defined but not used. Assuming newimg_trans is the correct tensor for model input
            # img_trans = transforms.ToTensor()(masked_images).unsqueeze(0) 

            coords = boxes[object_index]
            ground_truth_pil = Image.fromarray((masks[object_index]).astype(np.uint8))
            ground_mask_cropped = ground_truth_pil.crop(coords)

            new_image_cropped = masked_images.crop(coords)
            new_image_resized = new_image_cropped.resize((256,256))
            
            newimg_trans = transforms.ToTensor()(new_image_resized).unsqueeze(0).to(device_to_load)
            
            with torch.no_grad():
                raw_deeplab_output = deeplab(newimg_trans)
                raw_deeplabplus_output = deeplabv3plus(newimg_trans)
                
            deeplab_argmax_segmentation = np.argmax(np.squeeze(raw_deeplab_output['out']).cpu().numpy(), 0)
            raw_deeplabplus_segmentation = raw_deeplabplus_output.cpu().numpy() # Assuming this output is already suitable for direct use or needs argmax
            
            deeplab_segmentation_binary = (deeplab_argmax_segmentation==1)
            # Assuming deeplabplus_segmentation is single channel after model, and thresholding is applied.
            # If model outputs multiple channels for classes, an argmax might be needed before thresholding or specific channel selection.
            deeplabplus_threshold_segmentation = raw_deeplabplus_segmentation[0][0]>0.5 

            image_list_for_plot = [new_image_resized, ground_mask_cropped.resize((256,256)), deeplabplus_threshold_segmentation, deeplab_segmentation_binary]
            ground_truth_resized_for_metric = np.array(ground_mask_cropped.resize((256,256)))
            
            titles = ['Crop_Image','Crop_Ground','plus','1_channel']
            num_subplots = len(image_list_for_plot)

            fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5)) # Explicitly 1 row
            current_img_filename = test_file[file_index] 

            # Ensure axes is always iterable (for single subplot case)
            axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            for i, ax_idx in enumerate(axes_flat): 
                ax_idx.imshow(image_list_for_plot[i])
                ax_idx.set_title(titles[i])
                ax_idx.axis('off')
                ax_idx.set_adjustable('box')
            
            plt.subplots_adjust(top=0.8)

            if args.output_image_folder:
                if not os.path.exists(args.output_image_folder):
                    os.makedirs(args.output_image_folder)
                plt.savefig(os.path.join(args.output_image_folder, f'{current_img_filename}_{object_index}.png'))
            plt.close(fig) # Close the figure to free memory
              
            final_deeplab_prediction = np.where(deeplab_segmentation_binary ==True, 1 ,0)
            final_ground_truth = np.where(ground_truth_resized_for_metric>0, 1,0) # Use resized GT for metrics
            final_deeplabplus_prediction = np.where(deeplabplus_threshold_segmentation==True,1,0)
            
            accuracy_metric_instance = BinaryAccuracy().to(device_to_load) # Instance for potential use, though not directly in dict

            metric_dict = {
                'File Name': f'{current_img_filename}_{object_index}',
                'Deep Lab Accuarcy': metrics.accuracy(final_ground_truth, final_deeplab_prediction), 
                'Deep Plus Accuarcy': metrics.accuracy(final_ground_truth, final_deeplabplus_prediction), 
                'Deep DIce': metrics.dice_coefficient(final_ground_truth, final_deeplab_prediction), 
                'Plus Dice': metrics.dice_coefficient(final_ground_truth, final_deeplabplus_prediction), 
                'Deep Jaccard': metrics.iou_score(final_ground_truth,final_deeplab_prediction), 
                'Plus Jaccard': metrics.iou_score(final_ground_truth, final_deeplabplus_prediction), 
                'Deep f1': f1_score(final_ground_truth, final_deeplab_prediction, average='macro',zero_division=1), 
                'Plus F1': f1_score(final_ground_truth, final_deeplabplus_prediction, average='macro',zero_division=1) 
            }
            metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

    metric_df.to_csv(args.output_csv_file, index=False)
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f'Processing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

if __name__ == '__main__':
    main()