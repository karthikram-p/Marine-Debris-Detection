import os
import sys
import random
import logging
import rasterio
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(up(os.path.abspath(__file__)))
from unet import UNet
from dataloader import GenDEBRIS, bands_mean, bands_std

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from metrics import Evaluation, confusion_matrix
from assets import labels

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(
    filename=os.path.join(root_path, 'logs', 'evaluating_unet.log'),
    filemode='a',
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logging.info('*' * 10)

def save_mask_as_npy(mask, output_dir, roi_name, epoch):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{roi_name}_epoch_{epoch}.npy"), mask)

def compare_masks(mask1, mask2):
    if mask1.shape != mask2.shape:
        return False, f"Shape mismatch: {mask1.shape} vs {mask2.shape}"
    
    differences = mask1 != mask2
    mismatches = np.sum(differences)
    return mismatches == 0, f"Total mismatched pixels: {mismatches}"

def main(options):
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    dataset_test = GenDEBRIS(
        'test',
        transform=transform_test,
        standardization=standardization,
        agg_to_water=options['agg_to_water']
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=options['batch'],
        shuffle=False
    )

    global labels
    if options['agg_to_water']:
        labels = labels[:-4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on {device}")

    model = UNet(
        input_bands=options['input_channels'],
        output_classes=options['output_channels'],
        hidden_channels=options['hidden_channels']
    )
    model.to(device)

    model_file = options['model_path']
    logging.info(f'Loading model from: {model_file}')

    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_predicted = []

    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc="Testing"):
            image = image.to(device)
            target = target.to(device)

            logits = model(image)
            logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
            logits = logits.reshape((-1, options['output_channels']))
            target = target.reshape(-1)

            mask = target != -1
            logits = logits[mask]
            target = target[mask]

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            target = target.cpu().numpy()

            y_predicted += probs.argmax(1).tolist()
            y_true += target.tolist()

        acc = Evaluation(y_predicted, y_true)
        logging.info("\nSTATISTICS:\n")
        logging.info("Evaluation: " + str(acc))
        print("Evaluation:", acc)

        conf_mat = confusion_matrix(y_true, y_predicted, labels)
        logging.info("Confusion Matrix:\n" + str(conf_mat.to_string()))
        print("Confusion Matrix:\n", conf_mat.to_string())

        if options['predict_masks']:
            path = os.path.join(root_path, 'data', 'patches')
            ROIs = np.genfromtxt(os.path.join(root_path, 'data', 'splits', 'test_X.txt'), dtype='str')

            impute_nan = np.tile(bands_mean, (256, 256, 1))

            for roi in tqdm(ROIs):
                roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
                roi_name = '_'.join(['S2'] + roi.split('_'))
                roi_file = os.path.join(path, roi_folder, roi_name + '.tif')

                os.makedirs(options['gen_masks_path'], exist_ok=True)
                output_image = os.path.join(
                    options['gen_masks_path'],
                    os.path.basename(roi_file).split('.tif')[0] + f'_unet_epoch_{options["epoch"]}.tif'
                )

                with rasterio.open(roi_file, mode='r') as src:
                    tags = src.tags().copy()
                    meta = src.meta
                    image = src.read()
                    image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
                    dtype = src.read(1).dtype

                meta.update(count=1)

                with rasterio.open(output_image, 'w', **meta) as dst:
                    nan_mask = np.isnan(image)
                    image[nan_mask] = impute_nan[nan_mask]

                    image = transform_test(image)
                    image = standardization(image)
                    image = image.to(device)

                    logits = model(image.unsqueeze(0))
                    probs = torch.nn.functional.softmax(logits.detach(), dim=1).cpu().numpy()
                    predicted_mask = probs.argmax(1).squeeze() + 1

                    save_mask_as_npy(predicted_mask, options['gen_masks_path'], roi_name, options['epoch'])

                    dst.write_band(1, predicted_mask.astype(dtype).copy())
                    dst.update_tags(**tags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--agg_to_water', default=True, type=bool, help='Aggregate water-related classes.')
    parser.add_argument('--batch', default=5, type=int, help='Batch size for testing.')
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands.')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes.')
    parser.add_argument('--hidden_channels', default=16, type=int, help='Number of hidden features.')
    parser.add_argument('--epoch', default=44, type=int, help='Specify epoch for comparison.')
    parser.add_argument('--model_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models', 'best_model_marine_debris.pth'), help='Path to trained model.')
    parser.add_argument('--predict_masks', default=True, type=bool, help='Save predicted masks.')
    parser.add_argument('--gen_masks_path', default=os.path.join(root_path, 'data', 'predicted_unet'), help='Path to save masks.')

    args = parser.parse_args()
    options = vars(args)
    main(options)
