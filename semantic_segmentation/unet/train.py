import os
import ast
import sys
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, RandomResizedCrop
from albumentations import ElasticTransform, Compose, RandomBrightnessContrast, HorizontalFlip, VerticalFlip, RandomRotate90
from albumentations.pytorch import ToTensorV2

sys.path.append(up(os.path.abspath(__file__)))
from unet import UNet
from dataloader import GenDEBRIS, bands_mean, bands_std, RandomRotationTransform , class_distr, gen_weights
from unet_plus_plus import UNetPlusPlus
from loss_functions import DiceLoss

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from metrics import Evaluation

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(filename=os.path.join(root_path, 'logs','log_unet.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def compute_inverse_class_weights(class_distribution, device):
    """
    Compute inverse class weights based on class distribution.
    :param class_distribution: Tensor containing the class distribution.
    :param device: The device (CPU or GPU) to which the weights should be moved.
    :return: Inverse class weights as a PyTorch tensor.
    """
    weights = 1.0 / class_distribution
    weights /= weights.sum()
    logging.info(f"Inverse Class Weights: {weights}")
    return weights.to(device)

def compute_class_weights(dataset, num_classes):
    """
    Compute class weights based on the frequency of each class in the dataset.
    """
    class_counts = torch.zeros(num_classes)
    for _, target in dataset:
        unique, counts = torch.unique(target, return_counts=True)
        for u, c in zip(unique, counts):
            if u >= 0:
                class_counts[u] += c

    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    return class_weights

def copy_paste_augmentation(image, target, paste_class=0):
    """
    Perform Copy-Paste augmentation by copying regions of a specific class
    from one image and pasting them onto another image.
    :param image: Input image tensor (B, C, H, W)
    :param target: Target mask tensor (B, H, W)
    :param paste_class: Class index to copy and paste
    """
    batch_size, channels, height, width = image.shape
    mask = target == paste_class
    
    if torch.any(mask):
        source_idx = random.randint(0, batch_size - 1)
        target_idx = random.randint(0, batch_size - 1)
        
        source_mask = mask[source_idx]
        
        if torch.any(source_mask):
            source_mask = source_mask.unsqueeze(0).expand(channels, -1, -1)
            
            source_region = image[source_idx] * source_mask
            
            image[target_idx] = torch.maximum(image[target_idx], source_region)
            target[target_idx] = torch.maximum(target[target_idx], mask[source_idx])
    
    return image, target

def advanced_augmentation():
    """
    Define advanced augmentations using Albumentations.
    """
    return Compose([
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        RandomBrightnessContrast(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ToTensorV2()
    ])

from torch.utils.data import ConcatDataset

def oversample_marine_debris(dataset, class_index=0):
    """
    Oversample the marine debris class by duplicating its samples.
    """
    marine_debris_samples = [dataset[i] for i in range(len(dataset)) if torch.any(dataset[i][1] == class_index)]
    oversampled_dataset = ConcatDataset([dataset] + [marine_debris_samples] * 2)
    return oversampled_dataset

def main(options):
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    
    writer = SummaryWriter(os.path.join(root_path, 'logs', options['tensorboard']))
        
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        RandomRotationTransform([-90, 0, 90, 180]),
        transforms.RandomHorizontalFlip(),
        RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    
    albumentations_transform = advanced_augmentation()
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    standardization = transforms.Normalize(bands_mean, bands_std)
        
    if options['mode'] == 'train':
        dataset_train = GenDEBRIS('train', transform=transform_train, standardization=standardization, agg_to_water=options['agg_to_water'])
        dataset_train = oversample_marine_debris(dataset_train, class_index=options['paste_class'])
        dataset_test = GenDEBRIS('val', transform=transform_test, standardization=standardization, agg_to_water=options['agg_to_water'])

        class_weights = compute_class_weights(dataset_train, options['output_channels']).to(device)
        logging.info(f"Class Weights: {class_weights}")
        
        train_loader = DataLoader(
            dataset_train, 
            batch_size=options['batch'], 
            shuffle=True,
            num_workers=options['num_workers'],
            pin_memory=options['pin_memory'],
            persistent_workers=options['persistent_workers'],
            worker_init_fn=seed_worker,
            generator=g
        )
        
        test_loader = DataLoader(
            dataset_test, 
            batch_size=options['batch'], 
            shuffle=False,
            num_workers=options['num_workers'],
            pin_memory=options['pin_memory'],
            persistent_workers=options['persistent_workers'],
            worker_init_fn=seed_worker,
            generator=g
        )
        
    elif options['mode'] == 'test':
        dataset_test = GenDEBRIS('test', transform=transform_test, standardization=standardization, agg_to_water=options['agg_to_water'])
        test_loader = DataLoader(
            dataset_test, 
            batch_size=options['batch'], 
            shuffle=False,
            num_workers=options['num_workers'],
            pin_memory=options['pin_memory'],
            persistent_workers=options['persistent_workers'],
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        raise ValueError("Invalid mode. Choose between 'train' and 'test'.")
        
    model = UNetPlusPlus(input_bands=options['input_channels'], output_classes=options['output_channels'], hidden_channels=options['hidden_channels'])
    model.to(device)

    if options['resume_from_epoch'] > 1:
        resume_model_dir = os.path.join(options['checkpoint_path'], str(options['resume_from_epoch']))
        model_file = os.path.join(resume_model_dir, 'model.pth')
        logging.info('Loading model files from folder: %s' % model_file)

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint)

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global class_distr
    if options['agg_to_water']:
        agg_distr = sum(class_distr[-4:])
        class_distr[6] += agg_distr
        class_distr = class_distr[:-4]

    if options['weighting_mechanism'] == 'log':
        weight = gen_weights(class_distr, c=options['weight_param'])
        logging.info(f"Using log-based class weighting with parameter c={options['weight_param']}")
    elif options['weighting_mechanism'] == 'inverse':
        weight = compute_inverse_class_weights(class_distr, device)
        logging.info("Using inverse class weighting mechanism.")
    else:
        raise ValueError("Invalid weighting mechanism. Choose between 'log' and 'inverse'.")

    criterion = lambda logits, target: (
        torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=class_weights)(logits, target) +
        DiceLoss()(logits, target)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay'])

    if options['reduce_lr_on_plateau'] == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs'])

    start = options['resume_from_epoch'] + 1
    epochs = options['epochs']
    eval_every = options['eval_every']

    best_marine_debris_iou = float('-inf')
    best_overall_iou = float('-inf')
    best_marine_model_path = os.path.join(options['checkpoint_path'], 'best_model_marine_debris.pth')
    best_overall_model_path = os.path.join(options['checkpoint_path'], 'best_model_overall.pth')

    os.makedirs(options['checkpoint_path'], exist_ok=True)
    
    logging.info(f"Checkpoint directory: {options['checkpoint_path']}")
    logging.info(f"Best marine debris model will be saved to: {best_marine_model_path}")
    logging.info(f"Best overall model will be saved to: {best_overall_model_path}")

    if options['mode']=='train':
        dataiter = iter(train_loader)
        image_temp, _ = next(iter(dataiter))
        writer.add_graph(model, image_temp.to(device))
        
        model.train()
        
        for epoch in range(start, epochs+1):
            training_loss = []
            training_batches = 0
            
            i_board = 0
            for (image, target) in tqdm(train_loader, desc="training"):
                try:
                    image = image.to(device, non_blocking=True).contiguous()
                    target = target.to(device, non_blocking=True).contiguous()
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    logits = model(image)
                    loss = criterion(logits, target)
                    loss.backward()
                    
                    optimizer.step()
                    
                    del logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    print(f"Error during training: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                training_batches += target.shape[0]
                training_loss.append((loss.data*target.shape[0]).tolist())
                
                writer.add_scalar('training loss', loss , (epoch - 1) * len(train_loader)+i_board)
                i_board+=1
            
            logging.info("Training loss was: " + str(sum(training_loss) / training_batches))
            
            
            if epoch % eval_every == 0 or epoch==1:
                model.eval()
    
                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []
                
                with torch.no_grad():
                    for (image, target) in tqdm(test_loader, desc="testing"):
    
                        image = image.to(device)
                        target = target.to(device)
    
                        logits = model(image)
                        
                        loss = criterion(logits, target)
                                    
                        logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                        logits = logits.reshape((-1,options['output_channels']))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]
                        
                        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                        target = target.cpu().numpy()
                        
                        test_batches += target.shape[0]
                        test_loss.append((loss.data*target.shape[0]).tolist())
                        y_predicted += probs.argmax(1).tolist()
                        y_true += target.tolist()
                            
                        
                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)
                    
                    
                    acc = Evaluation(y_predicted, y_true)
                    logging.info("\n")
                    logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
                    logging.info("STATISTICS AFTER EPOCH " +str(epoch) + ": \n")
                    logging.info("Evaluation: " + str(acc))
    
                    if "classIoU" in acc:
                        marine_debris_iou = acc["classIoU"][0]
                        if marine_debris_iou > best_marine_debris_iou:
                            best_marine_debris_iou = marine_debris_iou
                            torch.save(model.state_dict(), best_marine_model_path)
                            logging.info(f"New best marine debris model saved with IoU: {best_marine_debris_iou}")

                    if "IoU" in acc:
                        overall_iou = acc["IoU"]
                        if overall_iou > best_overall_iou:
                            best_overall_iou = overall_iou
                            torch.save(model.state_dict(), best_overall_model_path)
                            logging.info(f"New best overall model saved with IoU: {best_overall_iou}")

                    model_dir = os.path.join(options['checkpoint_path'], str(epoch))
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
                    logging.info(f"Model for epoch {epoch} saved to {model_dir}")

                    writer.add_scalars('Loss per epoch', {'Test loss':sum(test_loss) / test_batches, 
                                                          'Train loss':sum(training_loss) / training_batches}, 
                                       epoch)
                    
                    writer.add_scalar('Precision/test macroPrec', acc["macroPrec"] , epoch)
                    writer.add_scalar('Precision/test microPrec', acc["microPrec"] , epoch)
                    writer.add_scalar('Precision/test weightPrec', acc["weightPrec"] , epoch)
                    
                    writer.add_scalar('Recall/test macroRec', acc["macroRec"] , epoch)
                    writer.add_scalar('Recall/test microRec', acc["microRec"] , epoch)
                    writer.add_scalar('Recall/test weightRec', acc["weightRec"] , epoch)
                    
                    writer.add_scalar('F1/test macroF1', acc["macroF1"] , epoch)
                    writer.add_scalar('F1/test microF1', acc["microF1"] , epoch)
                    writer.add_scalar('F1/test weightF1', acc["weightF1"] , epoch)
                    
                    writer.add_scalar('IoU/test MacroIoU', acc["IoU"] , epoch)
                    
                    logging.info(f"Class-Specific IoU: {acc['classIoU']}")
                    logging.info(f"Confusion Matrix:\n{np.array(acc['confusionMatrix'])}")
    
                if options['reduce_lr_on_plateau'] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()
                    
                model.train()
               
    elif options['mode']=='test':
        
        model.eval()

        test_loss = []
        test_batches = 0
        y_true = []
        y_predicted = []
        
        with torch.no_grad():
            for (image, target) in tqdm(test_loader, desc="testing"):

                image = image.to(device)
                target = target.to(device)

                logits = model(image)
                
                loss = criterion(logits, target)

                logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                logits = logits.reshape((-1,options['output_channels']))
                target = target.reshape(-1)
                mask = target != -1
                logits = logits[mask]
                target = target[mask]
                
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                target = target.cpu().numpy()
                
                test_batches += target.shape[0]
                test_loss.append((loss.data*target.shape[0]).tolist())
                y_predicted += probs.argmax(1).tolist()
                y_true += target.tolist()
                
            y_predicted = np.asarray(y_predicted)
            y_true = np.asarray(y_true)
            
            acc = Evaluation(y_predicted, y_true)
            logging.info("\n")
            logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
            logging.info("STATISTICS: \n")
            logging.info("Evaluation: " + str(acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--agg_to_water', default=True, type=bool,  help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')
  
    parser.add_argument('--mode', default='train', help='select between train or test ')
    parser.add_argument('--epochs', default=45, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=5, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')
    
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    parser.add_argument('--hidden_channels', default=16, type=int, help='Number of hidden features')
    parser.add_argument('--weight_param', default=1.03, type=float, help='Weighting parameter for Loss Function')

    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='learning rate decay')
    parser.add_argument('--reduce_lr_on_plateau', default=0, type=int, help='reduce learning rate when no increase (0 or 1)')
    parser.add_argument('--lr_steps', default='[40]', type=str, help='Specify the steps that the lr will be reduced')

    parser.add_argument('--checkpoint_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models'), help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    parser.add_argument('--num_workers', default=0, type=int, help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=True, type=bool, help='Use pinned memory or not')
    parser.add_argument('--persistent_workers', default=False, type=bool, help='This allows to maintain the workers Dataset instances alive.')
    parser.add_argument('--tensorboard', default='tsboard_segm', type=str, help='Name for tensorboard run')
    parser.add_argument('--weighting_mechanism', default='inverse', type=str, help="Choose weighting mechanism: 'log' or 'inverse'")
    parser.add_argument('--paste_class', default=0, type=int, help="Class index for Copy-Paste augmentation.")
    parser.add_argument('--model_architecture', default='unet++', type=str, help="Choose model architecture: 'unet++' or 'deeplabv3+'.")

    args = parser.parse_args()
    options = vars(args)
    
    lr_steps = ast.literal_eval(options['lr_steps'])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise
        
    options['lr_steps'] = lr_steps
    
    logging.info('parsed input parameters:')
    logging.info(json.dumps(options, indent = 2))
    main(options)
