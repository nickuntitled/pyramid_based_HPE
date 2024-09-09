import os, argparse, json, datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms

import data as validate_data
import data_training as data
import numpy as np 
import random
from model import nice_model as aggregate_model
from rich.console import Console

import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the EfficientNetV2-based network.')

    # Data Augmentation
    parser.add_argument('--flip', dest='flip', help='flip.',
          default=0, type=int)
    parser.add_argument('--augment', dest='augment', help='augment rate before entering two-step data augmentation.',
          default=0.5, type=float)

    # GPU Usage
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)

    # Number of Epochs
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=100, type=int)

    # Batch size
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=32, type=int)

    # Learning Rate
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.00001, type=float)
          
    # Training Dataset
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='300W_LP', type=str)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/300wlp_files.txt', type=str)

    # Validation Dataset (You can disable it by typing like None)
    parser.add_argument('--val_dataset', dest='val_dataset', help='val Dataset type.', default='None', type=str)
    parser.add_argument(
        '--val_data_dir', dest='val_data_dir', help='Directory path for data.',
        default='', type=str)
    parser.add_argument(
        '--val_filename_list', dest='val_filename_list',
        help='Path to text file containing relative paths for every example.',
        default='', type=str)

    # Image Size (default: 224)
    parser.add_argument('--target_size', dest='target_size', help='target_size',
          default=224, type=int)

    # Output string for a snapshot folder.
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    
    # Snapshot file for transfer learning for continuation of training
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--transfer', dest='transfer', help='transfer.',
          default=1, type=int)

    # Model (4 = EfficientNetV2-S, 3 = EfficientNetV2-M, other = EfficientNetV2-L)
    parser.add_argument('--efficient', dest='efficient', help='efficient.',
          default=4, type=int)

    args = parser.parse_args()
    return args

# Generator function that yields params that will be optimized.
def get_non_ignored_params(model):
    b = [model.stem, model.blocks, model.side, model.aspp]

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    b = [model.fc_yaw_coarse, model.fc_yaw_shift, 
         model.fc_pitch_coarse, model.fc_pitch_shift, 
         model.fc_roll_coarse, model.fc_roll_shift]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def wrapped_loss():
    def call(predict, target):
        a = torch.square(predict - target)
        b = torch.square(360 - (predict - target))
        c = torch.minimum(a, b)
        c = torch.mean(c)
        return c

    return call

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load Rich Console
    console = Console()

    # Save Snapshot Path
    if args.output_string != '':
        base_path = f'workdirs_{ args.output_string }'
    else:
        base_path = f'workdirs'
    
    output_path = f'{ base_path }/snapshots'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Model structure
    if args.efficient == 4:
        network = 'efficientnet_v2_s'
    elif args.efficient == 3:
        network = 'efficientnet_v2_m'
    else:
        network = 'efficientnet_v2_l'

    # Model
    model = aggregate_model(pretrained = True, network = network, start_size = args.target_size)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    console.log(f"This model has [bold magenta]{ pytorch_total_params }[/bold magenta] parameters.")
        
    optimizer = torch.optim.AdamW([
        { 'params': get_non_ignored_params(model), 'lr': args.lr },
        { 'params': get_fc_params(model), 'lr': args.lr * 5}
    ], lr = args.lr)

    model.cuda(gpu)

    if args.snapshot != '':
        console.log(f"Load Snapshot from { args.snapshot }.")
        saved_state_dict = torch.load(args.snapshot, map_location = 'cuda')

        if args.transfer != 1:
            model.load_state_dict(saved_state_dict['model'])
            start_epoch = saved_state_dict['epoch']

            if 'optimizer' in saved_state_dict:
                optimizer.load_state_dict(saved_state_dict['optimizer'])

            start_epoch += 1
        else:
            model.load_state_dict(saved_state_dict['model'] if 'model' in saved_state_dict else saved_state_dict)
            start_epoch = 0
    else:
        start_epoch = 0

    console.log('Loading data.')
    if args.dataset == '300W_LP':
        pose_dataset = data.Pose_300WLP(args.data_dir, 
            args.filename_list, flip = args.flip == 1,
            augment = args.augment, target_size = args.target_size)
    elif args.dataset == '300W_LP_8020':
        pose_dataset = data.Pose_300WLP(args.data_dir, 
            args.filename_list, flip = args.flip == 1,
            augment = args.augment, target_size = args.target_size,
            separate = True)
    elif args.dataset == 'BIWI':
        pose_dataset = data.BIWI(args.data_dir, args.filename_list, flip = args.flip == 1,
            augment = args.augment, target_size = args.target_size)
    else:
        console.log('Not implement')
        exit(1)

    transformations = transforms.Compose([
        transforms.Resize((args.target_size, args.target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.val_dataset =='AFLW2000':
        val_pose_dataset = validate_data.AFLW2000(args.val_data_dir, args.val_filename_list, None, transformations, sixd = False, ad = 0.2)
    elif args.val_dataset == 'BIWI':
        val_pose_dataset = validate_data.BIWI_kinect(args.val_data_dir, args.val_filename_list, None, transformations, 'val', sixd = False)
    else:
        val_pose_dataset = None
        
    console.log(f"Training { len(pose_dataset )} images.")

    # Training
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    # Validation
    if val_pose_dataset is not None:
        test_loader = torch.utils.data.DataLoader(dataset=val_pose_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=4)

    # Loss Function                                            
    class_criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = wrapped_loss()

    # Regression loss coefficient
    alpha = 1

    # SoftMax
    softmax = nn.Softmax().cuda(gpu)
    
    # Bins
    idx_tensor = [idx for idx in range(20)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
    ten_idx_tensor = np.array([idx for idx in range(10)])
    ten_idx_tensor = Variable(torch.FloatTensor(ten_idx_tensor)).cuda(gpu)

    console.log('Ready to train network.')
    f = open(f"{ base_path }/training.log", "a")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, (images, labels, shifted_labels, cont_labels, index) in enumerate(train_loader):
            batch_size = images.size(0)
            images = Variable(images).cuda(gpu)

            # Rotation euler labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)
            
            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Shifted Binned labels
            shifted_label_yaw = Variable(shifted_labels[:,0]).cuda(gpu)
            shifted_label_pitch = Variable(shifted_labels[:,1]).cuda(gpu)
            shifted_label_roll = Variable(shifted_labels[:,2]).cuda(gpu)

            yaw_coarse, yaw_shift, pitch_coarse, pitch_shift, roll_coarse, roll_shift = model(images)

            # 20-bin loss
            # Cross entropy loss
            loss_yaw = class_criterion(yaw_coarse, label_yaw)
            loss_pitch = class_criterion(pitch_coarse, label_pitch)
            loss_roll = class_criterion(roll_coarse, label_roll)

            # 20-shift bin loss
            # Cross entropy loss
            loss_shifted_yaw = class_criterion(yaw_shift, shifted_label_yaw)
            loss_shifted_pitch = class_criterion(pitch_shift, shifted_label_pitch)
            loss_shifted_roll = class_criterion(roll_shift, shifted_label_roll)
            loss_yaw   += loss_shifted_yaw
            loss_pitch += loss_shifted_pitch
            loss_roll  += loss_shifted_roll

            # Continuous
            # Coarse
            ten_yaw = softmax(yaw_coarse)
            ten_pitch = softmax(pitch_coarse)
            ten_roll = softmax(roll_coarse)

            yaw_predicted = torch.sum(ten_yaw * ten_idx_tensor, 1) * 20 - 100
            pitch_predicted = torch.sum(ten_pitch * ten_idx_tensor, 1) * 20 - 100
            roll_predicted = torch.sum(ten_roll * ten_idx_tensor, 1) * 20 - 100

            # Shift
            shifted_yaw = softmax(yaw_shift)
            shifted_pitch = softmax(pitch_shift)
            shifted_roll = softmax(roll_shift)

            yaw_predicted += torch.sum(shifted_yaw * idx_tensor, 1) 
            pitch_predicted += torch.sum(shifted_pitch * idx_tensor, 1)
            roll_predicted += torch.sum(shifted_roll * idx_tensor, 1)

            #  MSE Loss
            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            # Calculation
            total_loss = loss_yaw + loss_pitch + loss_roll

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                console.log('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f Pitch %.4f Roll %.4f '
                    %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, 
                    loss_yaw.cpu().item(), loss_pitch.cpu().item(), loss_roll.cpu().item()))
        
            dict_loss = {
                "mode": "train",
                "epoch": epoch + 1,
                "total_epoch": num_epochs,
                "iter": i+1,
                "total_iter": len(pose_dataset)//batch_size,
                "loss_yaw": loss_yaw.cpu().item(),
                "loss_pitch": loss_pitch.cpu().item(),
                "loss_roll": loss_roll.cpu().item()
            }

            f.write(json.dumps(dict_loss))
            f.write("\n")

        #Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            with console.status("[bold green]Taking snapshot...") as status:
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                torch.save(checkpoint, f'{ output_path }/'+ str(epoch+1) + '.pkl')

        model.eval()
        if val_pose_dataset is not None:
            yaw_error = .0
            pitch_error = .0
            roll_error = .0
            with torch.no_grad():
                data = { 'yaw': { "result": [], "label": [] }, 'pitch': { "result": [], "label": [] }, 'roll': { "result": [], "label": [] } }
                with console.status("[bold green]Testing...") as status:
                    for i, test_temp in enumerate(test_loader):
                        images, labels, cont_labels, raw_img, index = test_temp

                        images = Variable(images).cuda(gpu)
                        batch_size = cont_labels.size(0)

                        label_yaw = cont_labels[:,0].float()
                        label_pitch = cont_labels[:,1].float()
                        label_roll = cont_labels[:,2].float()

                        yaw_coarse, yaw_shift, pitch_coarse, pitch_shift, roll_coarse, roll_shift = model(images)

                        # Continuous
                        # Coarse
                        ten_yaw = softmax(yaw_coarse)
                        ten_pitch = softmax(pitch_coarse)
                        ten_roll = softmax(roll_coarse)

                        yaw_predicted = torch.sum(ten_yaw * ten_idx_tensor, 1) * 20 - 100
                        pitch_predicted = torch.sum(ten_pitch * ten_idx_tensor, 1) * 20 - 100
                        roll_predicted = torch.sum(ten_roll * ten_idx_tensor, 1) * 20 - 100

                        # Shift
                        shifted_yaw = softmax(yaw_shift)
                        shifted_pitch = softmax(pitch_shift)
                        shifted_roll = softmax(roll_shift)

                        yaw_predicted += torch.sum(shifted_yaw * idx_tensor, 1) 
                        pitch_predicted += torch.sum(shifted_pitch * idx_tensor, 1)
                        roll_predicted += torch.sum(shifted_roll * idx_tensor, 1)

                        # Mean absolute error
                        p_gt_deg = label_pitch
                        y_gt_deg = label_yaw
                        r_gt_deg = label_roll
                        p_pred_deg = pitch_predicted.cpu()
                        y_pred_deg = yaw_predicted.cpu()
                        r_pred_deg = roll_predicted.cpu()

                        pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                            p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg), torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
                        yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                            y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg), torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
                        roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                            r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg), torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])

                total = len(val_pose_dataset)

                mean_error = ((yaw_error + pitch_error + roll_error)) / 3
                console.log('Test error in degrees of the model on the ' + str(total) +
                ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, Mean: %.4f' % (yaw_error / total,
                pitch_error / total, roll_error / total, mean_error / total))
                
                f.write(json.dumps({
                    "mode": "val",
                    "epoch": epoch + 1,
                    "total_epoch": num_epochs,
                    "total_image": total,
                    "yaw_error": yaw_error.item() / total,
                    "pitch_error": pitch_error.item() / total,
                    "roll_error": roll_error.item() / total,
                    "mean_error": mean_error.item() / total
                }))

                f.write("\n")
    f.close()
