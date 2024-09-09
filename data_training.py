import os, torch, json, cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter
import utils, random
from torchvision import transforms
import albumentations as A
from scipy.spatial.transform.rotation import Rotation

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

# 300W_LP
class Pose_300WLP(Dataset):
    def __init__(self, root_path, filename_path, 
        img_ext='.jpg', annot_ext='.mat', 
        image_mode='RGB', target_size = 128, 
        augment = 0.5, flip = True, separate = False):

        self.data_dir = root_path
        self.transform = self.albu_augmentations(target_size)
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.X_train, self.y_train = [], []
        self.bbox = []

        filename_list = get_list_from_filenames(filename_path)
        if separate:
            mid = int(len(filename_list) * 0.8)
            filename_list = filename_list[:mid]
            
        self.X_train = filename_list
        self.y_train = filename_list

        self.image_mode = image_mode
        self.length = len(self.X_train)

        self.augment = augment
        self.flip = flip
    
    def albu_augmentations(self, target_size = 128, p = 0.5):
        albu_transformations = [A.GaussianBlur(p=p), 
                                A.RandomBrightnessContrast(p=p),
                                A.augmentations.transforms.GaussNoise(p=p)] 
        albu_transformations = [A.Compose([
            A.augmentations.geometric.resize.Resize(target_size, target_size, p = 1.0),
            x]) for x in albu_transformations]

        transformations = transforms.Compose([transforms.Resize((target_size, target_size)),
                                              transforms.RandomResizedCrop(size=target_size,scale=(0.8,1))])

        resize_compose = transforms.Compose([transforms.Resize((target_size, target_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return [*albu_transformations, transformations, resize_compose]

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        
        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)

        # And convert to degrees.
        pitch = pose[0] * (180 / np.pi)
        yaw = pose[1] * (180 / np.pi)
        roll = pose[2] * (180 / np.pi)

        # Data Augmentation
        augment_or_not = np.random.random_sample()
        if augment_or_not <= self.augment and self.augment > 0:
            # Bounding box Augmentation
            rand = random.randint(1, 4) if self.flip else random.randint(2, 4)
            if rand == 1: # Flip
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif rand == 2: # Random Shifting
                mid_x = int((x_max + x_min) / 2)
                mid_y = int((y_max + y_min) / 2)
                width = x_max - x_min
                height = y_max - y_min
                kx = np.random.random_sample() * 0.2 - 0.1
                ky = np.random.random_sample() * 0.2 - 0.1
                shiftx = mid_x + width * kx
                shifty = mid_y + height * ky
                x_min = shiftx - width/2
                x_max = shiftx + width/2
                y_min = shifty - height/2
                y_max = shifty + height/2
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            elif rand == 3: # Random Scaling
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                img = self.transform[-2](img)
            else:
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Image Augmentation
            rand = random.randint(1, 4)
            if rand >= 1 and rand <= 3:
                img = np.array(img)
                img = self.transform[rand-1](image = img)['image']
                img = Image.fromarray(img)
        else:
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            
        # finalize Transform
        img = self.transform[-1](img)
        
        # Bin values
        ten_bins = np.arange(-100, 81, 20)
        shifted_bins = np.array(range(0, 20, 1))

        # Twenty bins
        binned_pose = np.digitize([yaw, pitch, roll], ten_bins) - 1

        # Shifted bins
        shifted_pose = [yaw - ten_bins[binned_pose[0]], pitch - ten_bins[binned_pose[1]], roll - ten_bins[binned_pose[2]]]
        shifted_pose = np.digitize(shifted_pose, shifted_bins) - 1

        labels = torch.LongTensor(binned_pose)
        shifted_labels = torch.LongTensor(shifted_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        return img, labels, shifted_labels, cont_labels, index

    def __len__(self):
        # 122,415
        return self.length
    
# BIWI Dataset
class BIWI(Dataset):
    def __init__(self, root_path, filename_list, 
        img_ext='.jpg', annot_ext='.mat', 
        image_mode='RGB', target_size = 224, 
        augment = 0.5, flip = True):

        self.root_path = root_path

        self.transform = self.albu_augmentations(target_size)
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.X_train, self.y_train = [], []

        self.X_train = get_list_from_filenames(filename_list)
        self.y_train = get_list_from_filenames(filename_list)

        self.image_mode = image_mode
        self.length = len(self.X_train)

        self.augment = augment
        self.flip = flip

    def albu_augmentations(self, target_size = 128, p = 0.5):
        albu_transformations = [A.GaussianBlur(p=p), #p=1.0
                                A.RandomBrightnessContrast(p=p), #p=1.0
                                A.augmentations.transforms.GaussNoise(p=p)] #p=1.0
        albu_transformations = [A.Compose([
            A.augmentations.geometric.resize.Resize(target_size, target_size, p = 1.0),
            x]) for x in albu_transformations]

        transformations = transforms.Compose([transforms.Resize((target_size, target_size)),
                                              transforms.RandomResizedCrop(size=target_size,scale=(0.8,1))])

        resize_compose = transforms.Compose([transforms.Resize((target_size, target_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return [*albu_transformations, transformations, resize_compose]

    def __getitem__(self, index):
        data = np.load(os.path.join(self.root_path, self.X_train[index] + ".npz"))

        raw_img = np.uint8(data['image'].copy())
        img = np.uint8(data['image'])
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img.convert(self.image_mode)
        
        yaw, pitch, roll = data['pose']
        
        # Data Augmentation
        augment_or_not = np.random.random_sample()
        if augment_or_not <= self.augment and self.augment > 0:
            # Bounding box Augmentation (we do not have random shifted and cropped because the subject's faces are already cropped)
            rand = random.randint(1, 3) if self.flip else random.randint(2, 3)
            if rand == 1: # Flip
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif rand == 2: # Random Scaling
                img = self.transform[-2](img)

            # Image Augmentation
            rand = random.randint(1, 4)
            if rand >= 1 and rand <= 3:
                img = np.array(img)
                img = self.transform[rand-1](image = img)['image']
                img = Image.fromarray(img)
            
        # finalize Transform
        img = self.transform[-1](img)
        
        # Bin values
        ten_bins = np.arange(-100, 81, 20)
        shifted_bins = np.array(range(0, 20, 1))

        # Twenty bins
        binned_pose = np.digitize([yaw, pitch, roll], ten_bins) - 1

        # Shifted bins
        shifted_pose = [yaw - ten_bins[binned_pose[0]], pitch - ten_bins[binned_pose[1]], roll - ten_bins[binned_pose[2]]]
        shifted_pose = np.digitize(shifted_pose, shifted_bins) - 1

        labels = torch.LongTensor(binned_pose)
        shifted_labels = torch.LongTensor(shifted_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        return img, labels, shifted_labels, cont_labels, index

    def __len__(self):
        # 15,667
        return self.length