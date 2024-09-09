# Pyramid-based structure for head pose estimation

This repository proposes HPE architecture based on the research article, "Telemedicine Application of Deep-learning-based Head Pose Estimation from RGB Image Sequence".

This study aims to propose an application of deep-neural network adopting multi-level pyramidal feature extraction, a bi-directional Pyramidal Feature Aggregation Structure (PFAS) for feature fusion, a modified Atrous Spatial Pyramid Pooling (ASPP) module for spatial and channel feature enhancement, and a multi-bin classification and regression module, to derive the Euler angles as the head pose parameters.

---

## How to install packages

The best approach is to use Anaconda (or Miniconda) with the creation of the environment from Python version 3.9, and the installation of the following packages:

- PyTorch 1.13.0 (or 1.13.1)
- NumPy
- Matplotlib
- Pandas
- Scipy
- Pillow
- Albumentations
- OpenCV
- Rich

by using the following commands

```
conda env create --name <environment name> --file=environment.yml
```

---

## Dataset Preparation

Download the dataset, and extract the dataset into datasets/< dataset name folder>.

### 300W_LP

You have to download [300W_LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) dataset from the author, and extract into datasets/300W_LP folder. The structure should be like this.

```
datasets
-- 300W_LP
---- AFW
---- LFPW
---- HELEN
---- IBUG
---- AFW_Flip
---- HELEN_Flip
---- LFPW_Flip
---- IBUG_Flip
---- 300wlp_list.txt
```

### AFLW2000

You have to download [AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) dataset from the author, and extract into datasets/AFLW2000 folder. The structure should be like this.

```
datasets
-- AFLW2000
---- *.jpg and *.mat files
---- aflw2000_list.txt
```

### BIWI

You have to download the [prepared BIWI](https://drive.google.com/file/d/1T2VfY35hSVPF9uNzJteZQOnz_ADQqNqM/view?usp=sharing) dataset, and extract into datasets/BIWI folder. The structure should be like this.

```
datasets
-- BIWI
---- *.npz files
---- biwi_list.txt
---- biwi_train.txt
---- biwi_test.txt
```

The **biwi_list.txt** is for the entire BIWI dataset. The **biwi_train.txt** and **biwi_test.txt** are for the training and testing subset, which is separated in the ratio of 70 and 30.

---

## Training

Before training, you have to download the pretrained model for EfficientNetV2-S or EfficientNetV2-M which is called pretrained_effv2_s.pkl or pretrained_effv2_m.pkl on the [Google Drive](https://drive.google.com/drive/folders/1Rw1Mv1RMDFb0Gn2htVZ5iFSH719McszA?usp=sharing).

After downloading the pretrained model, you can train by using command below:

```
python train.py --dataset <dataset name> --data_dir datasets/<dataset name>/ --filename_list datasets/<dataset name>/<filename list> --num_epochs <number of epochs> --batch_size <no of epochs> --output_string <desired output> --batch_size <batch_size> --augment <probability that the training image will be processed through data augmentation> --flip <1 = allow flipping, 0 = not allow>
```

If you want to continue training, you have to type like this below.

```
python train.py --dataset <dataset name> --data_dir datasets/<dataset name>/ --filename_list datasets/<dataset name>/<filename list> --num_epochs <number of epochs> --batch_size <no of epochs> --output_string <desired output> --batch_size <batch_size> --transfer 0 --snapshot <snapshot path> --augment <probability that the training image will be processed through data augmentation> --flip <1 = allow flipping, 0 = not allow>
```

### 300W_LP

Train on the entire 300W_LP dataset (for AFLW2000 and BIWI).

```
python train.py --dataset 300W_LP --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/300wlp_list.txt --num_epochs 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 0 --output_string 300W_LP
```

Train on 80% of the entire 300W_LP dataset (for validation with the latter 20% of 300W_LP)

```
python train.py --dataset 300W_LP_8020 --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/300wlp_list.txt --num_epochs 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 0 --output_string 300W_LP_8020
```

### BIWI

Train on BIWI.

```
python train.py --dataset BIWI --data_dir datasets/BIWI/ --filename_list datasets/BIWI/biwi_train.txt --num_epochs 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 1 --output_string BIWI
```

The recommended approach is to transfer the pretrained model from 300W_LP which is fine-tune for BIWI.

```
python train.py --dataset BIWI --data_dir datasets/BIWI/ --filename_list datasets/BIWI/biwi_train.txt --num_epochs 100 --batch_size 32 --lr 0.00001 --augment 0.5 --flip 1 --output_string BIWI --transfer 1 --snapshot <pretrained 300W_LP path>
```

---

## Evaluation

If you don't want to train the model, you can use the pretrained model available on [The author's Google Drive](https://drive.google.com/drive/folders/1Rw1Mv1RMDFb0Gn2htVZ5iFSH719McszA?usp=sharing).

### For train by 300W_LP

Test on AFLW2000

```
python test.py --dataset AFLW2000 --data_dir datasets/AFLW2000 --filename_list datasets/AFLW2000/aflw2000_list.txt --snapshot <snapshot path> --batch_size 32
```

Test on BIWI

```
python test.py --dataset 300WLP2BIWI --data_dir datasets/BIWI --filename_list datasets/BIWI/biwi_list.txt --snapshot <snapshot path> --batch_size 32
```

Test on AFLW2000 on the whole snapshot folder

```
python test.py --dataset AFLW2000 --data_dir datasets/AFLW2000 --filename_list datasets/AFLW2000/aflw2000_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --batch_size 32
```

Test on BIWI on the whole snapshot folder

```
python test.py --dataset 300WLP2BIWI -data_dir datasets/BIWI --filename_list datasets/BIWI/biwi_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --batch_size 32
```

### For train by 80% of the entire 300W_LP

Test on the latter 20% of 300W_LP

```
python test.py --dataset 300W_LP --data_dir datasets/300W_LP --filename_list datasets/300W_LP/300wlp_list.txt --snapshot <snapshot path> --batch_size 32
```

Test on the latter 20% of 300W_LP on the whole snapshot folder

```
python test.py --dataset 300W_LP --data_dir datasets/300W_LP --filename_list datasets/300W_LP/300wlp_list.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --batch_size 32
```

### For train by BIWI

Test on BIWI

```
python test.py --dataset BIWI --data_dir datasets/BIWI --filename_list datasets/BIWI/biwi_test.txt --snapshot <snapshot path> --batch_size 32
```

Test on BIWI on the whole snapshot folder

```
python test.py --dataset BIWI --data_dir datasets/BIWI --filename_list datasets/BIWI/biwi_test.txt --snapshot <snapshot folder> --num_epoch <no of total epoch like 100> --batch_size 32
```