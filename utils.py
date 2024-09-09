# This part is based on the code from Ruiz's HopeNet
import numpy as np
import torch
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt3d_from_mat(mat_path):
    # Get 3D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt3d_68']
    return pt2d

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d