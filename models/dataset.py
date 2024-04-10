import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.img_dir = conf.get_string('img_dir')
        self.use_init = conf.get_bool('use_init', default=True)
        self.inlier_only = self.conf.get_bool('inlier_only', default=False)
        if self.use_init:
            self.render_cameras_name = conf.get_string('render_cameras_name')
            self.object_cameras_name = conf.get_string('object_cameras_name')
        self.data_type = conf.get_string('data_type', default='items')
        self.img_psfx = conf.get_string('img_psfx', default='.png')
        self.msk_psfx = conf.get_string('msk_psfx', default='.png')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        
        self.from_ckpt = self.conf.get_bool('from_ckpt', default=False)
        self.pose_ckpt = self.conf.get_string('pose_ckpt', default='')

        self.dtu_novel = self.conf.get_bool('dtu_novel', default=False)

        print(os.path.join(self.data_dir, self.img_dir))
        self.images_lis = sorted(glob(os.path.join(self.data_dir, self.img_dir, '*' + self.img_psfx)))
        f_inlier = os.path.join(self.data_dir, 'correct_list.txt')
        if self.inlier_only:
            assert os.path.exists(f_inlier)
            with open(f_inlier, 'r') as fin:
                inlier_list = [line.strip() for line in fin.readlines()]
            self.images_lis = [img for img in self.images_lis if os.path.basename(img) in inlier_list]

        if self.dtu_novel:
            f_test = os.path.dirname(self.data_dir) + '/test_set.txt'
            with open(f_test, 'r') as fin:
                test_idx = fin.readline().strip().split(',')
            self.test_idx = [int(idx) for idx in test_idx]
            self.dense_list = self.images_lis
            self.test_list = [img for img in self.images_lis if int(os.path.basename(img)[:-4]) in self.test_idx]
            self.images_lis = [img for img in self.images_lis if int(os.path.basename(img)[:-4]) not in self.test_idx]

        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*' + self.msk_psfx)))
        if len(self.masks_lis) != 0:
            self.using_mask = True
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
            self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        else:
            self.using_mask = False

        if 'dtu' in self.data_dir:
            self.data_type = 'DTU'

        if self.use_init:
            camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
            self.camera_dict = camera_dict
            if self.data_type == 'DTU':
                try:
                    self.world_mats_np = [camera_dict['world_mat_{}'.format(int(os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]
                except:
                    self.world_mats_np = [camera_dict['world_mat_{}'.format((os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]
            else:
                self.world_mats_np = [camera_dict['world_mat_{}'.format((os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]

            self.scale_mats_np = []

            if self.data_type == 'DTU':
                try:
                    self.scale_mats_np = [camera_dict['scale_mat_{}'.format(int(os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]
                except:
                    self.scale_mats_np = [camera_dict['scale_mat_{}'.format((os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]
            else:
                self.scale_mats_np = [camera_dict['scale_mat_{}'.format((os.path.basename(idx)[:-4]))].astype(np.float32) for idx in self.images_lis]

            if self.from_ckpt:
                assert self.pose_ckpt and os.path.exists(self.pose_ckpt)
                P = self.world_mats_np[0] @ self.scale_mats_np[0]
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                self.intrinsics_all = torch.from_numpy(intrinsics.astype(np.float32)).repeat(self.n_images, 1, 1).to(self.device)
                self.pose_all = torch.from_numpy(np.load(self.pose_ckpt).astype(np.float32)).to(self.device)
            else:
                self.intrinsics_all = []
                self.pose_all = []

                for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
                    P = world_mat @ scale_mat
                    P = P[:3, :4]
                    intrinsics, pose = load_K_Rt_from_P(None, P)
                    self.intrinsics_all.append(torch.from_numpy(intrinsics).float()) #n,4,4
                    self.pose_all.append(torch.from_numpy(pose).float()) # n,4,4

                self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
                self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
            self.focal = self.intrinsics_all[0][0, 0]

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        self.image_size = (int(self.W), int(self.H))

        if self.data_type == 'tank':
            del self.images, self.images_np

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])

        if self.use_init:
            object_scale_mat = self.scale_mats_np[0]
            object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
            self.object_bbox_min = object_bbox_min[:3, 0]
            self.object_bbox_max = object_bbox_max[:3, 0]
        else:
            self.object_bbox_min = object_bbox_min[:3]
            self.object_bbox_max = object_bbox_max[:3]

        print('Load data: End')

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return self.n_images