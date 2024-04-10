import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from utils.lie_group_helper import make_c2w, Exp

class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None

        if isinstance(init_c2w, str):
            poses = np.load(init_c2w).astype(np.float32)
            init_c2w = [torch.from_numpy(pose) for pose in poses]
            init_c2w = torch.stack(init_c2w)
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def getAll(self):
        c2w = []
        for r,t in zip(self.r, self.t):
            R = Exp(r)  # (3, 3)
            c2w.append(torch.cat([R, t.unsqueeze(1)], dim=1))  # (3, 4)
        return torch.stack(c2w) # n, 3,4

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w

class LearnIntrin(nn.Module):
    def __init__(self, H, W, req_grad, fx_only=True, order=2, init_focal=None):
        super().__init__()
        self.H = H
        self.W = W
        self.order = order  # check our supplementary section.
        self.device = torch.device('cuda')

        if isinstance(init_focal, str):
            init_focal = np.load(init_focal)
        elif isinstance(init_focal, float):
            init_focal = torch.tensor(init_focal, dtype=torch.float32)

        if init_focal is None:
            self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
        else:
            if self.order == 2:
                # a**2 * W = fx  --->  a**2 = fx / W
                # coe_x = torch.tensor(torch.sqrt(init_focal / float(W)), requires_grad=False).float()
                coe_x = torch.sqrt(init_focal / float(W)).clone().detach().float().requires_grad_(True)
            elif self.order == 1:
                # a * W = fx  --->  a = fx / W
                coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
            else:
                print('Focal init order need to be 1 or 2. Exit')
                exit()
            self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )

    def forward(self):
        fx = self.fx
        if self.order == 2:
            intrinsic = torch.eye(4)
            intrinsic[(0, 1), (0, 1)] = fx**2 * self.W
            intrinsic[0, 2] = self.W / 2
            intrinsic[1, 2] = self.H / 2
        else:
            intrinsic = np.array([[fx * self.W, 0., self.W/2, 0.],
                                [0., fx * self.W, self.H/2, 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]]).type(torch.float32).to(self.device)

        return intrinsic

class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad, fx_only, order=2, init_focal=None):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.

        if self.fx_only:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
        else:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )

    def forward(self):
        if self.fx_only:
            if self.order == 2:
                fxfy = torch.stack([self.fx ** 2 * self.W, self.fx ** 2 * self.W])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fx * self.W])
        else:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fy * self.H])
        return fxfy

class RaysGenerator:
    def __init__(self, img_lis, msk_lis, pose_net, intrin_net, learnable=False):
        super(RaysGenerator, self).__init__()
        self.pose_net = pose_net
        self.intrin_net = intrin_net
        self.learnable = learnable

        if not learnable:
             self.intrin_inv = torch.inverse(self.intrin_net)
        print('Load data: Begin')
        self.device = torch.device('cuda')

        self.images_lis = img_lis
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([(cv.imread(im_name)) for im_name in self.images_lis]) / 256.0
        if len(msk_lis) != 0:
            self.using_mask = True
            self.masks_lis = msk_lis
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
            self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        else:
            self.using_mask = False

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1, verbose=True, noise=None):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        if self.learnable:
            pose = self.pose_net(img_idx) # 4, 4
            intrinsic_inv = torch.inverse(self.intrin_net())
        else:
            pose = self.pose_net[img_idx]
            intrinsic_inv = self.intrin_inv[img_idx]
        if verbose:
            print(pose)
            print(intrinsic_inv)

        if noise is not None:
            noise_r, noise_t = noise
            noise_r = torch.tensor(noise_r).to(pose.device)
            noise_t = torch.tensor(noise_t).to(pose.device)
            pose[:3, :3] = pose[:3, :3]*(1 + noise_r)
            pose[:3, 3] = pose[:3, 3]*(1 + noise_t*pose[3, 3])

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        if self.using_mask:
            mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        else:
            mask = torch.ones_like(color) * 255
        if self.learnable:
            pose = self.pose_net(img_idx)
            intrinsic_inv = torch.inverse(self.intrin_net())
        else:
            pose = self.pose_net[img_idx] # pose_all
            intrinsic_inv = self.intrin_inv[img_idx]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

    def gen_rays_between(self, ratio, idx_0, idx_1, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        if self.learnable:
            pose_0 = self.pose_net(idx_0).detach().cpu().numpy()
            pose_1 = self.pose_net(idx_1).detach().cpu().numpy()
            intrinsic_inv = torch.inverse(self.intrin_net()).detach().cpu().numpy()
        else:
            pose_0 = self.pose_net[idx_0].detach().cpu().numpy()
            pose_1 = self.pose_net[idx_1].detach().cpu().numpy()
            intrinsic_inv = self.intrin_inv[0].detach().cpu().numpy()
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        intrinsic_inv = torch.tensor(intrinsic_inv)
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3

        trans = pose_0[:3, 3] * (1.0 - ratio) + pose_1[:3, 3] * ratio
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def image_at(self, idx, resolution_level):
        img = self.images_np[idx]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))*255).clip(0, 255)