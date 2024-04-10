import os
import cv2
import torch
import numpy as np

from utils.ColmapData import ColmapData
from models.dataset import Dataset

class MatchingDataset(Dataset):
    def __init__(self, conf, exp_dir=''):
        super(MatchingDataset, self).__init__(conf)
        self.n_match = self.conf.get_float('n_match', default=16)
        self.pair_thresh = self.conf.get_int('pair_thresh', default=-1)
        self.with_confidence = self.conf.get_bool('with_confidence', default=True)
        self.exp_dir = exp_dir

        self.get_idx_from_name()
        self.init_confidence = None

        # load colmap database
        self.db_file = os.path.join(self.conf['data_dir'], self.conf['colmap_db'])
        cam_path = os.path.join(self.data_dir, self.render_cameras_name)
        self.colmap = ColmapData(self.db_file, self.n_images, self.images_lis, thresh=self.pair_thresh, cam_path=cam_path)
        self.colmap.exec()
        self.n_pairs = len(self.colmap.img_pair_list)
        self.get_image_idx()
        # dataset idx -> matches counts
        matching_cnt = [0 for i in range(self.n_images)]
        image_cnt = [0 for i in range(self.n_images)]
        for i, j in self.colmap.img_pair_list:
            matching_cnt[self.image_idx_trans[i]] += len(self.colmap.matches[(i, j)])
            matching_cnt[self.image_idx_trans[j]] += len(self.colmap.matches[(i, j)])
            image_cnt[self.image_idx_trans[i]] += 1
            image_cnt[self.image_idx_trans[j]] += 1
        self.matching_cnt = np.array(matching_cnt)
        self.image_cnt = np.array(image_cnt)
        if self.with_confidence:
            self.balance_images = self.find_balance_images_by_confidence(self.matching_cnt / self.image_cnt, init=True)
        else:
            self.find_balance_images()

        self.balance_images = self.find_balance_images_by_confidence(self.matching_cnt / self.image_cnt, init=True)
        print('Done')

    def find_balance_images(self):
        self.balance_images = []
        max_value = max(self.image_cnt)
        cnt = max_value - self.image_cnt
        if cnt.sum() == 0:
            cnt += 1
        for i in range(self.n_images):
            name = os.path.basename(self.images_lis[i])
            self.balance_images.extend([name] * cnt[i])
    
    def find_balance_images_by_confidence(self, confidence, init=False, add_init=True,
                                          weight=-1, conf_weight=0.1):
        confidence[np.isnan(confidence)] = 0
        confidence = (confidence - np.min(confidence)) / (np.max(confidence) - np.min(confidence) + 1e-5) # normalize

        if init:
            self.init_confidence = confidence
        elif add_init and self.init_confidence is not None:
            confidence = confidence * conf_weight + self.init_confidence
            confidence = (confidence - np.min(confidence)) / (np.max(confidence) - np.min(confidence)) # re-normalize

        if weight > 0:
            # change weight by iteration
            k = 1 + 2 * weight
            confidence = k * confidence - weight
        confidence = 1 / (1 + np.exp(-confidence)) # sigmoid

        total_cnt = len(self.colmap.img_pair_list) * 3
        balance_cnt = confidence / np.sum(confidence) * total_cnt - self.image_cnt
        balance_cnt[balance_cnt < 0] = 0
        balance_images = []
        for img in self.images_lis:
            name = os.path.basename(img)
            idx = self.name2idx[name]
            balance_images.extend([name] * int(balance_cnt[idx]))

        return balance_images

    def defineNets(self, pose_net=None, intrin_net=None, learnable=False):
        self.learnable = learnable
        self.pose_net = pose_net
        self.intrin_net = intrin_net

    def get_image_idx(self):
        # transfer image idx from colmap to neus
        self.image_idx_trans = {}
        for i, image in enumerate(self.images_lis):
            colmap_idx = self.colmap.name2id[os.path.basename(image)]
            self.image_idx_trans[colmap_idx] = i

    def get_idx_from_name(self):
        self.name2idx = {}
        for i, img in enumerate(self.images_lis):
            self.name2idx[os.path.basename(img)] = i

    def gen_coarse_image_ada_down(self, sigma):
        if sigma <= 1:
            self.coarse_images = self.images
            self.coarse_flag = False
        else:
            scale = 1 / sigma # 0-1
            self.coarse_images = []
            for idx in range(self.n_images):
                if self.data_type == 'tank':
                    img = cv2.imread(self.images_lis[idx]) / 256.0
                else:
                    img = self.images_np[idx]
                gauss_image = cv2.GaussianBlur(img, (0, 0), sigma)
                downsampled_image = self.resize(gauss_image, scale)
                upsampled_image = cv2.resize(downsampled_image, self.image_size, interpolation=cv2.INTER_AREA)
                self.coarse_images.append(upsampled_image)
            self.coarse_images = torch.from_numpy(np.stack(self.coarse_images).astype(np.float32)).cpu()
    
    def resize(self, img, ratio):
        size = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img
    
    def gen_rays_from_colmap(self, pair_idx, batch_size):
        img_pair = self.colmap.img_pair_list[pair_idx] # colmap index
        match = self.colmap.matches[img_pair]

        n_match = int(min(self.n_match, match.shape[0]))
        n_random = batch_size - n_match
        if 0 < n_match < match.shape[0]:
            match = match[np.random.choice(match.shape[0], n_match, replace=False)] # ndarray: n, 2
        random_pt_x1 = torch.randint(low=0, high=self.W, size=[n_random])
        random_pt_y1 = torch.randint(low=0, high=self.H, size=[n_random])
        random_pt_x2 = torch.randint(low=0, high=self.W, size=[n_random])
        random_pt_y2 = torch.randint(low=0, high=self.H, size=[n_random])

        keypoints1 = self.colmap.keypoints[img_pair[0]].astype(int) # ndarray: n, 2
        keypoints2 = self.colmap.keypoints[img_pair[1]].astype(int)

        # idx in dataset
        img_idx1 = self.image_idx_trans[img_pair[0]]
        img_idx2 = self.image_idx_trans[img_pair[1]]

        if n_match > 0:
            match_pt1 = torch.from_numpy(keypoints1[match[:, 0]]).cuda()
            match_pt2 = torch.from_numpy(keypoints2[match[:, 1]]).cuda()
            pixels_x1 = torch.cat([match_pt1[:, 0], random_pt_x1])
            pixels_y1 = torch.cat([match_pt1[:, 1], random_pt_y1])
            pixels_x2 = torch.cat([match_pt2[:, 0], random_pt_x2])
            pixels_y2 = torch.cat([match_pt2[:, 1], random_pt_y2])
        else:
            pixels_x1 = random_pt_x1
            pixels_y1 = random_pt_y1
            pixels_x2 = random_pt_x2
            pixels_y2 = random_pt_y2

        color1 = self.coarse_images[img_idx1][(pixels_y1, pixels_x1)]    # batch_size, 3
        if self.using_mask:
            mask1 = self.masks[img_idx1][(pixels_y1, pixels_x1)]      # batch_size, 3
        else:
            mask1 = torch.ones_like(color1) * 255

        if self.learnable:
            pose1 = self.pose_net(img_idx1)
            intrinsic_inv = torch.inverse(self.intrin_net())
            p1 = torch.stack([pixels_x1, pixels_y1, torch.ones_like(pixels_y1)], dim=-1).float()  # batch_size, 3
            p1 = torch.matmul(intrinsic_inv[None, :3, :3], p1[:, :, None]).squeeze()  # batch_size, 3
            rays_v1 = p1 / torch.linalg.norm(p1, ord=2, dim=-1, keepdim=True)  # batch_size, 3
            rays_v1 = torch.matmul(pose1[None, :3, :3], rays_v1[:, :, None]).squeeze()  # batch_size, 3
            rays_o1 = pose1[None, :3, 3].expand(rays_v1.shape)  # batch_size, 3
        else:
            p1 = torch.stack([pixels_x1, pixels_y1, torch.ones_like(pixels_y1)], dim=-1).float()  # batch_size, 3
            p1 = torch.matmul(self.intrinsics_all_inv[img_idx1, None, :3, :3], p1[:, :, None]).squeeze() # batch_size, 3
            rays_v1 = p1 / torch.linalg.norm(p1, ord=2, dim=-1, keepdim=True)    # batch_size, 3
            rays_v1 = torch.matmul(self.pose_all[img_idx1, None, :3, :3], rays_v1[:, :, None]).squeeze()  # batch_size, 3
            rays_o1 = self.pose_all[img_idx1, None, :3, 3].expand(rays_v1.shape) # batch_size, 3

        color2 = self.coarse_images[img_idx2][(pixels_y2, pixels_x2)]    # batch_size, 3
        if self.using_mask:
            mask2 = self.masks[img_idx2][(pixels_y2, pixels_x2)]      # batch_size, 3
        else:
            mask2 = torch.ones_like(color2) * 255

        if self.learnable:
            pose2 = self.pose_net(img_idx2)
            intrinsic_inv = torch.inverse(self.intrin_net())
            p2 = torch.stack([pixels_x2, pixels_y2, torch.ones_like(pixels_y2)], dim=-1).float()  # batch_size, 3
            p2 = torch.matmul(intrinsic_inv[None, :3, :3], p2[:, :, None]).squeeze()  # batch_size, 3
            rays_v2 = p2 / torch.linalg.norm(p2, ord=2, dim=-1, keepdim=True)  # batch_size, 3
            rays_v2 = torch.matmul(pose2[None, :3, :3], rays_v2[:, :, None]).squeeze()  # batch_size, 3
            rays_o2 = pose2[None, :3, 3].expand(rays_v2.shape)  # batch_size, 3
        else:
            p2 = torch.stack([pixels_x2, pixels_y2, torch.ones_like(pixels_y2)], dim=-1).float()  # batch_size, 3
            p2 = torch.matmul(self.intrinsics_all_inv[img_idx2, None, :3, :3], p2[:, :, None]).squeeze() # batch_size, 3
            rays_v2 = p2 / torch.linalg.norm(p2, ord=2, dim=-1, keepdim=True)    # batch_size, 3
            rays_v2 = torch.matmul(self.pose_all[img_idx2, None, :3, :3], rays_v2[:, :, None]).squeeze()  # batch_size, 3
            rays_o2 = self.pose_all[img_idx2, None, :3, 3].expand(rays_v2.shape) # batch_size, 3

        data1 = torch.cat([rays_o1.cpu(), rays_v1.cpu(), color1, mask1[:, :1]], dim=-1).cuda() # batch_size, 10
        data2 = torch.cat([rays_o2.cpu(), rays_v2.cpu(), color2, mask2[:, :1]], dim=-1).cuda() # batch_size, 10

        return data1, data2, n_match, img_idx1, img_idx2

    def __getitem__(self, idx):
        return self.images[idx].permute(2, 0, 1), self.coarse_images[idx].permute(2, 0, 1)

    def __len__(self):
        return self.n_images




