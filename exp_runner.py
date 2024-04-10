import sys
import cv2
import datetime
import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from pyhocon import ConfigFactory

from models.IoULoss import IoULoss
from models.matching_dataset import MatchingDataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.poses import LearnPose, LearnIntrin, RaysGenerator

class Runner:
    def __init__(self, conf_path, case='CASE_NAME', is_continue=False, latest_ckpt=None):
        self.device = torch.device('cuda')
        # Configuration
        self.conf_path = conf_path
        self.is_continue = is_continue
        dttime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        conf_text = conf_text.replace('DATETIME', dttime)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        print(self.base_exp_dir)
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = MatchingDataset(self.conf['dataset'], exp_dir=self.base_exp_dir)

        self.iter_step = 0
        self.poses_iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.clear_freq = self.conf.get_int('train.clear_freq', default=50000)
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd', default=False)
        self.warm_up_end = self.conf.get_int('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_int('train.anneal_end', default=0.0)

        params_to_train = []
        self.learnable = self.conf.get_bool('train.learnable', default=False)
        self.learn_focal = self.conf.get_bool('model.focal.req_grad', default=False)
        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.model_list = []
        self.writer = None

        self.iou_res = self.conf.get_int('model.iou.resolution', default=64)
        self.iou_topk = self.conf.get_int('model.iou.topk', default=8)
        self.iou_weight = self.conf.get_float('model.iou.weight', default=0.2)

        self.device = self.dataset.device
        self.coarse2fine = self.conf.get_bool('model.c2f.coarse2fine', default=False)
        self.ada_alpha = self.conf.get_float('model.c2f.ada_alpha', default=0.5)
        self.gauss_init_ratio = self.conf.get_float('model.c2f.gauss_init_ratio', default=0.1)
        self.ada_thresh = self.conf.get_float('model.c2f.ada_thresh', default=0.2)
        self.coarse_iter = self.conf.get_int('model.c2f.coarse_iter', default=50000)
        self.c2f_debug = self.conf.get_bool('model.c2f.c2f_debug', default=True) if self.coarse2fine else False
        self.ada_patience = self.conf.get_int('model.c2f.ada_patience', default=5)
        self.level_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.update_confidence = self.conf.get_bool('model.confidence.update', default=True)
        self.confidence_add_init = self.conf.get_bool('model.confidence.add_init', default=True)
        self.increase_confidence = self.conf.get_bool('model.confidence.increase', default=True)
        self.confidence_weight = self.conf.get_float('model.confidence.weight', default=1.)
        self.conf_weight = self.conf.get_float('model.confidence.conf_weight', default=0.0)

        self.last_confidence = None

        self.c2f_debug_dir = os.path.join(self.base_exp_dir, 'c2f_debug')

        if self.coarse2fine:
            self.sigma = max(self.dataset.H, self.dataset.W) * self.gauss_init_ratio
            self.min_sigma = self.sigma
            self.ada_counter = 0
            self.dataset.coarse_flag = True
            self.dataset.gen_coarse_image_ada_down(self.sigma)

            self.save_debug_c2f()
        else:
            self.dataset.coarse_images = self.dataset.images

        if self.learnable:
            self.savepose_freq = self.conf.get_int('train.savepose_freq')
            self.evalpose_freq = self.conf.get_int('train.evalpose_freq')
            self.init_focal = self.conf.get_bool('train.init_focal')
            self.init_poses = self.conf.get_bool('train.init_poses')
            self.focal_lr = self.conf.get_float('train.focal_lr')
            self.pose_lr = self.conf.get_float('train.focal_lr')
            self.focal_lr_gamma = self.conf.get_float('train.focal_lr_gamma')
            self.pose_lr_gamma = self.conf.get_float('train.focal_lr_gamma')
            self.step_size = self.conf.get_int('train.step_size')

            init_focal = self.dataset.focal if self.init_focal else None
            init_poses = self.dataset.pose_all if self.init_poses else None
            # learn focal parameter
            self.intrin_net = LearnIntrin(self.dataset.H, self.dataset.W, **self.conf['model.focal'],
                                          init_focal=init_focal).to(self.device)
            # learn pose for each image
            self.pose_param_net = LearnPose(self.dataset.n_images, **self.conf['model.pose'], init_c2w=init_poses).to(
                self.device)

            self.optimizer_focal = torch.optim.Adam(
                [{'params': self.intrin_net.parameters(), 'initial_lr': self.focal_lr}], lr=self.focal_lr)
            self.optimizer_pose = torch.optim.Adam(
                [{'params': self.pose_param_net.parameters(), 'initial_lr': self.pose_lr}], lr=self.pose_lr)

            self.scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_focal,
                                                                        milestones=(self.warm_up_end, self.end_iter,
                                                                                    self.step_size),
                                                                        gamma=self.focal_lr_gamma,
                                                                        last_epoch=self.poses_iter_step)
            self.scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pose,
                                                                       milestones=range(self.warm_up_end, self.end_iter,
                                                                                        self.step_size),
                                                                       gamma=self.pose_lr_gamma,
                                                                       last_epoch=self.poses_iter_step)
        else:
            self.intrin_net = self.dataset.intrinsics_all
            self.pose_param_net = self.dataset.pose_all

        self.rays_generator = RaysGenerator(self.dataset.images_lis, self.dataset.masks_lis, self.pose_param_net,
                                            self.intrin_net, learnable=self.learnable)
        self.dataset.defineNets(self.pose_param_net, self.intrin_net, learnable=self.learnable)

        # Networks
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if latest_ckpt is not None and (latest_ckpt[-4:] == '.pth' and \
                                        latest_ckpt in os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))):
            is_continue = False
            latest_model_name = latest_ckpt

        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if not self.is_continue:
            self.file_backup()

    def exec_img(self, data, is_training=True):
        rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

        background_rgb = None
        if self.use_white_bkgd:
            background_rgb = torch.ones([1, 3])

        if self.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        mask_sum = mask.sum() + 1e-5
        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio(), is_training=is_training)

        color_fine = render_out['color_fine']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum'] + 1e-5

        # Loss
        color_error = (color_fine - true_rgb) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        eikonal_loss = gradient_error
        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        loss = color_fine_loss + \
               eikonal_loss * self.igr_weight + \
               mask_loss * self.mask_weight

        return {
            'core_pts': render_out['core_pts'],
            'weights': render_out['weights'],
            'loss': loss,
            'color_fine_loss': color_fine_loss.detach(),
            'eikonal_loss': eikonal_loss.detach(),
            's_val': s_val.detach(),
            'cdf_fine': cdf_fine.detach(),
            'mask': mask.detach(),
            'mask_sum': mask_sum.detach(),
            'weight_max': weight_max.detach(),
            'psnr': psnr.detach()
        }

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        pair_perm = self.get_pair_perm()
        balance_perm = self.get_balance_perm(init=True)
        print()

        if self.learnable:
            self.intrin_net.train()
            self.pose_param_net.train()

        flag = self.learnable

        sample_dist = 2.0 / self.renderer.n_samples
        iouloss = IoULoss(sample_dist, resolution=self.iou_res, topk=self.iou_topk)

        perm_idx = 0
        balancing = False

        self.n_latest_iter = self.dataset.n_pairs * 3 // self.dataset.n_images 
        self.confidence_term = np.full((self.dataset.n_images, self.n_latest_iter), np.nan)

        for iter_i in tqdm(range(res_step), ncols=80):
            if self.learnable and self.iter_step % self.clear_freq == 0:
                torch.cuda.empty_cache()

            if balancing and perm_idx == len(balance_perm):
                perm_idx = 0
                balancing = False
                pair_perm = self.get_pair_perm()

            if not balancing and perm_idx == len(pair_perm):
                perm_idx = 0
                balancing = True
                balance_perm = self.get_balance_perm()

            if flag:
                self.pose_param_net.train()
                self.intrin_net.train()
                flag = False
                self.dataset.learnable = True

            if balancing:
                balance_idx = balance_perm[perm_idx]
                img_name = self.balance_images[balance_idx]
                img_idx = self.dataset.name2idx[img_name]

                data = self.rays_generator.gen_random_rays_at(img_idx, self.batch_size)
                img_out = self.exec_img(data)

                loss = img_out['loss']
                psnr = img_out['psnr']
                color_fine_loss = img_out['color_fine_loss']
                eikonal_loss = img_out['eikonal_loss']
                s_val = img_out['s_val']
                weight_max = (img_out['weight_max'] * img_out['mask']).sum() / img_out['mask_sum']
                cdf = (img_out['cdf_fine'][:, :1] * img_out['mask']).sum() / img_out['mask_sum']

                self.confidence_term[img_idx, :-1] = self.confidence_term[img_idx, 1:]
                self.confidence_term[img_idx, -1] = psnr

            else:
                pair_idx = pair_perm[perm_idx]

                data1, data2, n_match, img_idx1, img_idx2 = self.dataset.gen_rays_from_colmap(pair_idx, self.batch_size)

                img1_out = self.exec_img(data1)
                img2_out = self.exec_img(data2)

                pts1 = img1_out['core_pts'][:n_match]
                pts2 = img2_out['core_pts'][:n_match]

                n_samples = self.renderer.n_samples + self.renderer.n_importance
                weights_core1 = img1_out['weights'][:n_match, :n_samples]
                weight_sum_core1 = weights_core1.sum(dim=-1, keepdim=True) + 1e-5
                weights_core1 = weights_core1 / weight_sum_core1
                weights_core2 = img2_out['weights'][:n_match, :n_samples]
                weight_sum_core2 = weights_core2.sum(dim=-1, keepdim=True) + 1e-5
                weights_core2 = weights_core2 / weight_sum_core2

                iou_loss = iouloss(pts1, weights_core1, pts2, weights_core2)
                loss = (img1_out['loss'] + img2_out['loss']) / 2

                if isinstance(iou_loss, int):
                    sumloss = loss + 0.5
                    perm_idx += 1
                    sumloss = sumloss.item()
                    loss = loss.item()
                    continue
                else:
                    sumloss = loss + iou_loss * self.iou_weight

                psnr = (img1_out['psnr'] + img2_out['psnr']) / 2
                color_fine_loss = (img1_out['color_fine_loss'] + img2_out['color_fine_loss']) / 2
                eikonal_loss = (img1_out['eikonal_loss'] + img2_out['eikonal_loss']) / 2
                s_val = torch.cat([img1_out['s_val'], img2_out['s_val']])

                weight_max1 = (img1_out['weight_max'] * img1_out['mask']).sum() / img1_out['mask_sum']
                weight_max2 = (img2_out['weight_max'] * img2_out['mask']).sum() / img2_out['mask_sum']
                weight_max = (weight_max1 + weight_max2) / 2

                cdf1 = (img1_out['cdf_fine'][:, :1] * img1_out['mask']).sum() / img1_out['mask_sum']
                cdf2 = (img2_out['cdf_fine'][:, :1] * img2_out['mask']).sum() / img2_out['mask_sum']
                cdf = (cdf1 + cdf2) / 2

                self.confidence_term[img_idx1, :-1] = self.confidence_term[img_idx1, 1:]
                self.confidence_term[img_idx2, :-1] = self.confidence_term[img_idx2, 1:]
                self.confidence_term[img_idx1, -1] = img1_out['psnr']
                self.confidence_term[img_idx2, -1] = img2_out['psnr']

                del img1_out, img2_out

            self.optimizer.zero_grad()
            if self.learnable:
                self.optimizer_focal.zero_grad()
                self.optimizer_pose.zero_grad()

            if balancing:
                loss.backward()
            else:
                sumloss.backward()

            self.optimizer.step()
            if self.learnable:
                self.optimizer_focal.step()
                self.optimizer_pose.step()
                self.poses_iter_step += 1

            self.iter_step += 1
            perm_idx += 1

            if self.coarse2fine and self.iter_step % self.coarse_iter == 0 and self.dataset.coarse_flag:
                if isinstance(sumloss, float):
                    c2f_loss = sumloss * 20
                else:
                    c2f_loss = sumloss.item() * 20
                if self.ada_counter > self.ada_patience:
                    self.sigma = 0.8 * self.sigma
                else:
                    self.sigma = min(self.sigma, self.ada_alpha * self.sigma + (1 - self.ada_alpha) * c2f_loss)
                    if self.min_sigma - self.sigma > self.ada_thresh:
                        self.ada_counter = 0
                    else:
                        self.ada_counter += 1
                    self.min_sigma = min(self.min_sigma, self.sigma)

                print("sigma updated: {}, current loss: {}".format(self.sigma, c2f_loss))

                self.writer.add_scalar('ada/c2f_loss', c2f_loss, self.iter_step)
                self.writer.add_scalar('ada/sigma', self.sigma, self.iter_step)

                self.dataset.gen_coarse_image_ada_down(self.sigma)

                self.save_debug_c2f()

            self.writer.add_scalar('Loss/loss', loss.item(), self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)

            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', cdf, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', weight_max, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if not balancing and iou_loss != -1:
                self.writer.add_scalar('Loss/iouloss', iou_loss.item(), self.iter_step)
                self.writer.add_scalar('Loss/sumloss', sumloss.item(), self.iter_step)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                res = 512 if self.iter_step % 50000 == 0 else 64
                self.validate_mesh(resolution=res)

            if self.dataset.learnable and self.iter_step % self.savepose_freq == 0:
                self.pose_param_net.eval()
                self.intrin_net.eval()
                self.store_current_pose()
                if self.learnable:
                    self.pose_param_net.train()
                    self.intrin_net.train()

            self.update_learning_rate()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def log(self, x):
        return np.log(x + 1)

    def save_debug_c2f(self, name=None):
        if not self.c2f_debug:
            return
        if name is None:
            cur_dir = self.c2f_debug_dir + '/' + str(self.iter_step)
        else:
            cur_dir = self.c2f_debug_dir + '/' + str(name)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        for i in range(self.dataset.n_images):
            cv2.imwrite(cur_dir + '/{}'.format(os.path.basename(self.dataset.images_lis[i])),
                        self.dataset.coarse_images[i].numpy() * 256.)

    def get_pair_perm(self):
        return torch.randperm(self.dataset.n_pairs)

    def get_balance_perm(self, init=False):
        if not self.dataset.with_confidence:
            self.balance_images = self.dataset.balance_images
        elif self.update_confidence and not init:
            confidence = np.nanmean(self.confidence_term, axis=1)
            weight = self.iter_step / self.end_iter * self.confidence_weight if self.increase_confidence else -1
            self.balance_images = self.dataset.find_balance_images_by_confidence(confidence, add_init=self.confidence_add_init,
                                                                                 weight=weight, conf_weight=self.conf_weight)
        elif self.increase_confidence and not init:
            confidence = self.dataset.init_confidence
            weight = self.iter_step / self.end_iter
            self.balance_images = self.dataset.find_balance_images_by_confidence(confidence, add_init=False,
                                                                                 weight=weight * self.confidence_weight, conf_weight=self.conf_weight)
        else:
            self.balance_images = self.dataset.balance_images

        return torch.randperm(len(self.balance_images))

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

        if self.learnable:
            self.scheduler_focal.step()
            self.scheduler_pose.step()

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        base_folder = os.path.dirname(sys.argv[0])
        print('[Debug] base_folder: ', base_folder)
        for dir_name in dir_lis:
            source_dir = os.path.join(base_folder, dir_name)
            print('[Info file_backup]', source_dir)
            if os.path.isfile(source_dir):
                cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
                if dir_name[-3:] == '.py':
                    copyfile(source_dir, cur_dir)
                continue
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(source_dir)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(source_dir, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        with open(os.path.join(self.base_exp_dir, 'recording', 'config.conf'), 'a+') as f:
            f.seek(0)
            f.write('# ' + ' '.join(sys.argv) + '\n')

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        if self.learnable:
            self.load_pnf_checkpoint(checkpoint_name.replace('ckpt', 'pnf'))
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
        if self.learnable:
            self.save_pnf_checkpoint()

    def load_pnf_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'pnf_checkpoints', checkpoint_name),
                                map_location=self.device)
        self.intrin_net.load_state_dict(checkpoint['intrin_net'])
        self.pose_param_net.load_state_dict(checkpoint['pose_param_net'])
        self.optimizer_focal.load_state_dict(checkpoint['optimizer_focal'])
        self.optimizer_pose.load_state_dict(checkpoint['optimizer_pose'])
        self.poses_iter_step = checkpoint['poses_iter_step']
        self.scheduler_pose.load_state_dict(checkpoint['scheduler_pose'])
        self.scheduler_focal.load_state_dict(checkpoint['scheduler_focal'])

    def save_pnf_checkpoint(self):
        pnf_checkpoint = {
            'intrin_net': self.intrin_net.state_dict(),
            'pose_param_net': self.pose_param_net.state_dict(),
            'optimizer_focal': self.optimizer_focal.state_dict(),
            'optimizer_pose': self.optimizer_pose.state_dict(),
            'poses_iter_step': self.poses_iter_step,
            'scheduler_pose': self.scheduler_pose.state_dict(),
            'scheduler_focal': self.scheduler_focal.state_dict(),
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'pnf_checkpoints'), exist_ok=True)
        torch.save(pnf_checkpoint,
                   os.path.join(self.base_exp_dir, 'pnf_checkpoints', 'pnf_{:0>6d}.pth'.format(self.iter_step)))

    def get_current_pose(self):
        if self.learnable:
            num_cams = self.pose_param_net.module.num_cams if isinstance(self.pose_param_net,
                                                                        torch.nn.DataParallel) else self.pose_param_net.num_cams
            c2w_list = []
            for i in range(num_cams):
                c2w = self.pose_param_net(i)  # (4, 4)
                c2w_list.append(c2w)
            c2w_list = torch.stack(c2w_list)  # (N, 4, 4)
            pose_cur = c2w_list[:, :3, :]

            intr_list = []
            for i in range(num_cams):
                intr = self.intrin_net(i)  # (4, 4)
                intr_list.append(intr)
            intr_list = torch.stack(intr_list)  # (N, 4, 4)
            hwf_cur = intr_list[torch.arange(num_cams).repeat(3,1).T.reshape(1,-1), [1,0,0]*num_cams, [2,2,0]*num_cams].reshape(-1, 3).unsqueeze(-1)
            hwf_cur[:, :2, :] = hwf_cur[:, :2, :] * 2
            para_cur = torch.cat([pose_cur, hwf_cur], dim=-1)
        else:
            c2w_list = self.dataset.pose_all
            pose_cur = c2w_list[:, :3, :]
            intr_list = self.dataset.intrinsics_all  # (N, 4, 4)
            num_cams = c2w_list.shape[0]
            hwf_cur = intr_list[torch.arange(num_cams).repeat(3,1).T.reshape(1,-1), [1,0,0]*num_cams, [2,2,0]*num_cams].reshape(-1, 3).unsqueeze(-1)
            hwf_cur[:, :2, :] = hwf_cur[:, :2, :] * 2
            para_cur = torch.cat([pose_cur, hwf_cur], dim=-1)

        return para_cur

    def store_current_pose(self):
        if not os.path.exists(os.path.join(self.base_exp_dir, 'cam_poses')):
            os.makedirs(os.path.join(self.base_exp_dir, 'cam_poses'), exist_ok=True)
        # self.pose_param_net.eval()
        num_cams = self.pose_param_net.module.num_cams if isinstance(self.pose_param_net,
                                                                     torch.nn.DataParallel) else self.pose_param_net.num_cams

        c2w_list = []
        for i in range(num_cams):
            c2w = self.pose_param_net(i)  # (4, 4)
            c2w_list.append(c2w)

        c2w_list = torch.stack(c2w_list)  # (N, 4, 4)
        c2w_list = c2w_list.detach().cpu().numpy()
        np.save(os.path.join(self.base_exp_dir, 'cam_poses', 'pose_{:0>6d}.npy'.format(self.iter_step)), c2w_list)

        if self.learn_focal:
            intr_list = []
            for i in range(num_cams):
                intr = self.intrin_net(i)  # (4, 4)
                intr_list.append(intr)
            intr_list = torch.stack(intr_list)  # (N, 4, 4)
            intr_list = intr_list.detach().cpu().numpy()

            np.save(os.path.join(self.base_exp_dir, 'cam_poses', 'intrin_{:0>6d}.npy'.format(self.iter_step)),
                    intr_list)
        return

    def validate_image(self, idx=-1, resolution_level=-1, out_dir=None):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        rays_o, rays_d = self.rays_generator.gen_rays_at(idx, resolution_level=resolution_level)
        rays_o = rays_o.detach()
        rays_d = rays_d.detach()
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            # near = near.detach()
            # far = far.detach()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              is_training=False)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            if self.learnable:
                rot = np.linalg.inv(self.pose_param_net(idx)[:3, :3].detach().cpu().numpy())
            else:
                rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        if out_dir == None:
            out_dir = self.base_exp_dir
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv2.imwrite(os.path.join(out_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.rays_generator.image_at(idx,
                                                                        resolution_level=resolution_level)]).astype(
                               np.uint8)
                           )
            if len(out_normal_fine) > 0:
                cv2.imwrite(os.path.join(out_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])
        del img_fine, normal_img

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = np.concatenate([vertices, np.ones_like(vertices[:, 0:1])], axis=1)
            vertices = vertices.transpose()
            vertices = self.dataset.scale_mats_np[0] @ vertices
            vertices = vertices.transpose()[:, :3]

            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_aligned.ply'.format(self.iter_step)))
        else:
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/teeth.conf')
    parser.add_argument('--case', type=str, default='', required=True)
    parser.add_argument('-c', '--is_continue', default=False, action="store_true")
    parser.add_argument('--ckpt', type=str, nargs='+', default=None)
    parser.add_argument('--gpu', type=int, default='0')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.case, is_continue=args.is_continue, latest_ckpt=args.ckpt)
    runner.train()
    
    print('Done!')