import os
import numpy as np
import cv2

from models.dataset import load_K_Rt_from_P
from utils.database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids

class ColmapData:
    def __init__(self, db_file, n_images=0, image_list=[], thresh=-1, cam_path=None):
        super(ColmapData, self).__init__()
        self.db_file = db_file
        self.n_images = n_images
        if image_list:
            self.n_images = len(image_list)
        self.name_list = [os.path.basename(image) for image in image_list]
        self.thresh = thresh
        self.cam_path = cam_path

    def exec(self):
        db = COLMAPDatabase.connect(self.db_file)
        self.get_matches(db)
        db.close()

    def get_matches(self, db):
        # id2name
        self.image_dict = dict(
            (image_id, name) for image_id, name in db.execute("SELECT image_id, name FROM images")
        )

        self.name2id = {}
        for k, v in self.image_dict.items():
            self.name2id[v] = k

        self.keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 2)))
            for image_id, data in db.execute(
                "SELECT image_id, data FROM keypoints"))

        self.matches = dict(
            (pair_id_to_image_ids(pair_id),
             blob_to_array(data, np.uint32, (-1, 2)))
            for pair_id, data in db.execute("SELECT pair_id, data FROM matches where rows <> 0")
        )

        if self.name_list:
            self.keypoints = {key: self.keypoints[key] for key in self.keypoints if self.image_dict[key] in self.name_list}
            self.matches = {key: self.matches[key] for key in self.matches
                            if self.image_dict[key[0]] in self.name_list and self.image_dict[key[1]] in self.name_list}

        self.img_pair_list = list(self.matches.keys())

        if self.cam_path is not None and self.thresh > 0 and self.name_list:
            cam = self.load_cam(self.cam_path, self.name_list)
            cam_name2idx = {name: idx for idx, name in enumerate(self.name_list)}
            thresh_pairs = []
            for pair in self.img_pair_list:
                idx0 = cam_name2idx[self.image_dict[pair[0]]]
                idx1 = cam_name2idx[self.image_dict[pair[1]]]
                R0 = cam[idx0, :3, :3]
                R1 = cam[idx1, :3, :3]
                cos = np.clip((np.trace(np.dot(R0.T, R1)) - 1) / 2, -1., 1.)
                e_R = np.rad2deg(np.abs(np.arccos(cos)))
                if round(e_R, 0) <= self.thresh:
                    thresh_pairs.append(pair)
            self.img_pair_list = thresh_pairs

    def resize(self, img, ratio):
        size = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    def load_cam(self, data_dir, images_list=None):
        camera_dict = np.load(data_dir)
        images_len = len(images_list)
        if 'dtu' in data_dir or 'DTU' in data_dir or 'llff' in data_dir:
            world_mats_np = [camera_dict['world_mat_{}'.format(idx)].astype(np.float32) for idx in range(images_len)]
        else:
            world_mats_np = [camera_dict['world_mat_{}'.format(name[:-4])].astype(np.float32) for name in images_list]

        para_all = []
        for world_mat in world_mats_np:
            intrinsics, pose = load_K_Rt_from_P(None, world_mat[:3, :4])
            hwf = np.array([intrinsics[1][2] * 2, intrinsics[0][2] * 2, intrinsics[0][0]]).reshape([3, 1])
            pose = pose[:3, :]
            para = np.hstack([pose, hwf])
            para_all.append(para)

        para_all = np.array(para_all)

        return para_all

