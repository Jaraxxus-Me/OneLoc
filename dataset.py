from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import zmq
from superpoint import sample_descriptors
import pickle
import torch
import os
from pycocotools.coco import COCO

class DemoDataset(Dataset):
    def __init__(self, cfg, is_training=True):
        super().__init__()
        self.cfg = cfg
        self.is_training = is_training
        self.features = dict()
        self.augs = A.Compose([
            A.GaussianBlur(blur_limit=(1, 3)),
            # A.GaussNoise(),
            # A.ISONoise(always_apply=False, p=0.5),                                                      
            # A.RandomBrightnessContrast(),
        ])
        self.context = None
        self.dataset = cfg.outdir.split('/')[-2]
        self.obj = cfg.outdir.split('/')[-1]
        self.video = os.path.join('data/BOP', self.dataset, 'ref_video', '{}.mp4'.format(self.obj))
        self.mask_video = os.path.join('data/BOP', self.dataset, 'ref_video_mask', '{}.mp4'.format(self.obj))
        vidcap = cv2.VideoCapture(self.video)
        vidcap_mask = cv2.VideoCapture(self.mask_video)
        self.imgs = []
        self.mask = []
        while True:  
            success, image = vidcap.read()
            success_mask, mask = vidcap_mask.read()
            if not success:
                break
            self.imgs.append(image[..., ::-1])
            self.mask.append(mask[..., ::-1])
        
            
    def __getitem__(self, idx):
        if self.context is None:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect('tcp://localhost:5555')

        rgb = self.imgs[idx]
        if self.is_training:
            rgb = self.augs(image=rgb)['image']

        mask = self.mask[idx]
        # mask to box:
        x, y = np.where(mask[:,:,0] != 0)
        
        box_bounds = np.stack([np.array([np.min(y), np.min(x)]), np.array([np.max(y), np.max(x)])]).astype(int)
        
        center2d = np.mean(box_bounds, 0)
        size = np.array([box_bounds[1, 0] - box_bounds[0, 0], box_bounds[1, 1] - box_bounds[0, 1]])
        
        self.socket.send(pickle.dumps(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
        res = pickle.loads(self.socket.recv())
        
        descs = res['raw_descs']
        kps = np.stack(np.meshgrid(np.arange(box_bounds[0, 0], box_bounds[1, 0]), np.arange(box_bounds[0, 1], box_bounds[1, 1])), -1).reshape(-1, 2)
        sub_idx = np.random.randint(kps.shape[0], size=(self.cfg.num_samples,))
        kps = kps[sub_idx]

        offsets = center2d - kps  # N x 2
        offsets /= (np.linalg.norm(offsets, axis=-1, keepdims=True) + 1e-9)
        
        rel_size = np.abs(center2d - kps) / size * 10
        feats = sample_descriptors(torch.from_numpy(kps).float()[None], torch.from_numpy(np.moveaxis(descs, -1, 0)[None]))[0].numpy().T
        
        
        return {
            'rgbs': (np.moveaxis(rgb, [0, 1, 2], [1, 2, 0]) / 255.).astype(np.float32),
            'point_kps': kps.astype(np.float32),
            'point_feats': feats.astype(np.float32),
            'rel_size': rel_size.astype(np.float32),
            'offsets': offsets.astype(np.float32),
        }
    
    def __len__(self):
        return len(self.imgs)

class VideoFrameDataset(Dataset):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        
    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_tensor = torch.from_numpy(frame.transpose((2, 0, 1))).float() / 255.0
            frame = torch.from_numpy((cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255.).astype(np.float32))
            return frame
        else:
            raise ValueError(f"Unable to read frame {idx} from video {self.video_path}")

LMO_OBJECT = ('1', '5', '6', '8', '9',
            '10', '11', '12')
YCB_CLASSES = ('1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15', '16', '17', '18', '19',
            '20', '21')

class BOPDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        test_anno = os.path.join(cfg.test_imgs, '{:06d}'.format(cfg.scene_id), 'scene_gt_coco_ins.json')
        self.coco = COCO(test_anno)

        self.img_ids = self.coco.getImgIds(catIds=[cfg.id])
        self.data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            anns = self.coco.getAnnIds(info['id'])
            for i, ann in enumerate(self.coco.loadAnns(anns)):
                ann['filename'] = info['file_name']
                self.data_infos.append(ann)
        
        self.img_path = os.path.join(cfg.test_imgs, '{:06d}'.format(cfg.scene_id))
        
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        img_info = self.data_infos[idx]
        img_path = os.path.join(self.img_path, img_info['filename'])
        frame = cv2.imread(img_path)
        frame = torch.from_numpy((cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255.).astype(np.float32))
        img_id = img_info['image_id']
        return img_id, frame