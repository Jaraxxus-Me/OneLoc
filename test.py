from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset import DemoDataset, VideoFrameDataset, BOPDataset
from torch.optim import Adam, SGD, AdamW
import hydra
import cv2
from model import Model
import os
import time
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob
import pickle as pkl
from tqdm import tqdm


def load_intrinsic(name):
    lines = open(name).read().splitlines()
    intrinsic = np.eye(3)
    intrinsic[0, 0] = float(lines[0].split()[-1])
    intrinsic[1, 1] = float(lines[1].split()[-1])
    intrinsic[0, 2] = float(lines[2].split()[-1])
    intrinsic[1, 2] = float(lines[3].split()[-1])
    return intrinsic

def init_fn(i):
    return np.random.seed(torch.initial_seed() % 2 ** 32 - i)

class VideoFramePredictor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Model(self.cfg).cuda()
        np.random.seed(int(round(time.time() * 1000)) % (2**32 - 1))

    def predict(self, batch):
        frames = batch
        prediction = self.model.predict(frames)
        return prediction

@hydra.main(config_path='.', config_name='config_test', version_base='1.2')
def main(cfg):
    dataset = BOPDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    trained_model = os.path.join(cfg.outdir, 'obj_{:06d}'.format(cfg.id), 'lightning_logs/version_0/checkpoints/epoch=49-step=2000.ckpt')
    frame_predictor = VideoFramePredictor(cfg)
    frame_predictor = frame_predictor.load_from_checkpoint(trained_model, cfg=cfg)
    frame_predictor.eval()

    predictions = {}
    scene_id = int(cfg.scene_id)
    out_file = os.path.join(cfg.outdir, '{:02d}_obj_{:06d}.pkl'.format(scene_id, cfg.id))

    if len(dataloader)==0:
        return

    for batch in tqdm(dataloader):
        img_id = int(batch[0][0].data)
        frame = batch[1]
        prediction = frame_predictor.predict(frame)
        predictions[img_id] = prediction
    
    with open(out_file, 'wb') as f:
        pkl.dump(predictions, f)

if __name__ == '__main__':
  main()