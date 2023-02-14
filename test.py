from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset import DemoDataset, VideoFrameDataset
from torch.optim import Adam, SGD, AdamW
import hydra
import cv2
from model import Model
import os
import time
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob


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

@hydra.main(config_path='.', config_name='config', version_base='1.2')
def main(cfg):
    video_path = "data/test.mp4"
    dataset = VideoFrameDataset(video_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    trained_model = cfg.checkpoint  # load your trained PyTorch model
    frame_predictor = VideoFramePredictor(cfg)
    frame_predictor = frame_predictor.load_from_checkpoint(trained_model, cfg=cfg)
    frame_predictor.eval()

    predictions = []

    for batch in dataloader:
        prediction = frame_predictor.predict(batch)
        predictions.append(prediction)

if __name__ == '__main__':
  main()