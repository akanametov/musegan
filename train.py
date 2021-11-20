"""Train script for MuseGAN."""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from musegan import MuseGAN
from data.utils import MidiDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="top", description="Train MusaGAN.")
    parser.add_argument("--path", type=str, default="data/chorales/Jsb16thSeparated.npz", help="Path to dataset.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--z_dimension", type=int, default=32, help="Z(noise)-space dimension.")
    parser.add_argument("--g_channels", type=int, default=1024, help="Generator hidden channels.")
    parser.add_argument("--g_features", type=int, default=1024, help="Generator hidden features.")
    parser.add_argument("--g_lr", type=float, default=0.001, help="Generator learning rate.")
    parser.add_argument("--c_channels", type=int, default=128, help="Critic hidden channels.")
    parser.add_argument("--c_features", type=int, default=1024, help="Critic hidden features.")
    parser.add_argument("--c_lr", type=float, default=0.001, help="Critic learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")
    args = parser.parse_args()
    # parameters of musegan
    gan_args = args.__dict__.copy()
    gan_args.pop("path", None)
    gan_args.pop("epochs", None)
    gan_args.pop("batch_size", None)
    # train
    print("Loading dataset ...")
    dataset = MidiDataset(path=args.path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print("Loading model ...")
    musegan = MuseGAN(**gan_args)
    print("Start training ...")
    musegan.train(dataloader=dataloader, epochs=args.epochs)
    print("Training finished.")
