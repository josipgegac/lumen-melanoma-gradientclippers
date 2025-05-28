import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import LumenMelanomaDataset
from src.model import ResNet50
from src.train import train
from src.utils.seed import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Model train script')

    parser.add_argument('seed', type=int, nargs='?', default=10, help='Set seed')
    parser.add_argument('df_path', type=str, nargs='?', default='./processed_data/train_df.feather', help='Path to train dataframe')
    parser.add_argument('save_dir', type=str, nargs='?', default='./models', help='Directory where to save trained model')

    args = parser.parse_args()

    return args

def main(args):
    # Set seed
    set_seed(args.seed)

    # Hyperparameters
    num_epochs = 10

    batch_size = 64

    lr = 0.0005
    weight_decay = 0.00005

    # Focal loss hyperparameters
    alpha = 0.25
    gamma = 2
    reduction = 'mean'

    # Load dataframe
    df = pd.read_feather(args.df_path)

    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2)

    # Create datasets
    train_ds = LumenMelanomaDataset(train_df, train=True)
    test_ds = LumenMelanomaDataset(test_df, train=False)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False)

    # Initialize model
    model = ResNet50(finetune=True)
    model = model.to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize loss function
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = lambda inputs, targets: sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma, reduction=reduction)

    # Train model
    model = train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_epochs, device)

    # Save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, f'resnet50_focal_{alpha}a_{gamma}g.pt')
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    args = parse_args()

    main(args)