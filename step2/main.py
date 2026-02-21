import os
import shutil
import torch
import argparse

from transformers import GPT2Tokenizer

from modules.trainer import Trainer
from modules.data_ct import getDatasets
from modules.dataloaders import R2DataLoader
from modules.optimizers import build_lr_scheduler
from modules.utils_dir import parse_yaml
from models.report_generation_model import ReportGenerationModel


def make_output_dir(output_dir):
    '''
    Create folder where results are saved
    '''

    # make directory to save figures, checkpoints
    path_checkpoints      = os.path.join(output_dir, "checkpoints")
    path_generations      = os.path.join(output_dir, "generations")

    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_generations, exist_ok=True)
    
    return output_dir

def get_tokenizer():
    """
    Return GPT-2 tokenizer.
    """
    checkpoint          = args.gpt2
    tokenizer           = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def main(args):

    # directory where results are saved
    args.save_dir = make_output_dir(args.save_dir)

    # save config file in result folder
    shutil.copy(args.yaml_file, os.path.join(args.save_dir, 'config.yaml'))

    # device
    args.device = torch.device(f'{args.device}:{args.device_idx}')

    # create tokenizer
    tokenizer = get_tokenizer()

    # create datasets
    train_ds, valid_ds, test_ds = getDatasets(args, tokenizer)

    # create data loader
    train_dataloader = R2DataLoader(args, train_ds, tokenizer, split='train', shuffle=False)
    valid_dataloader = R2DataLoader(args, valid_ds, tokenizer, split='valid', shuffle=False)
    test_dataloader  = R2DataLoader(args, test_ds,  tokenizer, split='test',  shuffle=False)

    # build model architecture
    model = ReportGenerationModel(args=args)

    # build optimizer, learning rate scheduler
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr_decoder, weight_decay=args.weight_decay_adamw)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    scaler       = torch.cuda.amp.GradScaler()

    # build trainer and start to train
    trainer = Trainer(model, optimizer, scaler, tokenizer, args, lr_scheduler, train_dataloader, valid_dataloader, test_dataloader)

    # train and evaluate
    trainer.train()


if __name__ == '__main__':

    # read config file
    parser = argparse.ArgumentParser(description='Script with parameters from JSON file')
    parser.add_argument('--yaml_file', type=str, default='configs/train/default.yaml', help='YAML file containing parameters')
    args = parser.parse_args()

    params = parse_yaml(args.yaml_file)
    yaml_file = args.yaml_file
    args = argparse.Namespace(**params)
    args.yaml_file = yaml_file
    
    main(args)

