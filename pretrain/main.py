import os
import time
import torch
import argparse

from src.models.ctnet import CTNet
from src.optimizer import build_lr_scheduler
from src.utils import make_output_dir, parse_yaml
from src.data.dataset import getDatasets, collate_fn

from torch.utils.data import DataLoader

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        params: dict,
    ) -> None:
        self.logs_, self.metrics_train_, self.metrics_val_ = {}, {}, {}
        self.params = params
        self.device = self.params['device']
        self.model = model.to(self.device)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.sizes = (len(self.train_data), len(self.test_data))
        self.num_batches = len(train_data)
        self.params['size_train'] = len(train_data)
        self.params['size_valid'] = len(valid_data)
        self.params['size_test']  = len(test_data)

    def _run_batch(self, volumes, labels):
        self.optimizer.zero_grad()
        loss = self.model(volumes, labels)
        loss.mean().backward()
        self.optimizer.step()
        loss_value = loss.mean().detach().item()
        return {'loss_train': loss_value}

    def _run_batch_val(self, volumes, labels):
        with torch.no_grad():
            loss = self.model(volumes, labels)
        loss_value = loss.mean().detach().item()
        return {'loss_val': loss_value}

    def _run_batch_test(self, volumes, labels):
        with torch.no_grad():
            loss = self.model(volumes, labels)
        loss_value = loss.mean().detach().item()
        return {'loss_test': loss_value}

    def _run_epoch(self, epoch):
        logs = {}

        # training step
        self.model.train()

        for i, data in enumerate(self.train_data):
            _, volumes, labels = data
            volumes = volumes.to(self.device)
            labels  = labels.to(self.device)
            loss    = self._run_batch(i, epoch, volumes, labels)
            logs    = accum_log(logs, loss)

        self.model.eval()
        # validation
        for i, (_, volumes, labels) in enumerate(self.valid_data):
            volumes = volumes.to(self.device)
            labels  = labels.to(self.device)
            loss    = self._run_batch_val(volumes, labels)
            logs    = accum_log(logs, loss)

        # test
        for i, (_, volumes, labels) in enumerate(self.test_data):
            volumes = volumes.to(self.device)
            labels  = labels.to(self.device)
            loss    = self._run_batch_test(volumes, labels)
            logs    = accum_log(logs, loss)

        self.lr_scheduler.step()
        
        return logs

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = os.path.join(self.params['results']['path_folder'], f"checkpoints/checkpoint_{epoch}.pt")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        print(f'\n>>> Starting to train...\n')
        self.start_epoch = time.time()
        for epoch in range(max_epochs):
            logs = self._run_epoch(epoch)
            self._save_checkpoint(epoch)


def prepare_dataloader(trainset, validset, testset, params):

    train_loader = DataLoader(trainset, batch_size=params['parameters']['batch_size_train'], 
                                num_workers=params['config']['num_workers'], 
                                persistent_workers=params['config']['persistent_workers'], collate_fn=collate_fn)
    
    valid_loader = DataLoader(validset, batch_size=params['parameters']['batch_size_valid'], 
                                num_workers=params['config']['num_workers_valid'], 
                                persistent_workers=params['config']['persistent_workers_val'], collate_fn=collate_fn)
    
    test_loader  = DataLoader(testset, batch_size=params['parameters']['batch_size_test'], 
                                num_workers=params['config']['num_workers_test'], 
                                persistent_workers=params['config']['persistent_workers_test'], collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader

def main(params):
    '''
    Executed function to train the model
    '''

    # device
    params['device'] = torch.device(f'cuda:{params['config']['gpu_index']}')

    # dataloader
    trainset, validset, testset             = getDatasets(params)
    train_loader, valid_loader, test_loader = prepare_dataloader(trainset, validset, testset, params)
    params['data']['num_batches']           = len(train_loader)

    # model
    model = CTNet(params)

    # optimizer, scheduler
    lr           = params['parameters']['lr']
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = build_lr_scheduler(params, optimizer)

    # trainer
    total_epochs = params['parameters']['epochs']
    trainer = Trainer(model, train_loader, valid_loader, test_loader, optimizer, lr_scheduler, params)
    trainer.train(total_epochs)
    

if __name__ == "__main__":
    
    # read args file
    parser = argparse.ArgumentParser(description='Script with parameters from JSON file')
    parser.add_argument('--yaml_file', type=str, default='configs/train/default.yaml', help='YAML file containing parameters')
    args = parser.parse_args()
    params = parse_yaml(args.yaml_file)

    # result directory
    params['results']['path_folder'] = make_output_dir(params['results']['path_folder'])

    # train
    main(params)
