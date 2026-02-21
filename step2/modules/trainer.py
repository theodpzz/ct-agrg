import os
import csv
import json
import torch
from numpy import inf

from tqdm import tqdm
from abc import abstractmethod

class BaseTrainer(object):
    def __init__(self, model, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.args.accumulation_steps = self.args.batch_size_sim / self.args.batch_size_train
        self.losses = []

        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.path_checkpoints      = os.path.join(args.save_dir, 'checkpoints')
        self.path_generations      = os.path.join(args.save_dir, 'generations')
        self.path_results          = args.save_dir

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        losses = {}
        for epoch in range(self.start_epoch, self.epochs):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            losses[epoch] = log

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
                self._save_losses(losses)

    def _save_losses(self, losses):
        filename = os.path.join(self.path_results, f'losses.json')
        with open(filename, 'w') as json_file:
                json.dump(losses, json_file)

    def _prepare_device(self, args):
        n_gpu = torch.cuda.device_count()
        if args.n_gpu > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            args.n_gpu = 0
        if args.n_gpu > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    args.n_gpu, n_gpu))
            args.n_gpu = n_gpu
        device = torch.device(f'cuda:{str(args.device_idx)}' if args.n_gpu > 0 else 'cpu')
        list_ids = list(range(args.n_gpu))
        return device, list_ids

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.path_checkpoints, f'current_checkpoint_{epoch}.pth')
        torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))



class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, scaler, tokenizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, optimizer, args)
        self.lr_scheduler     = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader   = val_dataloader
        self.test_dataloader  = test_dataloader
        self.scaler           = scaler
        self.tokenizer        = tokenizer

    def _train_epoch(self, epoch):
        
        train_loss = 0

        # Train
        self.model.train()
        with tqdm(total=len(self.train_dataloader)) as pbar:
            self.optimizer.zero_grad()

            # iterate over train samples
            for batch_idx, (images_id, labels, volumes, reports_ids, reports_masks) in enumerate(self.train_dataloader):
                
                volumes       = volumes.to(self.args.device)
                labels        = labels.to(self.args.device)
                reports_ids   = reports_ids.to(self.args.device)
                reports_masks = reports_masks.to(self.args.device)

                # forward pass
                loss = self.model(images_id, labels, volumes, reports_ids, reports_masks)

                # gradient accumulation
                if loss != -1:
                    train_loss += loss.mean().item()
                    loss = loss.mean() / self.args.accumulation_steps
                    self.scaler.scale(loss).backward()

                # step optimizer
                if ((batch_idx + 1) % self.args.accumulation_steps == 0) or (batch_idx==len(self.train_dataloader)):
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.update(1)

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # Validation: Generate reports
        self.model.eval()
        with torch.no_grad():
           val_gts, val_res, val_ids = [], [], []
           path_ground_truths = f"{self.path_generations}/val_ground_truths_epoch_{str(epoch)}.csv"
           path_generations   = f"{self.path_generations}/val_generations_epoch_{str(epoch)}.csv"
           path_ids           = f"{self.path_generations}/val_ids_epoch_{str(epoch)}.csv"

           with open(path_ground_truths, "w", newline = "") as gtss:
            with open(path_generations, "w" , newline = "") as ress:
             with open(path_ids, "w", newline = "") as idss:
                with tqdm(total=len(self.val_dataloader)) as pbar:

                    # iterate over samples
                    for batch_idx, (images_id, labels, volumes, reports_ids, reports_masks) in enumerate(self.val_dataloader):

                            # to device
                            volumes       = volumes.to(self.args.device)
                            labels        = labels.to(self.args.device)
                            reports_ids   = reports_ids.to(self.args.device)
                            reports_masks = reports_masks.to(self.args.device)

                            # extract generated report as a string
                            generated_report = self.model.generate(
                                tokenizer            = self.tokenizer, 
                                volumes              = volumes, 
                                max_length           = self.args.max_seq_length,
                                num_beams            = self.args.beam_size, 
                                num_beam_groups      = self.args.group_size,
                                do_sample            = self.args.do_sample, 
                                num_return_sequences = self.args.num_return_sequences,
                                early_stopping       = self.args.early_stopping
                            )
                            
                            # extract true report as a string
                            ground_truths = self.tokenizer.batch_decode(reports_ids[0].cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            ground_truths = ' '.join(s for s in ground_truths if s)

                            # write it
                            val_res.extend(generated_report)
                            val_gts.extend(ground_truths)
                            val_ids.extend(images_id[0])
                            gt_writer  = csv.writer(gtss)
                            gen_writer = csv.writer(ress)
                            ids_writer = csv.writer(idss)
                            gt_writer.writerow([str(ground_truths)])
                            gen_writer.writerow([str(generated_report)])
                            ids_writer.writerow([str(images_id[0])])

                            pbar.update(1)

            gtss.close()
            ress.close()
            idss.close()

        # Test: Generate reports
        with torch.no_grad():
           val_gts, val_res, val_ids = [], [], []
           path_ground_truths = f"{self.path_generations}/test_ground_truths_epoch_{str(epoch)}.csv"
           path_generations   = f"{self.path_generations}/test_generations_epoch_{str(epoch)}.csv"
           path_ids           = f"{self.path_generations}/test_ids_epoch_{str(epoch)}.csv"

           with open(path_ground_truths, "w", newline = "") as gtss:
            with open(path_generations, "w" , newline = "") as ress:
             with open(path_ids, "w", newline = "") as idss:
                with tqdm(total=len(self.test_dataloader)) as pbar:

                    # iterate over samples
                    for batch_idx, (images_id, labels, volumes, reports_ids, reports_masks) in enumerate(self.test_dataloader):

                            # to device
                            volumes       = volumes.to(self.args.device)
                            labels        = labels.to(self.args.device)
                            reports_ids   = reports_ids.to(self.args.device)
                            reports_masks = reports_masks.to(self.args.device)
                                    
                            # extract generated report as a string
                            generated_report = self.model.generate(
                                tokenizer            = self.tokenizer, 
                                volumes              = volumes,
                                max_length           = self.args.max_seq_length,
                                num_beams            = self.args.beam_size, 
                                num_beam_groups      = self.args.group_size,
                                do_sample            = self.args.do_sample, 
                                num_return_sequences = self.args.num_return_sequences,
                                early_stopping       = self.args.early_stopping
                            )
                            
                            # extract true report as a string
                            ground_truths = self.tokenizer.batch_decode(reports_ids[0].cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            ground_truths = ' '.join(s for s in ground_truths if s)

                            # write it
                            val_res.extend(generated_report)
                            val_gts.extend(ground_truths)
                            val_ids.extend(images_id[0])
                            gt_writer  = csv.writer(gtss)
                            gen_writer = csv.writer(ress)
                            ids_writer = csv.writer(idss)
                            gt_writer.writerow([str(ground_truths)])
                            gen_writer.writerow([str(generated_report)])
                            ids_writer.writerow([str(images_id[0])])

                            pbar.update(1)
                            
            gtss.close()
            ress.close()
            idss.close()

        
        self.lr_scheduler.step()

        return log
    

