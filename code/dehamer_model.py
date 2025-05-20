#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.optim import Adam, lr_scheduler
from  dehazenet import UNet_emb
from  utilss import *
import os
import json
import torchvision.utils as vutils
from torchvision.models import vgg16
from perceptual import LossNetwork
from  ECLoss import *
torch.backends.cudnn.enabled = False

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
for param in vgg_model.parameters():
     param.requires_grad = False

loss_network = LossNetwork(vgg_model).cuda()
loss_network.eval()
adversarial_loss = torch.nn.MSELoss()
class dehamer(object):
    """Implementation of dehamer from Guo et al. (2022)."""

    def __init__(self, params, trainable):
        """Initializes model."""
        self.p = params
        self.trainable = trainable
        self._compile()


    def _compile(self):

        self.model = UNet_emb()

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        # if self.use_cuda:
        self.model = self.model.cuda()
        if self.trainable:
                self.loss = self.loss.cuda()

    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = self.p.dataset_name #f'{datetime.now():{self.p.dataset_name}-%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.dataset_name

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/dehamer-{}.pt'.format(self.ckpt_dir, self.p.dataset_name)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/dehamer-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1 , valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/dehamer-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        # if self.use_cuda:
        self.model.load_state_dict(torch.load(ckpt_fname))
        # else:
        #     self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""
        # import pdb;pdb.set_trace()
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)
        # self.scheduler2.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = self.p.loss.upper()#f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('./results'):
        os.makedirs('./results/source')
        os.makedirs('./results/target')
        os.makedirs('./results/dehaze')
    def eval(self, valid_loader):
        with torch.no_grad():
            self.model.train(False)

            valid_start = datetime.now()
            loss_meter = AvgMeter()
            psnr_meter = AvgMeter()

            for batch_idx, (source, target,haze_name) in enumerate(valid_loader):
                # if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

                h, w = source.size(2), source.size(3)
                haze_name= haze_name[0]

                pad_h = h % 16
                pad_w = w % 16
                source = source[:, :, 0:h - pad_h, 0:w - pad_w]
                target = target[:, :, 0:h - pad_h, 0:w - pad_w]

                # dehaze
                cor, trr = self.model.Encoder(source)
                source_dehazed, _ = self.model.Decoder(cor, trr , s=2)
                vutils.save_image(source.data, './results/source/'+haze_name, padding=0,
                                  normalize=True)  # False
                vutils.save_image(target.data, './results/target/'+haze_name , padding=0, normalize=True)
                vutils.save_image(source_dehazed.data, './results/dehaze/' +haze_name, padding=0,
                                  normalize=True)
                # Update loss
                loss = self.loss(source_dehazed, target)
                loss_meter.update(loss.item())

                # Compute PSRN
                for i in range(source_dehazed.shape[0]):
                    # import pdb;pdb.set_trace()
                    source_dehazed = source_dehazed.cpu()
                    target = target.cpu()
                    psnr_meter.update(psnr(source_dehazed[i], target[i]).item())
            valid_loss = loss_meter.avg
            valid_time = time_elapsed_since(valid_start)[0] 
            psnr_avg = psnr_meter.avg

            return valid_loss, valid_time, psnr_avg 
 

    def train(self, train_loader1,train_loader2, valid_loader):
        """Trains denoiser on training set."""  
 
        self.model.train(True)
        
        if self.p.ckpt_load_path is not None:
            self.model.load_state_dict(torch.load(self.p.ckpt_load_path), strict=False)
            print('The pretrain model is loaded.')
        self._print_params()
        num_batches1 = len(train_loader1)
        # assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'
        num_batches2= len(train_loader2)
        # Dictionaries of tracked stats
        stats = {'dataset_name': self.p.dataset_name, 
                 'train_loss': [],
                 'valid_loss': [], 
                 'valid_psnr': []}  
  
        # Main training loop 
        train_start = datetime.now()
        ite=0
        for epoch in range(self.p.nb_epochs): 
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()
            if not epoch % 2:
                train_loader = train_loader1
                num_batches=num_batches1
            else:
                train_loader = train_loader2
                num_batches = num_batches2

            # Minibatch SGD
            for batch_idx, (source, target,realhazy) in enumerate(train_loader):
              batch_start = datetime.now()
              progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                # if self.use_cuda:
              source = source.cuda()
              target = target.cuda()
              real = realhazy.cuda()
              if target.size(1) == 3 and real.size(1) == 3:

                tmp_weight = self.model.state_dict()  # task_net <- meta_net

                cos, trs = self.model.Encoder(source)
                cot, trt = self.model.Encoder(target)

                fake_target,query_trt = self.model.Decoder(cot,trs )#, s=2
                fake_source,query_trs = self.model.Decoder(cos,trt)#, s=2

                cor, trr = self.model.Encoder(real.detach())
                recover_real, query_real = self.model.Decoder(cor, trr)

                source_dehazed1, query_des = self.model.Decoder(cos, trs, s=2)

                loss_fake =  self.loss(fake_target, target) + loss_network(fake_target, target) * 0.04 \
                             + self.loss(fake_source, source) + loss_network(fake_source, source) * 0.04 \
                             + self.loss(recover_real, real) + loss_network(recover_real, real) * 0.04 \
                             +self.loss(source_dehazed1, target) + loss_network(source_dehazed1, target) * 0.04\
                             +self.loss(query_des, query_trt)
                loss_fake =100* loss_fake

                self.optim.zero_grad()
                loss_fake.backward()
                self.optim.step()

                cott, trtt = self.model.Encoder(target)

                cos, trs = self.model.Encoder(source.detach())
                source_dehazed,query_dess = self.model.Decoder(cos ,trs ,s=2 )

                cor, trr = self.model.Encoder(real.detach())
                real_dehazed,query_derr = self.model.Decoder(cor ,trr ,s=2 )

                loss_de = self.loss(source_dehazed.detach(), target.detach())+ loss_network(source_dehazed, target.detach()) * 0.04\

                loss_cf = self.loss(query_dess ,cott) / (self.loss(query_dess , cos)+ self.loss(query_dess, cor) )\
                            +self.loss(query_derr , cott) / (self.loss(query_derr ,  cos)+ self.loss(query_derr, cor) )\
                            # +self.loss(real_dehazed ,real.detach()) /( (real_dehazed , target.detach())+ self.loss(query_dess,source.detach()) )\

                loss_dcp = DCLoss((real_dehazed + 1) / 2, 8)#+ DCLoss((source_dehazed + 1) / 2, 16)

                loss = 100*loss_de + 0.01 *loss_cf + 0.01 *loss_dcp
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.model.load_state_dict(tmp_weight)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg,loss_fake, loss_de ,loss_dcp,loss_cf, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()
                ite += 1
                if ite % 100 == 0:
                    # print(output)
                    vutils.save_image(source.data, './source.png', normalize=True)
                    vutils.save_image(source_dehazed.data, './source_dehazed.png', normalize=True)
                    vutils.save_image(target.data, './target.png', normalize=True)
                    vutils.save_image(source_dehazed1.data, './source_dehazed1.png', normalize=True)
                    vutils.save_image(fake_target.data, './fake_target.png', normalize=True)
                    vutils.save_image(fake_source.data, './fake_source.png', normalize=True)
                    vutils.save_image(real.data, './real.png', normalize=True)
                    vutils.save_image(real_dehazed.data, './real_dehazed.png', normalize=True)

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))




