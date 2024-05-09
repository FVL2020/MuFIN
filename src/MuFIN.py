import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset, prepare_data
from .models import InpaintingModel, RNN_ENCODER
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
import time
np.set_printoptions(precision=4, suppress=True)

class MuFIN():
    def __init__(self, config):
        self.config = config

        self.debug = False
        self.model_name = 'inpaint'
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        
        self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, mode='test', training=False) 
        self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, mode='train', training=True)
        self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST, augment=False, mode='val', training=True)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')       



    def load(self):
       self.inpaint_model.load()

    def save(self):
        self.inpaint_model.save()

       
    def save_best(self):
        self.inpaint_model.save_best()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )
        
        text_encoder = RNN_ENCODER(self.train_dataset.n_words, nhidden=self.config.image_size)
        text_encoder_path = self.config.text_encoder_path
        state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', text_encoder_path)
        text_encoder.eval()

        text_encoder = text_encoder.to(self.config.DEVICE)
     
        epoch = 0
        keep_training = True
        max_epoch = int(float((self.config.MAX_EPOCH)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        max_average_psnr = 0        
        while(keep_training):
            epoch += 1
            if epoch > max_epoch:
                keep_training = False
                break
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.inpaint_model.train()

                images, masks, captions, cap_lens = prepare_data(items, device=self.config.DEVICE)
                
                hidden = text_encoder.init_hidden(self.config.BATCH_SIZE)
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                text_mask = (captions == 0)
                num_words = words_embs.size(2)
                if text_mask.size(1) > num_words:
                    text_mask = text_mask[:, :num_words]

                mask_percent = self.mask_percent(masks)
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, words_embs, sent_emb, text_mask)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                logs.append(('mask_percent', mask_percent.item()))

                # backward
                iteration = self.inpaint_model.iteration

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

            self.save()
            print('\nstart eval...\n')
            average_psnr = self.eval()
            print(average_psnr)
            if max_average_psnr < average_psnr:
                max_average_psnr = average_psnr
                self.save_best() 
        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=4,
            num_workers=0,
        )
        text_encoder = RNN_ENCODER(self.test_dataset.n_words, nhidden=self.config.image_size)
        text_encoder_path = self.config.text_encoder_path
        state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', text_encoder_path)
        text_encoder.eval()

        text_encoder = text_encoder.to(self.config.DEVICE)
        average_psnr = 0
        self.inpaint_model.eval()
        create_dir(self.results_path)
        iteration = 0
        for items in val_loader:
            name = self.test_dataset.load_name(iteration)
            images, masks, captions, cap_lens = prepare_data(items, device=self.config.DEVICE)
            hidden = text_encoder.init_hidden(4)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            text_mask = (captions == 0)
            num_words = words_embs.size(2)
            if text_mask.size(1) > num_words:
                text_mask = text_mask[:, :num_words]
            iteration += 1

            mask_percent = self.mask_percent(masks)

            outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, words_embs, sent_emb, text_mask)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            average_psnr += psnr.sum().item()
            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))
            logs.append(('mask_percent', mask_percent.item()))
            
            logs = [("it", iteration), ] + logs
            eval_file = os.path.join(self.results_path, 'log_' + 'eval' + '.dat')

            with open(eval_file, 'a') as f:
                f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
       
        average_psnr = average_psnr / iteration
        return average_psnr
        

    def test(self):
        self.inpaint_model.eval()
        
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        text_encoder = RNN_ENCODER(self.test_dataset.n_words, nhidden=self.config.image_size)
        text_encoder_path = self.config.text_encoder_path
        state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', text_encoder_path)
        text_encoder.eval()
        text_encoder = text_encoder.to(self.config.DEVICE)

        index = 0       
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, masks, captions, cap_lens = prepare_data(items, self.config.DEVICE)
            hidden = text_encoder.init_hidden(1)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            text_mask = (captions == 0)
            num_words = words_embs.size(2)
            if text_mask.size(1) > num_words:
                text_mask = text_mask[:, :num_words]
            index += 1
            mask_percent = self.mask_percent(masks)

            outputs = self.inpaint_model(images, masks, words_embs, sent_emb, text_mask)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
            logs = [('index', index),]
            logs.append(('name', name))
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))
            logs.append(('mask_percent', mask_percent.item()))

            path = os.path.join(self.results_path, name)
            output = self.postprocess(outputs_merged)[0]
            print(index, name)
            imsave(output, path)
            
            if self.debug:
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

            test_file = os.path.join(self.results_path, 'log_' + 'test' + '.dat')

            with open(test_file, 'a') as f:
                f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

        print('\nEnd test....')

    
    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return
        self.inpaint_model.eval()
        text_encoder = RNN_ENCODER(self.val_dataset.n_words, nhidden=self.config.image_size)
        text_encoder_path = self.config.text_encoder_path
        state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', text_encoder_path)
        text_encoder.eval()
        text_encoder = text_encoder.to(self.config.DEVICE)

        items = next(self.sample_iterator)
        images, masks, captions, cap_lens = prepare_data(items, device=self.config.DEVICE)
        mask_percent = self.mask_percent(masks)
        
        hidden = text_encoder.init_hidden(self.config.SAMPLE_SIZE)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        text_mask = (captions == 0)
        num_words = words_embs.size(2)
        if text_mask.size(1) > num_words:
            text_mask = text_mask[:, :num_words]

       
        iteration = self.inpaint_model.iteration
        inputs = (images * (1 - masks)) + masks
        outputs = self.inpaint_model(images, masks, words_embs, sent_emb, text_mask)
        outputs_merged = (outputs * masks) + (images * (1 - masks))


      
        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        path = os.path.join(self.samples_path, self.model_name)
        create_dir(path)
        
        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )

        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        print('\nsaving sample ' + name)
        images.save(name)
        

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def mask_percent(self, mask):
        holes = torch.sum((mask > 0).float())
        pixel_num = torch.sum((mask >= 0).float())
        percent = holes/pixel_num
        return percent