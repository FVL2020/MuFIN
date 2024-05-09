
import os
import glob
import scipy
import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
# from imageio import imread
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask, create_center_mask


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import sys
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from torch.autograd import Variable

def prepare_data(data, device=None):
    imgs, masks, captions, captions_lens = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)
    imgs = imgs[sorted_cap_indices]
    if device is not None:
        real_imgs = (Variable(imgs).to(device))
    else:
        real_imgs = (Variable(imgs).cuda())
    masks = masks[sorted_cap_indices]    
    if device is not None:
        sotred_masks = (Variable(masks).to(device))
    else:
        sotred_masks = (Variable(masks).cuda())

    captions = captions[sorted_cap_indices]

    captions = captions.squeeze()

    if imgs.shape[0] == 1:
        captions = captions.unsqueeze(dim=0)
  
    if device is not None:
        captions = Variable(captions).to(device)  
        sorted_cap_lens = Variable(sorted_cap_lens).to(device)  
    else:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    return [real_imgs, sotred_masks, captions, sorted_cap_lens]
            
class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, mode, augment=True, training=True, CAPTIONS_PER_IMAGE=1, WORDS_NUM=18): #ting
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)
        self.embeddings_num = CAPTIONS_PER_IMAGE
        self.WORDS_NUM = WORDS_NUM
        self.caption_path = config.caption_path
        
        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(self.caption_path, mode)


        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # load mask
        mask = self.load_mask(img, index)

        
        
        if self.augment and random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
        if self.training == True:
            sent_ix = random.randint(0, self.embeddings_num)
        else:
            sent_ix = 0
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return self.to_tensor(img), self.to_tensor(mask), caps, cap_len


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = random.randint(1, 4)

        # center block
        if mask_type == 0:
            return create_center_mask(imgw, imgh)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3 and self.training == True:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 128).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6 or self.training == False:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 128).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])
        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            #print(cap_path)
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf-8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions, val_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions + val_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        val_captions_new = []
        for t in val_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            val_captions_new.append(rev)

        return [train_captions_new, test_captions_new, val_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def caption2idx(self, captions, wordtoix):
        captions_new = []
        for t in captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            captions_new.append(rev)
        return captions_new

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        val_names = self.load_filenames(data_dir, 'val')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)
            val_captions = self.load_captions(data_dir, val_names)

            train_captions, test_captions, val_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions, val_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions, val_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions, val_captions = x[0], x[1], x[2]
                ixtoword, wordtoix = x[3], x[4]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        elif split == 'test':
            captions = test_captions
            filenames = test_names
            # filenames = test_names
            # captions = self.load_captions(data_dir, test_names)
            # captions = self.caption2idx(captions, wordtoix)
        else:
            captions = val_captions
            filenames = val_names
        return filenames, captions, ixtoword, wordtoix, n_words


    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.txt' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                filenames = f.readlines()
            filenames = [i.split('\n')[0] for i in filenames]
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            random.shuffle(ix)
            ix = ix[:self.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.WORDS_NUM
        return x, x_len

