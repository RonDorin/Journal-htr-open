from Crnn_arch import CRNN
import torch
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pylab as plt
from PIL import Image


class Model():
    def __init__(self):
        alphabet = " %(),-./0123456789:;?[]«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё-"
        self.device = torch.device("cpu")
        self.model = CRNN(256, len(alphabet)).to(self.device)
        self.model.load_state_dict(torch.load('./Models/crnn.pth', map_location=self.device))
        self.coder = LabelCoder(alphabet)
    
    def resize_image(self, img, size):
        img = img.convert('L')
        img = np.asanyarray(img)
        w, h  = img.shape
        new_w = size[0]
        new_h = int(h * (new_w / w))
        img = cv2.resize(img, (new_h, new_w))
        w, h = img.shape

        img = img.astype('float32')

        new_h = size[1]
        if h < new_h:
            add_zeros = np.full((w, new_h - h), img.mean())
            img = np.concatenate((img, add_zeros), axis=1)

        if h > new_h:
            img = cv2.resize(img, (new_h, new_w))

        return img


    def predict(self, input_img):
        self.model.eval()
        with torch.no_grad():
            plt.imsave('tmp.png', input_img)
            input_img = Image.open('tmp.png')
            img_resized = self.resize_image(input_img, (64, 340))
            transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Grayscale(1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
            norm_tensor = transform(img_resized.astype(np.uint8))
            norm_tensor = torch.unsqueeze(norm_tensor, dim=0)
            logits = self.model(norm_tensor.to(self.device))
            logits = logits.contiguous().cpu()
            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            probs, pos = logits.max(2)
            pos = pos.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.coder.decode(pos.data, pred_sizes.data, raw=False)

            return sim_preds

class LabelCoder(object):
    def __init__(self, alphabet, ignore_case=False):
        self.alphabet = alphabet
        self.char2idx = {}
        for i, char in enumerate(alphabet):
            self.char2idx[char] = i + 1
        self.char2idx[''] = 0

    def encode(self, text: str):
        length = []
        result = []
        for item in text:
            length.append(len(item))
            for char in item:
                if char in self.char2idx:
                    index = self.char2idx[char]
                else:
                    index = 0
                result.append(index)

        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts