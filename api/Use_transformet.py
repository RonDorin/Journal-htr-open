from Transformer_arch import TransformerModel
import torch
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pylab as plt
from PIL import Image


class Model():
    def __init__(self):
        alphabet = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»',
         'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 
         'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 
         'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        hidden = 512
        self.char2idx = {char: idx for idx, char in enumerate(alphabet)}
        self.idx2char = {idx: char for idx, char in enumerate(alphabet)}
        self.device = torch.device("cpu")
        self.model = TransformerModel('resnet50', len(alphabet), hidden=hidden, enc_layers=2, dec_layers=2, nhead=4, dropout=0.0).to(self.device)
        self.model.load_state_dict(torch.load('./Models/htr_transformer_11_04_2.pth', map_location=self.device))
    
    def labels_to_text(self, s, idx2char):
        S = "".join([idx2char[i] for i in s])
        if S.find('EOS') == -1:  
            return S
        else:
            return S[:S.find('EOS')]

    def resize_image(self, img, size):
        img = np.asanyarray(img)
        w, h, _ = img.shape
        new_w = size[0]
        new_h = int(h * (new_w / w))
        img = cv2.resize(img, (new_h, new_w))
        w, h, _ = img.shape

        img = img.astype('float32')
        new_h = size[1]
        if h < new_h:
            add_zeros = np.full((w, new_h - h, 3), 255)
            img = np.concatenate((img, add_zeros), axis=1)

        if h > new_h:
            img = cv2.resize(img, (new_h, new_w))

        return img.astype('uint8')

    def predict(self, input_img):
        self.model.eval()
        with torch.no_grad():
            plt.imsave('tmp.png', input_img)
            input_img = Image.open('tmp.png').convert('RGB')
            img_resized = self.resize_image(input_img, (64, 256))
            transform = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize((64, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
            norm_img = transform(img_resized.astype('uint8'))
            src = torch.FloatTensor(norm_img).unsqueeze(0).to(self.device)
            out_indexes = [self.char2idx['SOS'], ]

            for i in range(100):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(self.device)
                output = self.model(src,trg_tensor)
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == self.char2idx['EOS']:
                    break

            pred = self.labels_to_text(out_indexes[1:], self.idx2char)
        
        return pred 