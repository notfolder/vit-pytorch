import torch
import torchvision
import numpy as np
import pandas as pd
import zipfile
from PIL import Image


class DatasetTiff(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform

        self.df = pd.read_csv(csv_file)

        self.datanum = len(self.df)

        self.out_data = None
        self.out_label = None

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx,:]
            out_label = row['label']
            tiff_filename = row['tiff_filename']
            index_r = row['index_r']
            index_g = row['index_g']
            index_b = row['index_b']
            img = Image.open(tiff_filename)
            img.seek(index_r)
            im_r = np.array(img.copy().resize((128,128)).convert('L'))
            im_r = (im_r - im_r.mean()) / im_r.std()
            im_r = Image.fromarray((im_r*64 + 128).clip(0,255)).convert('L')
            img.seek(index_g)
            im_g = np.array(img.copy().resize((128,128)).convert('L'))
            im_g = (im_g - im_g.mean()) / im_g.std()
            im_g = Image.fromarray((im_g*64 + 128).clip(0,255)).convert('L')
            img.seek(index_b)
            im_b = np.array(img.copy().resize((128,128)).convert('L'))
            im_b = (im_b - im_b.mean()) / im_b.std()
            im_b = Image.fromarray((im_b*64 + 128).clip(0,255)).convert('L')
            out_data = np.array(Image.merge('RGB',(im_r,im_g,im_b)))

            if self.transform:
                out_data = self.transform(out_data)

            return out_data, out_label
        except:
            if self.out_data is None or self.out_label is None:
                pass
            print("####### Exception raised!! START #######")
            import traceback
            traceback.print_exc()
            print("####### Exception raised!! END   #######")
            return self.out_data, self.out_label
    
    def get_labels(self):
        return self.df['label']


if __name__ == '__main__':
    import os

    # trans = torchvision.transforms.ToTensor()
    trans = torchvision.transforms.ToPILImage()
    dataset = DatasetTiff('./data-tiff/dataset-tiff.csv', trans)

    os.makedirs('./tmp', exist_ok=True)

    for i in range(len(dataset)):
        (img, label) = dataset[i]
        img.save(os.path.join('./tmp', f'{i}.png'))
