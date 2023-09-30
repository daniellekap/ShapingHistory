import pandas as pd
from glob import glob
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2


def get_IDS(IMG_DIR='output/images', era=False, CATALOGUE_FN='output/cdli_catalogue_data.csv'):
    img_fns = glob(os.path.join(IMG_DIR, '*.png'))
    IDS = [os.path.basename(fn).rstrip('.png') for fn in img_fns]
    if era:
        IDS = list(set(IDS) & set(pd.read_csv(
            CATALOGUE_FN, usecols=['id_text', 'era'], dtype={'id_text': object}
        ).dropna(subset=['era']).set_index('id_text').to_dict()['era'].keys()))
    return IDS

def pad_zeros(x):
    x_new = str(x)
    return (6-len(x_new))*'0'+x_new

class TabletEraDataset(Dataset):
    
    ERA_INDICES = {
        'early_bronze': 0,
        'mid_late_bronze': 1,
        'iron': 2
    }
    
    def __init__(self, CATALOGUE_FN='output/cdli_catalogue_data.csv', IMG_DIR='output/images', IDS=None):
        self.id2era = pd.read_csv(
            CATALOGUE_FN, usecols=['id_text', 'era'], dtype={'id_text': object}
        ).dropna(subset=['era']).set_index('id_text').to_dict()['era']
        
        self.img_fns = glob(os.path.join(IMG_DIR, '*.png'))
        self.IDS = [os.path.basename(fn).rstrip('.png') for fn in self.img_fns]
        
        if IDS is not None:
            print(f'Filtering {len(self.IDS)} IDS down to provided {len(IDS)}...')
            IDS_set = set(IDS)
            indices = [i for i, ID in enumerate(self.IDS) if ID in IDS_set]
            self.img_fns = [self.img_fns[i] for i in indices]
            self.IDS = [self.IDS[i] for i in indices]
        
    def __len__(self):
        return len(self.IDS)
        
    def __getitem__(self, idx):
        fn = self.img_fns[idx]
        ID = self.IDS[idx]
        era = self.id2era[ID]
        img = np.asarray(Image.open(fn))
        return img.astype(np.float32) / 255, self.ERA_INDICES[era]


class TabletPeriodDataset(Dataset):
    
    # based on (normed) periods with at least 100 photos:
    PERIOD_INDICES = {
        
        'other': 0,
        'Ur III': 1,
        'Neo-Assyrian': 2,
        'Old Babylonian': 3,
        'Middle Babylonian': 4,
        'Neo-Babylonian': 5,
        'Old Akkadian': 6,
        'Achaemenid': 7,
        'Early Old Babylonian': 8,
        'ED IIIb': 9,
        'Middle Assyrian': 10,
        'Old Assyrian': 11,
        'Uruk III': 12,
        'Proto-Elamite': 13,
        'Lagash II': 14,
        'Ebla': 15,
        'ED IIIa': 16,
        'Hellenistic': 17,
        'ED I-II': 18,
        'Middle Elamite': 19,
        'Middle Hittite': 20,
        'Uruk IV': 21
    }
    
    GENRE_INDICES = {
        
        'Administrative': 1,
        'Letter': 2,
        'Legal': 3,
        'Royal/Monumental': 4,
        'Literary': 5,
        'Lexical': 6,
        'Omen': 7,
        'uncertain': 8,
        'Administrative ?': 1,
        'School': 9,
        'Mathematical': 10,
        'Prayer/Incantation': 11,
        'Lexical ?': 6,
        'Scientific': 12,
        'Ritual': 13,
        'Letter ?': 2,
        'Literary ?': 5,
        'fake (modern)': 14,
        'Lexical; Literary': 6,
        'Legal ?': 3,
        'Literary; Mathematical': 5,
        'Astronomical': 15,
        'Lexical; Mathematical': 6,
        'School ?': 9,
        'Mathematical ?': 10,
        'Royal/Monumental ?': 4,
        'Private/Votive': 16,
        'fake (modern) ?': 14,
        'Other (see subgenre)': 8,
        'Historical': 2,
        'Literary; Lexical': 5,
        'Lexical; Literary; Mathematical': 6,
        'Literary; Administrative': 5,
        'Literary; Letter': 5,
        'Scientific ?': 12,
        'Royal/Monumental; Literary': 4,
        'Private/Votive ?': 16,
        'School; Literary': 9,
        'Prayer/Incantation ?': 11,
        'Ritual ?': 13,
        'Lexical; School': 6
    }
    
    def __init__(self, CATALOGUE_FN='output/cdli_catalogue_data.csv', IMG_DIR='output/images', IDS=None, mask=False):
        
        df = pd.read_csv(
            CATALOGUE_FN, usecols=['id_text', 'era', 'period_normed', 'genre'], dtype={'id_text': object}
        ).dropna(subset=['era'])
        
        df["id_text"] = df.id_text.apply(lambda x: pad_zeros(x))
        df = df[df['period_normed'].isin(TabletPeriodDataset.PERIOD_INDICES.keys())]
        
        self.id2period = df.set_index('id_text').to_dict()['period_normed']
        self.id2genre = df.set_index('id_text').to_dict()['genre']
        self.genre = df.set_index('id_text').to_dict()['genre']
        self.img_fns = glob(os.path.join(IMG_DIR, '*.png'))
        self.IDS = [os.path.basename(fn).rstrip('.png') for fn in self.img_fns]
        
        if IDS is not None:
            print(f'Filtering {len(self.IDS)} IDS down to provided {len(IDS)}...')
            IDS_set = set(IDS)
            indices = [i for i, ID in enumerate(self.IDS) if ID in IDS_set]
            self.img_fns = [self.img_fns[i] for i in indices]
            self.IDS = [self.IDS[i] for i in indices]
        
        self.mask = mask
        
    def __len__(self):
        return len(self.IDS)
        
    def __getitem__(self, idx):
        fn = self.img_fns[idx]
        ID = self.IDS[idx]
        try:
            period = self.id2period[ID]
        except KeyError as ke:
            #print('Key Not Found in Period Dictionary:', ke)
            period = 0

        try:
            genre = self.id2genre[ID]
        except KeyError as ke:
            #print('Key Not Found in Period Dictionary:', ke)
            genre = 8 # other/uncertain
        
        img = np.asarray(Image.open(fn))
        alpha = 3 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = img.astype(np.float32) / 255
        img = cv2.GaussianBlur(img, (11,11), 0)
        if self.mask:
            img = (img > 0.125).astype(np.float32) ### 0.25 was great for most besides the really dark ones
        
        
        return img, self.PERIOD_INDICES.get(period, 0), self.GENRE_INDICES.get(genre, 8) # 0: other


    def getitem_extended(self, idx):
        fn = self.img_fns[idx]
        ID = self.IDS[idx]
        try:
            period = self.id2period[ID]
        except KeyError as ke:
            #print('Key Not Found in Period Dictionary:', ke)
            period = 0
        
        try:
            genre = self.id2genre[ID]
        except KeyError as ke:
            #print('Key Not Found in Period Dictionary:', ke)
            genre = 8 # other/uncertain
        
        img = np.asarray(Image.open(fn))
        alpha = 3 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = img.astype(np.float32) / 255
        img = cv2.GaussianBlur(img, (11,11), 0)
        if self.mask:
            img = (img > 0.125).astype(np.float32) ### 0.25 was great for most besides the really dark ones
        
        
        return img, self.PERIOD_INDICES.get(period, 0), self.GENRE_INDICES.get(genre, 8) # 0: other