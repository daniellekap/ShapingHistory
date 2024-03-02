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

    PROVENIENCE_INDICES = {
        'Nineveh': 1,
         'Nippur': 2,
         'unknown': 3,
         'Umma': 4,
         'Puzris-Dagan': 5,
         'Girsu': 6,
         'Ur': 7,
         'Uruk': 8,
         'Kanesh': 9,
         'Assur': 10,
         'Adab': 11,
         'Garsana': 12,
         'Gasur/Nuzi': 13,
         'Susa': 14,
         'Sippar-Yahrurum': 15,
         'Larsa': 16,
         'Nerebtum': 17,
         'mod. Babylonia': 18,
         'Parsa': 19,
         'Kish': 20,
         'Kalhu': 21,
         'Tuttul': 22,
         'Suruppak': 23,
         'Babili': 24,
         'Ebla': 25,
         'mod. Beydar': 26,
         'Akhetaten': 27,
         'Esnunna': 28,
         'Borsippa': 29,
         'Kar-Tukulti-Ninurta': 30,
         'mod. Jemdet Nasr': 31,
         'mod. northern Babylonia': 32,
         'Alalakh': 33,
         'Hattusa': 34,
         'Isin': 35,
         'Elbonia': 36,
         'Sibaniba': 37,
         'Tutub': 38,
         'Pi-Kasi': 39,
         'Irisagrig': 40,
         'Ansan': 41,
         'Dilbat': 42,
         'Zabalam': 43,
         'mod. Mugdan/ Umm al-Jir': 44,
         'Marad': 45,
         'Eridu': 46,
         'Seleucia': 47,
         'mod. Abu Halawa': 48,
         'Dur-Untas': 49,
         'Nagar': 50,
         'Lagaba': 51,
         'Asnakkum': 52,
         'Dur-Kurigalzu': 53,
         'mod. Tell Sabaa': 54,
         'mod. Abu Jawan': 55,
         'mod. Tell Fakhariyah': 56,
         'Dur-Abi-esuh': 57,
         'Ugarit': 58,
         'mod. Diqdiqqah': 59,
         'Tarbisu': 60,
         'Lagash': 61,
         'Kisurra': 62,
         'Elammu': 63,
         'Du-Enlila': 64,
         'Kutha': 65,
         'mod. Umm el-Hafriyat': 66,
         'Dur-Sarrukin': 67,
         'Bad-Tibira': 68,
         'Bit-zerija': 69,
         'Kilizu': 70,
         'mod. Pasargadae': 71,
         'Abdju': 72,
         'Surmes': 73,
         'mod. Qatibat': 74,
         'Tigunanum': 75,
         'mod. Tell al-Lahm': 76,
         'mod. Mesopotamia': 77,
         'Subat-Enlil': 78,
         'mod. Konar Sandal': 79,
         'Gissi': 80,
         'Agamatanu': 81,
         'Aqa': 82,
         'Kapri-sa-naqidati': 83,
         'Esura': 84,
         'Nahalla': 85,
         'Bit-Sahtu': 86,
         'mod. Sepphoris': 87,
         'Dusabar': 88,
         'mod. Tell Sifr': 89,
         'Nasir': 90,
         'Kumu': 91,
         'Kazallu': 92,
         'Kapru': 93,
         'Hurruba': 94,
         'mod. Deh-e-no, Iran': 95,
         "mod. Za'aleh": 96,
         'mod. Tepe Farukhabad': 97,
         'Hursagkalama': 98,
         'Carchemish': 99,
         'mod. Ben Shemen, Israel': 100,
         'Kutalla': 101,
         'Der': 102,
         'Imgur-Enlil': 103,
         'mod. Hillah': 104,
         'mod. Uhudu': 105,
         'mod. Mahmudiyah': 106,
         'Terqa': 107,
         'Arrapha': 108,
         'mod. Tell en-Nasbeh': 109,
         'mod. Kalah Shergat': 110,
         'Kar-Nabu': 111,
         'Harran': 112,
         'mod. Til-Buri': 113,
         'Shuruppak': 114,
         'mod. Abu Salabikh': 115,
         "Ma'allanate": 116,
         'Kar-Mullissu': 117,
         'mod. Naqs-i-Rustam': 118
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
            CATALOGUE_FN, usecols=['id_text', 'era', 'period_normed', 'provenience_normed', 'genre'], dtype={'id_text': object}
        ).dropna(subset=['era'])
        
        df["id_text"] = df.id_text.apply(lambda x: pad_zeros(x))
        df = df[df['period_normed'].isin(TabletPeriodDataset.PERIOD_INDICES.keys())]
        
        self.id2period = df.set_index('id_text').to_dict()['period_normed']
        self.id2provenience = df.set_index('id_text').to_dict()['provenience_normed']
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

        try:
            provenience = self.id2provenience[ID]
        except KeyError as ke:
            #print('Key Not Found in Period Dictionary:', ke)
            provenience = 3 # unknown
        
        img = np.asarray(Image.open(fn))
        alpha = 3 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = img.astype(np.float32) / 255
        img = cv2.GaussianBlur(img, (11,11), 0)
        if self.mask:
            img = (img > 0.125).astype(np.float32) ### 0.25 was great for most besides the really dark ones
        
        
        return img, self.PERIOD_INDICES.get(period, 0), self.GENRE_INDICES.get(genre, 8), self.PROVENIENCE_INDICES.get(provenience, 3) # 0: other