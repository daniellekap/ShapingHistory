import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import re
import requests
from bs4 import BeautifulSoup as bs

def get_url(offset=0):
    return f'''
https://cdli.ucla.edu/search/search_results.php
?SearchMode=Text&PrimaryPublication=&MuseumNumber=&Provenience=
&Period=&TextSearch=&ObjectID=&requestFrom=Submit
&offset={offset}
'''.replace('\n', '')

def scrape(offset=0):
    URL = get_url(offset=offset)
    print('Scraping URL:', URL)
    res = requests.get(URL)
    print('Response OK?', res.ok)
    soup = bs(res.content, features='lxml')
    tables = soup.find_all('table', class_='full_object_table')
    data = [CDLIDatum(table) for table in tables]
    print(f'{len(data)} items found ({sum(D.has_id for D in data)} have IDs, {sum(D.has_photo for D in data)} have photos)')
    return data
    

def find_obverse(CC):
    H, W = CC.shape
    vertical_slice = CC[:, W // 2]
    nonzeros = [x for x in vertical_slice if x != 0]
    first = nonzeros[0]
    candidates = [
        x for x in nonzeros if x != first
    ]
    if len(candidates) > 0:
        return candidates[0]
    
def crop(img):
    xzeros = (img == 0).all(axis=0)
    yzeros = (img == 0).all(axis=1)
    x0, x1 = np.where(~xzeros)[0][[0, -1]]
    y0, y1 = np.where(~yzeros)[0][[0, -1]]
    return img[y0:y1, x0:x1].copy(), (y0, y1), (x0, x1)

def removeprefix(txt, pref):
    if len(txt) >= len(pref) and txt[:len(pref)] == pref:
        return txt[len(pref):]
    return txt

class CDLIDatum:
    
    PREFIX = 'https://cdli.ucla.edu/dl/'
    
    def __init__(self, table):
        
        self.table = table
        
        columns = table.find_all('td')
        
        self.has_text = False
        if len(columns) > 0:
            self.text = columns[-1].text
            self.has_text = True
        
        ID_tds = [td for td in columns if td.text == 'CDLI no.']
        self.has_id = False
        if len(ID_tds) > 0:
            self.ID = removeprefix(ID_tds[0].parent.text, 'CDLI no.')
            self.has_id = True
        
        period_tds = [td for td in columns if td.text == 'Period']
        self.has_period = False
        if len(period_tds) > 0:
            self.period = removeprefix(period_tds[0].parent.text, 'Period')
            self.has_period = True
            
        genre_tds = [td for td in columns if td.text == 'Genre']
        self.has_genre = False
        if len(genre_tds) > 0:
            self.genre = removeprefix(genre_tds[0].parent.text, 'Genre')
            self.has_genre = True
            
        subgenre_tds = [td for td in columns if td.text == 'Sub-genre']
        self.has_subgenre = False
        if len(subgenre_tds) > 0:
            self.subgenre = removeprefix(subgenre_tds[0].parent.text, 'Sub-genre')
            self.has_subgenre = True
            
        material_tds = [td for td in columns if td.text == 'Material']
        self.has_material = False
        if len(material_tds) > 0:
            self.material = removeprefix(material_tds[0].parent.text, 'Material')
            self.has_material = True
            
        objecttype_tds = [td for td in columns if td.text == 'Object type']
        self.has_objecttype = False
        if len(objecttype_tds) > 0:
            self.objecttype = removeprefix(objecttype_tds[0].parent.text, 'Object type')
            self.has_objecttype = True
        
        urls = {
            re.sub('^/dl/', '', a['href']) for a in table.find_all('a', href=True)
            if a['href'].startswith('/dl/')
        }
        
        photo_urls = [y for y in urls if y.startswith('photo')]
        lineart_urls = [y for y in urls if y.startswith('lineart')]

        self.has_photo = False
        if len(photo_urls) > 0:
            self.photo_url = self.PREFIX + [y for y in urls if y.startswith('photo')][0]
            self.has_photo = True
        
        self.has_lineart = False
        if len(lineart_urls) > 0:
            self.lineart_url = self.PREFIX + [y for y in urls if y.startswith('lineart')][0]
            self.has_lineart = True
            
        self.loaded = False
    
    def load(self, gray=True):
        if not self.loaded:
            if self.has_photo:
                self.photo_res = requests.get(self.photo_url)
                self.photo = np.asarray(Image.open(BytesIO(self.photo_res.content)))
                self.photo = cv2.cvtColor(self.photo, cv2.COLOR_RGB2GRAY)
                
                img_blurred = cv2.GaussianBlur(self.photo, (101, 101), 0)
                _, self.blobs = cv2.threshold(img_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                n_CC, self.CC = cv2.connectedComponents(self.blobs)
                self.mask = (self.CC != 0).astype(int)

                self.obverse_idx = find_obverse(self.CC)

                self.n_faces = n_CC - 1

                if self.obverse_idx is not None:
                    self.obverse = self.photo.copy()
                    self.obverse[self.CC != self.obverse_idx] = 0
                    self.obverse, (y0, y1), (x0, x1) = crop(self.obverse)
                    self.obverse_mask = self.mask[y0:y1, x0:x1].copy()
            
            if self.has_lineart:
                self.lineart_res = requests.get(self.lineart_url)
                self.lineart = np.asarray(Image.open(BytesIO(self.lineart_res.content)))
                self.lineart = cv2.cvtColor(self.lineart, cv2.COLOR_RGB2GRAY)

        
        self.loaded = True
    
    def display(self, figsize=None):
        
        if not self.has_photo:
            print(f'Warning: No photo for {self.ID}')
        else:
        
            if not self.loaded:
                self.load()

            has_obverse = self.obverse_idx is not None

            fig, axs = plt.subplots(1, 2 + self.has_lineart + 2 * has_obverse, figsize=figsize)
            axs[0].imshow(self.photo, cmap='gray')
            axs[1].imshow(self.blobs, cmap='gray')
            if has_obverse:
                axs[1].imshow(self.CC == self.obverse_idx, cmap='jet', alpha=0.5)
                axs[2].imshow(self.obverse, cmap='gray')
                axs[3].imshow(self.obverse_mask, cmap='gray')
            if self.has_lineart:
                axs[-1].imshow(self.lineart, cmap='gray')

            if self.has_id:
                fig.suptitle(self.ID)
            
            return fig, axs
