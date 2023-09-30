from PIL import Image
import numpy as np
import cv2
from tqdm.auto import tqdm
from itertools import product


def crop(img):
    xzeros = (img == 0).all(axis=0)
    yzeros = (img == 0).all(axis=1)
    x0, x1 = np.where(~xzeros)[0][[0, -1]]
    y0, y1 = np.where(~yzeros)[0][[0, -1]]
    return img[y0:y1, x0:x1].copy(), (y0, y1), (x0, x1)

class ColorbarTemplate:
    
    def __init__(self, fn='data/colorbar.png'):
        
        self.fn = fn
        self.img = np.asarray(Image.open(self.fn))
        self.img = self.img[..., 0]
        
        self.h, self.w = self.img.shape
        
    def get_template(self, scale=1., hscale=1., wscale=1., ref_shape=None):
        r = 1.
        if ref_shape is not None:
            r = min(
                ref_shape[0] / self.h,
                ref_shape[1] / self.w
            )
        newh = int(self.h * scale * hscale * r)
        neww = int(self.w * scale * wscale * r)
        dims = (neww, newh)
        return cv2.resize(self.img, dims)
    
    def match_template(self, ref_img, viz=False, **kwargs):
        
        template = self.get_template(ref_shape=ref_img.shape, **kwargs)
        h, w = template.shape
        
        res = cv2.matchTemplate(ref_img, template, cv2.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        if viz:
            img_res = ref_img.copy()
            cv2.rectangle(img_res, top_left, bottom_right, 255, 2) # only draw border
            
            return max_val, top_left, bottom_right, img_res
        
        return max_val, top_left, bottom_right
    
    def match_template_multiscale(self, ref_img, viz=False, pbar=False):
        scales = np.linspace(0.1, 1, 25)
        hscales = np.linspace(0.5, 1, 3)
        wscales = np.linspace(0.5, 1, 3)
        
        max_val = 0
        best_scale = None
        best_top_left = None
        best_bottom_right = None
        
        it = (
            tqdm(list(product(scales, hscales, wscales)))
            if pbar
            else product(scales, hscales, wscales)
        )
        for s, hs, ws in it:
            val, top_left, bottom_right = self.match_template(ref_img, scale=s, hscale=hs, wscale=ws)
            if val > max_val:
                max_val = val
                best_scale = {'scale': s, 'hscale': hs, 'wscale': ws}
                best_top_left = top_left
                best_bottom_right = bottom_right
        
        if viz:
            _, _, _, img_res = self.match_template(ref_img, viz=True, **best_scale)
            return max_val, best_top_left, best_bottom_right, best_scale, img_res
        
        return max_val, best_top_left, best_bottom_right, best_scale
    
    def erase_from_image(self, ref_img, threshold=0.8, crop_and_resize=True):
        max_val, top_left, bottom_right, best_scale = self.match_template_multiscale(ref_img)
        if max_val < threshold:
            return ref_img.copy()
        output = ref_img.copy()
        cv2.rectangle(output, top_left, bottom_right, 0, -1) # filled with black
        
        if crop_and_resize:
            shape = output.shape
            output, _, _ = crop(output)
            output = cv2.resize(output, shape)
        
        return output
