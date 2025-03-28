import numpy as np
import torch
import cv2
from torchvision import transforms
from network_weight import UNet
from network import UNet as HUNet
import argparse
from PIL import Image

"""
    Height and Weight Information from Unconstrained Images
    https://github.com/canaltinigne/DeepHeightWeight
    
    Run the following command to get mask, joint locations
    height and weight of a person in an image.
    (Image should include a person with full-body visible)

    python HWFinder.py -i [IMAGE ADDRESS] -g [GPU NUMBER] -r [RESOLUTION]
"""

def get_models():
    np.random.seed(23)
    
    # Height
    model_h = HUNet(128)
    pretrained_model_h = torch.load('attribute_identifier/height_and_weight/models/model_ep_48.pth.tar')

    # Weight
    model_w = UNet(128, 32, 32)
    pretrained_model_w = torch.load('attribute_identifier/height_and_weight/models/model_ep_37.pth.tar')
    
    model_h.load_state_dict(pretrained_model_h["state_dict"])
    model_w.load_state_dict(pretrained_model_w["state_dict"])

    return model_h, model_w

def _find_res(image_path):
    img = Image.open(image_path)
    width, height = img.size
    res = max([width, height])

    if res > 256:
        return 256
    elif res < 128:
        return 128
    return res

def get_height_and_weight(model_h, model_w, image_path):
    np.random.seed(23)
    # Reading Image 
    assert ".jpg" in image_path or ".png" in image_path or ".jpeg" in image_path, "Please use .jpg or .png format"
    
    RES = _find_res(image_path)
    
    X = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype('float32')
    scale = RES / max(X.shape[:2])
    
    X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
    
    if X_scaled.shape[1] > X_scaled.shape[0]:
        p_a = (RES - X_scaled.shape[0])//2
        p_b = (RES - X_scaled.shape[0])-p_a
        X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
    elif X_scaled.shape[1] <= X_scaled.shape[0]:
        p_a = (RES - X_scaled.shape[1])//2
        p_b = (RES - X_scaled.shape[1])-p_a
        X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
    
    o_img = X.copy()
    X /= 255
    X = transforms.ToTensor()(X).unsqueeze(0)
        
    if torch.cuda.is_available():
        X = X.cuda()
    
    if torch.cuda.is_available():
        model = model_w.cuda(0)
    else:
        model = model_w

    model.eval()
    with torch.no_grad():
        m_p, j_p, _, w_p = model(X)
    
    del model
    
    if torch.cuda.is_available():
        model = model_h.cuda(0)
    else:
        model = model_h
        
    model.eval()
    with torch.no_grad():
        _, _, h_p = model(X)
    
    del model

    return 100*h_p.item(), 100*w_p.item()