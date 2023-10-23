# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import importlib
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor
from model.i3d import InceptionI3d
from scipy import linalg
from model.MiDaS.midas.dpt_depth import DPTDepthModel
from model.MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from model.DGDVI import DepthCompletion, ContentReconstruction, ContentEnhancement

parser = argparse.ArgumentParser(description="DGDVI")
parser.add_argument("--video_path", type=str, default='data/DAVIS/JPEGImages/480p/surf')
parser.add_argument("--mask_path", type=str, default='data/dataset_masks/davis/test_masks/surf')
parser.add_argument("--stage1_model_path", type=str, default="checkpoints/stage1.pth")
parser.add_argument("--stage2_model_path", type=str, default="checkpoints/stage2.pth")
parser.add_argument("--stage3_model_path", type=str, default="checkpoints/stage3.pth")
parser.add_argument("--midas_path", type=str, default='checkpoints/dpt_large-midas-2f21e586.pt')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--outw", type=int, default=432)
parser.add_argument("--outh", type=int, default=240)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=6)
parser.add_argument("--savefps", type=int, default=24)
parser.add_argument("--use_mp4", action='store_true')
parser.add_argument("--dump_results", action='store_true')
args = parser.parse_args()


w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def midas_transform():
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
        transform = transforms.Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
        return transform


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index


def _to_tensor(depths):
    b,t,c,h,w = depths.shape
    depths = depths.view(b,t,c,-1)
    depth_min = torch.min(depths, dim=-1)[0].unsqueeze(-1)
    depth_max = torch.max(depths, dim=-1)[0].unsqueeze(-1)

    out = (depths - depth_min) / (depth_max - depth_min)
    out = out.view(b,t,c,h,w)
    return out


# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video 
def read_frame_from_videos(args):
    vname = args.video_path
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
    return frames       

def main_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set up stage-1 (depth compeltion) pretrained model
    stage1_model = DepthCompletion().to(device)
    data = torch.load(args.stage1_model_path, map_location=device)
    stage1_model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.stage1_model_path))
    stage1_model.eval()
    # set up stage-2 (content reconstruction) pretrained model
    stage2_model = ContentReconstruction().to(device)
    data = torch.load(args.stage2_model_path, map_location=device)
    stage2_model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.stage2_model_path))
    stage2_model.eval()
    # set up stage-3 (content enhancement) pretrained model
    stage3_model = ContentEnhancement().to(device)
    data = torch.load(args.stage3_model_path, map_location=device)
    stage3_model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.stage3_model_path))
    stage3_model.eval()

    midas = DPTDepthModel(
            path=args.midas_path,
            backbone="vitl16_384",
            non_negative=True,
        ).to(device)
    midas.eval()
    print('loading from: {}'.format(args.midas_path))

    frames_PIL = read_frame_from_videos(args)
    video_length = len(frames_PIL)
    imgs = _to_tensors(frames_PIL).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

    masks = read_mask(args.mask_path)    
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)

    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None]*video_length
    comp_depths = [None]*video_length

    if not os.path.exists('results'):
        os.mkdir('results')
    name = args.video_path.split("/")[-1]
    if not os.path.exists(os.path.join('results', f'{name}')):
        os.mkdir(os.path.join('results', f'{name}'))
    name = args.video_path.split("/")[-1]
    writer = cv2.VideoWriter(f"results/{name}/{name}_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(0,video_length):
        frame = frames[f].astype(np.double) + binary_masks[f].astype(np.double)*30
        frame = np.clip(frame,0.,255.)
        writer.write(cv2.cvtColor(np.array(frame).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(f"{name}_mask.mp4"))

    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        len_temp = len(neighbor_ids) + len(ref_ids)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
        selected_midas_imgs = torch.from_numpy(np.stack([midas_transform()({"image": np.array(frames[idx]).astype(np.float)/255.})["image"] for idx in neighbor_ids+ref_ids], axis=0)).unsqueeze(0).to(device)           
        with torch.no_grad():
            b, t, c, h, w = selected_midas_imgs.size()
            depths = midas.forward(selected_midas_imgs.view(b*t, c, h, w))
            b, t, c, h, w = selected_imgs.size()
            depths = torch.nn.functional.interpolate(
                depths,
                size=[h,w],
                mode="bicubic",
                align_corners=False).view(b,t,1,h,w)
            depths = _to_tensor(depths)
            pred_depths = stage1_model(depths*(1.-selected_masks).float(), selected_masks)
            input_imgs = selected_imgs*(1.-selected_masks).float()
            pred_img, feat = stage2_model(input_imgs, pred_depths.float())
            bt, c_, h_, w_ = feat.shape
            feat = feat.view(b, t, c_, h_, w_)
            feat = feat[:, :len(neighbor_ids), ...]
            pred_img = pred_img.view(b, t, c, h, w)
            pred_img = pred_img[:, :len(neighbor_ids), ...]
            selected_imgs = selected_imgs[:, :len(neighbor_ids), ...].contiguous()
            selected_masks = selected_masks[:, :len(neighbor_ids), ...].contiguous()
            pred_img = stage3_model(selected_imgs*(1.-selected_masks) + pred_img*selected_masks, feat)

            pred_img = (pred_img + 1.) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            depths = depths.view(b*t, 1, h, w).cpu().permute(0, 2, 3, 1).numpy()*255
            depths = depths.clip(0,255).astype(np.uint8)
            pred_depths = pred_depths.view(b*t, 1, h, w).cpu().permute(0, 2, 3, 1).numpy()*255
            pred_depths = pred_depths.clip(0,255).astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5

                depth = np.array(pred_depths[i]).astype(np.uint8)*binary_masks[idx] + depths[i] * (1-binary_masks[idx])
                if comp_depths[idx] is None:
                    comp_depths[idx] = depth
                else:
                    comp_depths[idx] = comp_depths[idx].astype(np.float32)*0.5 + depth.astype(np.float32)*0.5
    writer = cv2.VideoWriter(f"results/{name}/{name}_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        if w != args.outw:
            comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(f"{name}_result.mp4"))
    writer = cv2.VideoWriter(f"results/{name}/{name}_result_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.double)*binary_masks[f].astype(
            np.double) + frames[f].astype(
            np.double) * (1.-binary_masks[f].astype(
            np.double)) + binary_masks[f].astype(
            np.double) * 25
        comp = np.clip(comp,0,255)
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(f"{name}_result_mask.mp4"))

if __name__ == '__main__':
    main_worker()
