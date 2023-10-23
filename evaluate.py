# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import importlib
import os
import argparse
import copy
import random
import sys
import json
import time
#from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from core.utils import create_random_shape_with_random_motion

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor
from model.i3d import InceptionI3d
from scipy import linalg
from model.MiDaS.midas.dpt_depth import DPTDepthModel
from model.MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from model.DGDVI import DepthCompletion, ContentReconstruction, ContentEnhancement


parser = argparse.ArgumentParser(description="DGDVI")
parser.add_argument("-v", "--video", type=str, required=False)
parser.add_argument("-m", "--mask",   type=str, required=False)
parser.add_argument("--stage1_model_path", type=str, default="checkpoints/stage1.pth")
parser.add_argument("--stage2_model_path", type=str, default="checkpoints/stage2.pth")
parser.add_argument("--stage3_model_path", type=str, default="checkpoints/stage3.pth")
parser.add_argument("--midas_path", type=str, default='checkpoints/dpt_large-midas-2f21e586.pt')
parser.add_argument("--dataset", type=str, default='davis')
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
parser.add_argument("--generate_videos", action='store_true')
args = parser.parse_args()

w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps
i3d_model = None

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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)

def get_fid_score(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)

def init_i3d_model():
    global i3d_model
    if i3d_model is not None:
        return

    print("[Loading I3D model for FID score ..]")
    i3d_model_weight = './checkpoints/i3d_rgb_imagenet.pt'
    #if not os.path.exists(i3d_model_weight):
    #    os.mkdir(os.path.dirname(i3d_model_weight))
    #    urllib.request.urlretrieve('http://www.cmlab.csie.ntu.edu.tw/~zhe2325138/i3d_rgb_imagenet.pt', i3d_model_weight)
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_weight))
    i3d_model.to(torch.device('cuda:0'))

def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    """
    init_i3d_model()
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat

def get_frame_mask_list(args):
    if args.dataset == 'davis':
        data_root = "./data/DAVIS"
        mask_dir = "./data/dataset_masks/davis/test_masks"
        frame_dir = os.path.join(data_root, "JPEGImages", "480p")
    elif args.dataset == 'youtubevos':
        data_root = "./data/YouTubeVOS/"
        mask_dir = "./data/dataset_masks/youtube-vos/test_masks"
        frame_dir = os.path.join(data_root, "test_all_frames", "JPEGImages")

    mask_folder = sorted(os.listdir(mask_dir))
    mask_list = [os.path.join(mask_dir, name) for name in mask_folder]
    frame_folder = sorted(os.listdir(frame_dir))
    frame_list = [os.path.join(frame_dir, name) for name in mask_folder]

    print("[Finish building dataset {}]".format(args.dataset))
    return frame_list, mask_list

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
                ref_index.append(i)
                if len(ref_index) >= num_ref:
                    break
    return ref_index


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
def read_frame_from_videos(vname):

    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [vname+'/'+name for name in lst]
    frames = []
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w,h)))
    return frames       

def create_square_masks(video_length, h, w):
    masks = []
    for i in range(video_length):
        this_mask = np.zeros((h, w))
        this_mask[int(h/4):h-int(h/4), int(w/4):w-int(w/4)] = 1
        this_mask = Image.fromarray((this_mask*255).astype(np.uint8))
        masks.append(this_mask.convert('L'))
    return masks

def get_res_list(dir):
    folders = sorted(os.listdir(dir))
    return [os.path.join(dir, f) for f in folders]

def _to_tensor(depths):
    b,t,c,h,w = depths.shape
    depths = depths.view(b,t,c,-1)
    depth_min = torch.min(depths, dim=-1)[0].unsqueeze(-1)
    depth_max = torch.max(depths, dim=-1)[0].unsqueeze(-1)

    out = (depths - depth_min) / (depth_max - depth_min)
    out = out.view(b,t,c,h,w)
    return out

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

    frame_list, mask_list = get_frame_mask_list(args)
    assert len(frame_list) == len(mask_list)
    #print(len(frame_list))
    #print(len(mask_list))
    video_num = len(frame_list)

    ssim_all, psnr_all, len_all = 0., 0., 0.
    s_psnr_all = 0.
    video_length_all = 0
    vfid = 0.
    output_i3d_activations = []
    real_i3d_activations = []
    time_all = 0

    if not os.path.exists('results'):
        os.mkdir('results')
    dump_results_dir = os.path.join('results', f'{args.dataset}')
    if args.dump_results:
        if not os.path.exists(dump_results_dir):
            os.mkdir(dump_results_dir)
    for video_no in range(video_num):
        time_start = time.time()
        print("[Processing no.{}: {}]".format(video_no,frame_list[video_no].split("/")[-1]))
        if args.dump_results:
            this_dump_results_dir = os.path.join(dump_results_dir, frame_list[video_no].split("/")[-1])
            os.makedirs(this_dump_results_dir, exist_ok=True)
            this_dump_depth_dir = os.path.join("depth", frame_list[video_no].split("/")[-1])
            os.makedirs(this_dump_depth_dir, exist_ok=True)

        frames_PIL = read_frame_from_videos(frame_list[video_no])
        video_length = len(frames_PIL)
        imgs = _to_tensors(frames_PIL).unsqueeze(0)*2-1
        frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

        masks = read_mask(mask_list[video_no])    
        binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
        masks = _to_tensors(masks).unsqueeze(0)
    
        imgs, masks = imgs.to(device), masks.to(device)
        comp_frames = [None]*video_length
        comp_depths = [None]*video_length

        
        if args.generate_videos:
            #if not os.path.exists("result"):
            #    os.mkdir("result")
            name = frame_list[video_no].split("/")[-1]
            writer = cv2.VideoWriter(f"{dump_results_dir}/{name}_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
            for f in range(0,video_length):
                frame = frames[f].astype(np.double) + binary_masks[f].astype(np.double)*30
                frame = np.clip(frame,0.,255.)
                writer.write(cv2.cvtColor(np.array(frame).astype(np.uint8), cv2.COLOR_BGR2RGB))
            writer.release()

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
        time_end = time.time()
        time_all += (time_end - time_start)
        if args.generate_videos:
            name = frame_list[video_no].split("/")[-1]
            writer = cv2.VideoWriter(f"{dump_results_dir}/{name}_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
            for f in range(video_length):
                comp = np.array(comp_frames[f]).astype(
                    np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
                writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
            writer.release()
            writer = cv2.VideoWriter(f"{dump_results_dir}/{name}_result_mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
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
        ssim, psnr, s_psnr = 0., 0., 0.
        comp_PIL = []
        frames_PIL = []
        for ori, comp in zip(frames, comp_frames):
            frames_PIL.append(Image.fromarray(ori.astype(np.uint8)))
            comp_PIL.append(Image.fromarray(comp.astype(np.uint8)))
        for f in range(video_length):
            comp = comp_frames[f]
            comp = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)
            comp_depth = comp_depths[f]
            comp_depth = np.array(comp_depth)

            if args.dump_results:
                cv2.imwrite(os.path.join(this_dump_results_dir, "{:04}.png".format(f)), comp)
                cv2.imwrite(os.path.join(this_dump_depth_dir, "{:04}.png".format(f)), comp_depth)
            gt = cv2.cvtColor(np.array(frames[f]).astype(np.uint8), cv2.COLOR_BGR2RGB)
            ssim += compare_ssim(comp, gt, data_range=255, multichannel=True, win_size=65)
            s_psnr += compare_psnr(gt, comp, data_range=255)

        ssim_all += ssim
        s_psnr_all += s_psnr
        video_length_all += (video_length)
        #if video_no % 50 ==1:
        print("ssim {}, psnr {}".format(ssim_all/video_length_all, s_psnr_all/video_length_all))
        
        # FVID computation
        imgs = _to_tensors(comp_PIL).unsqueeze(0).to(device)
        gts = _to_tensors(frames_PIL).unsqueeze(0).to(device)
        output_i3d_activations.append(get_i3d_activations(imgs).cpu().numpy().flatten())
        real_i3d_activations.append(get_i3d_activations(gts).cpu().numpy().flatten())
    fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
    time_ave = time_all/video_length_all
    print(video_length_all)
    print("[Finish evaluating, ssim is {}, psnr is {}]".format(ssim_all/video_length_all, s_psnr_all/video_length_all))
    print("[fvid score is {}]".format(fid_score))
    print("[Average time(seconds) per frame is {}]".format(time_ave))


if __name__ == '__main__':
    main_worker()
