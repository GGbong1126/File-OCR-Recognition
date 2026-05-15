# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 21:09:21 2025

@author: asus no.2
"""

# -*- coding: utf-8 -*-
#
# V13.0 HDC去噪增强版
# [核心改进]: 引入 HDC (Hybrid Dilated Convolution) 解码模块，使用 1-2-5 混合空洞率。
# [目的]: 极大增加感受野，通过上下文信息消除背景中的顽固噪点，同时避免网格效应。
# [基础]: 包含 V12.5 的所有稳定性修复。
#

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF 
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from tqdm import tqdm
import random
import glob
import time
import sys
import math

# ==============================================================================
# 1. ATTENTION & BASE BLOCKS
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_planes = max(in_planes // ratio, 4)
        self.fc1   = nn.Conv2d(in_planes, mid_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(mid_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        mid_channels = 256 
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv3x3_rate6 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv3x3_rate12 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv3x3_rate18 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(nn.Conv2d(mid_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        input_size = x.shape[2:] 
        b1 = self.conv1x1(x); b2 = self.conv3x3_rate6(x); b3 = self.conv3x3_rate12(x); b4 = self.conv3x3_rate18(x)
        b5 = self.global_pool(x); b5 = F.interpolate(b5, size=input_size, mode='bilinear', align_corners=False)
        return self.conv_out(torch.cat([b1, b2, b3, b4, b5], dim=1))

# ==============================================================================
# 2. HDC DECODER BLOCKS (NEW)
# ==============================================================================
class HDCDecoderBlock(nn.Module):
    """
    Hybrid Dilated Convolution Decoder Block
    策略: Dilation 1 -> 2 -> 5
    作用: 逐层扩大感受野，捕获长距离上下文信息以消除噪点，同时避免网格效应。
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(HDCDecoderBlock, self).__init__()
        
        # Layer 1: Standard Conv (Detail)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Dilated Conv (Context)
        self.conv2 = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3: Large Dilated Conv (Global Context for Denoising)
        self.conv3 = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cbam = CBAM(out_channels)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        if x.shape[2:] != skip.shape[2:]: 
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cbam(x)
        return self.upsample(x)

class FusedHDCDecoderBlock(nn.Module):
    """
    用于最深层的融合解码块，也引入了轻量级 HDC (1->2) 以保持一致性。
    """
    def __init__(self, in_bottleneck_channels, in_skip_channels, middle_channels, out_channels):
        super().__init__()
        self.upsample_bottleneck = nn.ConvTranspose2d(in_bottleneck_channels, in_bottleneck_channels, kernel_size=2, stride=2)
        
        # Layer 1: Fusion & Detail
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_bottleneck_channels + in_skip_channels, middle_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Context (Dilation 2) - 稍微扩大感受野
        self.conv2 = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cbam = CBAM(out_channels)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x_bottleneck, skip_feature):
        x_upsampled = self.upsample_bottleneck(x_bottleneck)
        if x_upsampled.shape[2:] != skip_feature.shape[2:]: 
            x_upsampled = nn.functional.interpolate(x_upsampled, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
        
        x_fused = torch.cat((x_upsampled, skip_feature), dim=1)
        x = self.conv1(x_fused)
        x = self.conv2(x)
        x = self.cbam(x)
        return self.upsample(x)

# ==============================================================================
# 3. NETWORK ARCHITECTURE
# ==============================================================================
class MobileNetV3_Unet(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3_Unet, self).__init__()
        try: 
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.mobilenet_v3 = models.mobilenet_v3_large(weights=weights)
        except AttributeError: 
            self.mobilenet_v3 = models.mobilenet_v3_large(pretrained=pretrained)
        
        self.mobilenet_v3.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_layer1 = self.mobilenet_v3.features[0:2]
        self.encoder_layer2 = self.mobilenet_v3.features[2:4]
        self.encoder_layer3 = self.mobilenet_v3.features[4:7]
        self.encoder_layer4 = self.mobilenet_v3.features[7:13]
        self.bottleneck_layer = self.mobilenet_v3.features[13:17]
        
        self.aspp = ASPP(in_channels=960, out_channels=256)
        
        # 使用新的 HDC Decoder Block
        self.decoder_stage4 = FusedHDCDecoderBlock(256, 112, 512, 256)
        self.decoder_stage3 = HDCDecoderBlock(256 + 40, 256, 128)
        self.decoder_stage2 = HDCDecoderBlock(128 + 24, 128, 64)
        self.decoder_stage1 = HDCDecoderBlock(64 + 16, 64, 32)
        
        # 1-Channel Output
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1) 
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        skip1 = self.encoder_layer1(x)
        skip2 = self.encoder_layer2(skip1)
        skip3 = self.encoder_layer3(skip2)
        skip4 = self.encoder_layer4(skip3)
        bottleneck_features = self.bottleneck_layer(skip4)
        bottleneck_out = self.aspp(bottleneck_features)
        
        d4 = self.decoder_stage4(bottleneck_out, skip4)
        d3 = self.decoder_stage3(d4, skip3)
        d2 = self.decoder_stage2(d3, skip2)
        d1 = self.decoder_stage1(d2, skip1)
        return self.final_activation(self.final_conv(d1))

class pix2pixD_256(nn.Module):
    def __init__(self):
        super(pix2pixD_256, self).__init__()
        def base_Conv_bn_lkrl(in_channels, out_channels, stride=2): 
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1), nn.BatchNorm2d(out_channels, momentum=0.8), nn.LeakyReLU(0.2, inplace=True))
        self.D_model = nn.Sequential(
            nn.Conv2d(4, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            base_Conv_bn_lkrl(64, 128), base_Conv_bn_lkrl(128, 256), base_Conv_bn_lkrl(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1), 
        )
    def forward(self, x1, x2): return self.D_model(torch.cat([x1, x2], dim=1))

# ==============================================================================
# 4. DATASET & UTILS
# ==============================================================================
def split_data(dir_root):
    image_files = set()
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        for case_ext in [ext.lower(), ext.upper()]:
             for f in glob.glob(os.path.join(dir_root, case_ext)): image_files.add(os.path.normpath(os.path.abspath(f)).lower())
    files = list(image_files); random.shuffle(files)
    return files

class CreateDatasets(Dataset):
    def __init__(self, image_path_list, img_size=512, augment=False):
        self.image_path_list = image_path_list
        self.img_size = img_size
        self.augment = augment
        self.clean_folder_candidates = ['clean', 'cleanval', 'Clean', 'CleanVal']
        
        if self.augment: 
            self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
            
        self.resize_bicubic = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC)
        self.resize_nearest = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()

    def __len__(self): 
        return len(self.image_path_list)
        
    def __getitem__(self, item):
        if item >= len(self.image_path_list): return None, None
        image_item_path = self.image_path_list[item]
        try:
            image_pil = Image.open(image_item_path).convert('RGB') 
            noise_dir = os.path.dirname(image_item_path)
            base_dir = os.path.dirname(noise_dir)
            base_name_no_ext, _ = os.path.splitext(os.path.basename(image_item_path))
            label_item_path = None
            
            for clean_folder in self.clean_folder_candidates:
                clean_dir = os.path.join(base_dir, clean_folder)
                if os.path.exists(clean_dir):
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG']:
                        p = os.path.join(clean_dir, base_name_no_ext + ext)
                        if os.path.exists(p): label_item_path = p; break
                if label_item_path: break
                
            if label_item_path is None: return None, None
            label_pil = Image.open(label_item_path).convert('L') 
            
            if self.augment:
                # 1. 光照与色彩扰动 (保留原有)
                image_pil = self.color_jitter(image_pil)
                if random.random() > 0.5: 
                    random_gamma = random.uniform(0.8, 2.0)
                    image_pil = TF.adjust_gamma(image_pil, random_gamma)
                
                # 2. [新增] 随机高斯模糊: 模拟泡水墨水晕染、线条边缘模糊
                if random.random() > 0.6:
                    kernel_size = random.choice([3, 5]) # 随机核大小
                    sigma = random.uniform(0.5, 2.0)
                    image_pil = TF.gaussian_blur(image_pil, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

                # 3. [新增] 随机裁剪: 截取 60%~100% 区域，改变表格线间距和粗细
                if random.random() > 0.5:
                    # 获取随机裁剪的坐标和宽高
                    i, j, h, w = transforms.RandomResizedCrop.get_params(image_pil, scale=(0.6, 1.0), ratio=(0.9, 1.1))
                    image_pil = TF.crop(image_pil, i, j, h, w)
                    label_pil = TF.crop(label_pil, i, j, h, w)
                
                # 4. 几何翻转 (保留原有)
                if random.random() > 0.5: 
                    image_pil = TF.hflip(image_pil)
                    label_pil = TF.hflip(label_pil)
                if random.random() > 0.5: 
                    image_pil = TF.vflip(image_pil)
                    label_pil = TF.vflip(label_pil)
                    
                # 5. [新增] 随机正交旋转: 横竖线互换，直接针对表格特征翻倍数据
                if random.random() > 0.5:
                    angle = random.choice([90, 180, 270])
                    image_pil = TF.rotate(image_pil, angle)
                    label_pil = TF.rotate(label_pil, angle)

            # 最后统一 Resize 回 512x512 并转张量
            image = self.to_tensor(self.resize_bicubic(image_pil))
            label = self.to_tensor(self.resize_nearest(label_pil))
            
            if image.shape[0] != 3 or label.shape[0] != 1: return None, None
            return image, label
            
        except Exception as e: 
            # 方便调试时查看报错
            # print(f"Error processing {image_item_path}: {e}")
            return None, None

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

class EdgeLoss(nn.Module):
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        k = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]; k_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.kx = torch.tensor(k, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.ky = torch.tensor(k_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.loss = nn.L1Loss()
    def forward(self, pred, target):
        c = pred.size(1); kx = self.kx.expand(c, 1, 3, 3); ky = self.ky.expand(c, 1, 3, 3)
        tgx = F.conv2d(target, kx, padding=1, groups=c); tgy = F.conv2d(target, ky, padding=1, groups=c)
        pgx = F.conv2d(pred, kx, padding=1, groups=c); pgy = F.conv2d(pred, ky, padding=1, groups=c)
        return self.loss(torch.abs(pgx)+torch.abs(pgy), torch.abs(tgx)+torch.abs(tgy))

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 100.0 if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))

# ==============================================================================
# 5. TRAIN LOOP
# ==============================================================================
def train_one_epoch(G, D, loader, opt_G, opt_D, dev, epoch, n_critic, grad_clip, amp):
    loss_G_acc, loss_D_acc = 0.0, 0.0
    G.train(); D.train()
    edge_fn = EdgeLoss(dev)
    l1_fn = nn.L1Loss()
    scaler_G = torch.amp.GradScaler(enabled=amp)
    scaler_D = torch.amp.GradScaler(enabled=amp)
    bar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    
    for idx, batch in enumerate(bar):
        if batch is None: continue 
        imgs, lbls = batch 
        real, target = imgs.to(dev), lbls.to(dev) 
        
        # Train D
        opt_D.zero_grad(set_to_none=True)
        with torch.amp.autocast(dev.type, enabled=amp):
            d_real = D(target, real); loss_d_real = -torch.mean(d_real)
            with torch.no_grad(): fake = G(real)
            d_fake = D(fake.detach(), real); loss_d_fake = torch.mean(d_fake)
            alpha = torch.rand((target.size(0), 1, 1, 1), device=dev)
            interp = (alpha * target + (1 - alpha) * fake.detach()).requires_grad_(True)
            d_interp = D(interp, real)
            grads = torch.autograd.grad(d_interp, interp, torch.ones_like(d_interp), create_graph=True, retain_graph=True)[0]
            gp = ((grads.view(target.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
            loss_D = loss_d_real + loss_d_fake + 10 * gp
        
        scaler_D.scale(loss_D).backward()
        if grad_clip > 0: scaler_D.unscale_(opt_D); torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
        scaler_D.step(opt_D); scaler_D.update(); loss_D_acc += loss_D.item()

        # Train G
        if idx % n_critic == 0:
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type, enabled=amp):
                fake_g = G(real)
                d_fake_g = D(fake_g, real); loss_adv = -torch.mean(d_fake_g)
                l1 = l1_fn(fake_g, target)
                edge = edge_fn(fake_g, target)
                eps = 1e-7; cl = torch.clamp(fake_g, eps, 1-eps)
                bce = -(target * torch.log(cl) + (1-target) * torch.log(1-cl)).mean()
                
                # 如果去噪效果仍不理想，可以尝试增加 L1/BCE 权重，减少 loss_adv
                loss_G = loss_adv + l1 * 15 + bce * 20 + edge * 5
            
            scaler_G.scale(loss_G).backward()
            if grad_clip > 0: scaler_G.unscale_(opt_G); torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            scaler_G.step(opt_G); scaler_G.update(); loss_G_acc += loss_G.item()
            bar.set_postfix({"G": f"{loss_G.item():.3f}", "D": f"{loss_D.item():.3f}"})
            
    return loss_G_acc / (len(loader)+1e-5), loss_D_acc / (len(loader)+1e-5)

def train(opt):
    savePath = opt.savePath; os.makedirs(savePath, exist_ok=True)
    
    # [新增] 定义日志文件路径
    log_file_path = os.path.join(savePath, "train_log.txt")
    # 如果是第一次运行，写入表头
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("Epoch,Loss_G,Loss_D,Val_PSNR,Time_Sec\n")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = opt.use_amp and dev.type == 'cuda'
    
    train_root_dir = os.path.dirname(opt.pretrainDataPath.rstrip(os.sep))
    val_noise_path = opt.valDataPath if opt.valDataPath else os.path.join(train_root_dir, 'val', 'noiseval')
    print(f"Train Path: {opt.pretrainDataPath}"); print(f"Val Path: {val_noise_path}")
    
    val_files = []; val_loader = None
    if os.path.exists(val_noise_path):
        val_files = split_data(val_noise_path)
        if len(val_files) > 0:
            print(f"Val Images: {len(val_files)}")
            val_ds = CreateDatasets(val_files, img_size=opt.img_size, augment=False)
            val_loader = DataLoader(val_ds, batch_size=opt.batch, shuffle=False, collate_fn=custom_collate_fn)
    else: print("Validation set not found. Skipping validation.")
    
    train_files = split_data(opt.pretrainDataPath)
    if not train_files: print("No training files found!"); return
    train_ds = CreateDatasets(train_files, img_size=opt.img_size, augment=opt.augment)
    train_loader = DataLoader(train_ds, batch_size=opt.batch, shuffle=True, num_workers=4, drop_last=True, collate_fn=custom_collate_fn)

    pix_G = MobileNetV3_Unet(pretrained=True).to(dev)
    pix_D = pix2pixD_256().to(dev)

    if opt.freeze_backbone:
        print("\n\033[93m>>> [Frozen Mode] Freezing backbone... \033[0m")
        for param in pix_G.parameters(): param.requires_grad = False
        for param in pix_G.final_conv.parameters(): param.requires_grad = True
        for param in pix_G.decoder_stage1.parameters(): param.requires_grad = True
        for param in pix_G.decoder_stage2.parameters(): param.requires_grad = True

    optim_G = optim.Adam(filter(lambda p: p.requires_grad, pix_G.parameters()), lr=opt.lrG, betas=(0.5, 0.999))
    optim_D = optim.Adam(pix_D.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
    
    start_epoch = 0
    best_psnr = 0.0

    last_model_path = os.path.join(savePath, f"{opt.modelPrefix}_last.pth")
    if os.path.isfile(last_model_path):
        print(f"\n\033[94m>>> Resuming from checkpoint: {last_model_path} \033[0m")
        try:
            ckpt = torch.load(last_model_path, map_location=dev, weights_only=True)
            pix_G.load_state_dict(ckpt['G_model'], strict=False)
            pix_D.load_state_dict(ckpt['D_model'])
            try: optim_G.load_state_dict(ckpt['optim_G'])
            except: pass
            try: optim_D.load_state_dict(ckpt['optim_D'])
            except: pass
            start_epoch = ckpt.get('epoch', 0) + 1
            best_psnr = ckpt.get('best_psnr', 0.0)
            print(f">>> Continued from Epoch {start_epoch}, Best PSNR: {best_psnr:.2f}")
        except Exception as e:
            print(f">>> Resume failed: {e}. Starting fresh.")
    elif opt.pretrainWeight:
        try:
            print(f"\n>>> Loading Pretrained Weights: {opt.pretrainWeight}")
            ckpt = torch.load(opt.pretrainWeight, map_location=dev, weights_only=True)
            sd = ckpt['G_model'] if 'G_model' in ckpt else ckpt
            model_dict = pix_G.state_dict()
            pretrained_dict = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            pix_G.load_state_dict(model_dict)
            print(f">>> Matched Layers: {len(pretrained_dict)} / {len(model_dict)}")
        except Exception as e: print(f"Weight Load Failed: {e}")

    for epoch in range(start_epoch, opt.epoch):
        t0 = time.time()
        lG, lD = train_one_epoch(pix_G, pix_D, train_loader, optim_G, optim_D, dev, epoch, opt.n_critic, opt.grad_clip, use_amp)
        
        val_psnr = 0.0
        if val_loader:
            pix_G.eval()
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    if imgs is None: continue
                    fake = pix_G(imgs.to(dev))
                    val_psnr += calculate_psnr(fake, lbls.to(dev)).item()
            val_psnr /= len(val_loader)
            pix_G.train()
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({'G_model': pix_G.state_dict(), 'D_model': pix_D.state_dict()}, os.path.join(savePath, f"{opt.modelPrefix}_best.pth"))

        time_cost = time.time()-t0
        print(f"Epoch {epoch} | G: {lG:.4f} D: {lD:.4f} | Val PSNR: {val_psnr:.2f} | Time: {time_cost:.1f}s")
        
        # [新增] 将数据追加写入到 txt 文件
        with open(log_file_path, "a") as f:
            f.write(f"{epoch},{lG:.4f},{lD:.4f},{val_psnr:.2f},{time_cost:.1f}\n")
        
        save_dict = {
            'G_model': pix_G.state_dict(), 'D_model': pix_D.state_dict(),
            'optim_G': optim_G.state_dict(), 'optim_D': optim_D.state_dict(),
            'epoch': epoch, 'best_psnr': best_psnr
        }
        torch.save(save_dict, last_model_path)

def cfg():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrainDataPath', type=str, required=True)
    p.add_argument('--valDataPath', type=str, default='')
    p.add_argument('--pretrainWeight', type=str, default='')
    p.add_argument('--savePath', type=str, default='./weights_finetune')
    p.add_argument('--modelPrefix', type=str, default='mnet_hdc_real')
    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lrG', type=float, default=1e-5)
    p.add_argument('--lrD', type=float, default=1e-5)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--n_critic', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--freeze_backbone', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    train(cfg())