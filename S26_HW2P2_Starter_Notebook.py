# @title Click to acknowledge then run cell!
ACKNOWLEDGED = True # @param {"type":"boolean","placeholder":"False"}

import os
import pprint
os.environ['KAGGLE_USERNAME'] = "avidube"
os.environ['KAGGLE_API_TOKEN'] = "KGAT_e4a59dacaeaefc1cf225368be254aeae"

# Verify
import kaggle
api = kaggle.api  # Already authenticated on import
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision
from torchvision.io import decode_image
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.v2 as T
import gc
from tqdm import tqdm
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics as mt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import glob
import wandb
import matplotlib.pyplot as plt
from pytorch_metric_learning import samplers
import csv
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

config = {
    'data_root': "/home/avid/Intro_Deep_Learning/hw2p2_data",
    'batch_size': 256,
    'lr': 0.05,
    'epochs': 40,
    'num_classes': 8631,
    'checkpoint_dir': "/home/avid/Intro_Deep_Learning/hw2_finetuning_checkpoint",
    'augment': True,
    'embed_dim'      : 256,
    'arc_s'          : 64.0,
    'arc_m'          : 0.45,
    'weight_decay'   : 1e-4,
    'warmup_epochs'  : 2,
    'num_workers'   : 4,
}

def create_transforms(image_size: int = 112, augment: bool = True) -> T.Compose:
    transform_list = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.ToDtype(torch.float32, scale=True),
    ]

    if augment:
        transform_list.extend([
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation((-15, 15)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.1),
        ])

    transform_list.extend([
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if augment:
        transform_list.extend([
            T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
        ])
    return T.Compose(transform_list)

train_transforms = create_transforms(augment=config['augment'])
val_transforms   = create_transforms(augment=False)


class ImageDataset(Dataset):
    def __init__(self, root, transform, num_classes=None, preload=False):
        self.root = root
        self.transform = transform
        self.preload = preload

        self.image_paths = []
        self.labels = None
        self.classes = None
        self.images = []

        labels_file = os.path.join(self.root, "labels.txt")
        images_dir = os.path.join(self.root, "images")

        has_labels = os.path.exists(labels_file)

        if has_labels:
            self.labels = []
            self.classes = set()

            with open(labels_file, "r") as f:
                lines = f.readlines()

            lines = sorted(lines, key=lambda x: int(x.strip().split(" ")[-1]))
            all_labels = sorted(set(int(line.strip().split(" ")[1]) for line in lines))

            if num_classes is not None:
                selected_classes = set(all_labels[:num_classes])
            else:
                selected_classes = set(all_labels)

            for line in tqdm(lines, desc="Loading labeled dataset"):
                img_path, label = line.strip().split(" ")
                label = int(label)

                if label in selected_classes:
                    self.image_paths.append(os.path.join(images_dir, img_path))
                    self.labels.append(label)
                    self.classes.add(label)

            self.classes = sorted(self.classes)
            assert len(self.image_paths) == len(self.labels), "Images and labels mismatch!"

        else:
            image_files = sorted(os.listdir(images_dir))
            for img in tqdm(image_files, desc="Loading unlabeled dataset"):
                self.image_paths.append(os.path.join(images_dir, img))

        if self.preload:
            self.images = [
                decode_image(p, mode="RGB") / 255.0
                for p in tqdm(self.image_paths, desc="Preloading images")
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.preload:
            image = self.images[idx]
        else:
            image = decode_image(self.image_paths[idx], mode="RGB") / 255.0

        image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image


gc.collect()
cls_data_dir = config['data_root'] + '/cls_data'

cls_train_dataset = ImageDataset(os.path.join(cls_data_dir, "train"), train_transforms, config['num_classes'], False)
cls_val_dataset   = ImageDataset(os.path.join(cls_data_dir, "dev"), val_transforms, config['num_classes'], False)
cls_test_dataset  = ImageDataset(os.path.join(cls_data_dir, "test"), val_transforms, config['num_classes'], False)

assert cls_train_dataset.classes == cls_val_dataset.classes == cls_test_dataset.classes, "Class mismatch!"

cls_train_loader = DataLoader(cls_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
cls_val_loader   = DataLoader(cls_val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
cls_test_loader  = DataLoader(cls_test_dataset, batch_size=config['batch_size'], shuffle=False)


class ImagePairDataset(Dataset):
    def __init__(self, root, pairs_file, transform, preload=False):
        self.root = root
        self.transform = transform
        self.preload = preload

        self.image1_paths = []
        self.image2_paths = []
        self.matches = None

        with open(pairs_file, "r") as f:
            lines = f.readlines()

        first_cols = lines[0].strip().split()
        has_labels = len(first_cols) == 3

        if has_labels:
            self.matches = []

        for line in tqdm(lines, desc="Loading dataset"):
            parts = line.strip().split()

            if has_labels:
                img1, img2, match = parts
                self.matches.append(int(match))
            else:
                img1, img2 = parts

            self.image1_paths.append(os.path.join(self.root, img1))
            self.image2_paths.append(os.path.join(self.root, img2))

        assert len(self.image1_paths) == len(self.image2_paths)
        if has_labels:
            assert len(self.matches) == len(self.image1_paths)

        if self.preload:
            self.image1_cache = [
                decode_image(p, mode="RGB") / 255.0
                for p in tqdm(self.image1_paths, desc="Preloading image1")
            ]
            self.image2_cache = [
                decode_image(p, mode="RGB") / 255.0
                for p in tqdm(self.image2_paths, desc="Preloading image2")
            ]

    def __len__(self):
        return len(self.image1_paths)

    def __getitem__(self, idx):
        if self.preload:
            img1 = self.image1_cache[idx]
            img2 = self.image2_cache[idx]
        else:
            img1 = decode_image(self.image1_paths[idx], mode="RGB") / 255.0
            img2 = decode_image(self.image2_paths[idx], mode="RGB") / 255.0

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        if self.matches is not None:
            return img1, img2, self.matches[idx]
        else:
            return img1, img2


gc.collect()
ver_data_dir = config['data_root'] + '/ver_data'

ver_val_dataset  = ImagePairDataset(ver_data_dir, os.path.join(config['data_root'], "val_pairs.txt"), val_transforms, False)
ver_test_dataset = ImagePairDataset(ver_data_dir, os.path.join(config['data_root'], "test_pairs.txt"), val_transforms, False)

ver_val_loader   = DataLoader(ver_val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
ver_test_loader  = DataLoader(ver_test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)

class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.se    = SEBlock(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + self.shortcut(x))


class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        self.easy_margin = easy_margin

    def forward(self, feats, labels=None):
        with torch.autocast(device_type='cuda', enabled=False):
            feats  = feats.float()
            weight = F.normalize(self.weight.float())
            
            cosine = F.linear(F.normalize(feats), weight)

            if labels is None:
                return cosine * self.s

            sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
            
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
            
            logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            
            logits *= self.s
            
            return logits


class ResNet(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 512,
                 arc_s: float = 64.0, arc_m: float = 0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            SEResBlock(128, 128, stride=1),
            SEResBlock(128, 128, stride=1)
        )
        self.stage2 = nn.Sequential(
            SEResBlock(128, 256, stride=2),
            SEResBlock(256, 256, stride=1)
        )
        self.stage3 = nn.Sequential(
            SEResBlock(256, 512, stride=2),
            SEResBlock(512, 512, stride=1)
        )
        self.stage4 = nn.Sequential(
            SEResBlock(512, 1024, stride=2),
        )

        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten   = nn.Flatten()
        self.embedding = nn.Sequential(
            nn.Linear(1024, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim)
        )
        # Uncomment when fine tuning with ArcFace
        self.arc = ArcFaceLayer(embed_dim, num_classes, s=arc_s, m=arc_m)
        # self.classifier = nn.Linear(embed_dim, num_classes) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None, return_feats: bool = False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.flatten(self.pool(x))
        feats = self.embedding(x)

        if return_feats:
            return {"feats": feats}
        # Uncomment when fine tuning with ArcFace
        arc_logits = self.arc(feats, labels)
        # out = self.classifier(feats)
        return {"feats": feats, "out": arc_logits}


model = ResNet(
    num_classes = config['num_classes'],
    embed_dim   = config['embed_dim'],
    arc_s       = config['arc_s'],
    arc_m       = config['arc_m'],
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'], nesterov=True)

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_scheduler(optimizer, config['warmup_epochs'], config['epochs'])

scaler = torch.amp.GradScaler(device='cuda')


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    maxk = min(max(topk), logits.size(1))
    batch_size = targets.size(0)

    _, preds = logits.topk(maxk, dim=1, largest=True, sorted=True)
    preds = preds.t()
    correct = preds.eq(targets.view(1, -1))

    accuracies = []
    for k in topk:
        k = min(k, maxk)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accuracies.append(correct_k * 100.0 / batch_size)

    return accuracies


def verification_metrics(labels, scores, fpr_targets=None):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    fpr, tpr, thresholds = mt.roc_curve(labels, scores)
    roc_interp = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))

    eer = brentq(lambda x: 1.0 - x - roc_interp(x), 0.0, 1.0) * 100.0
    auc = mt.auc(fpr, tpr) * 100.0

    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    acc_scores = []
    for i in range(len(thresholds)):
        current_tpr = tpr[i]
        current_fpr = fpr[i]
        current_tnr = 1.0 - current_fpr
        current_acc = (current_tpr * pos_count + current_tnr * neg_count) / (pos_count + neg_count)
        acc_scores.append(current_acc)

    acc = np.max(acc_scores) * 100.0

    tpr_at_fpr = []
    if fpr_targets is not None:
        for fpr_val in fpr_targets:
            tpr_at_fpr.append(
                (f"TPR@FPR={fpr_val}", 100.0 * float(roc_interp(fpr_val)))
            )

    return {
        "ACC": acc,
        "EER": eer,
        "AUC": auc,
        "TPRs": tpr_at_fpr,
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    criterion,
):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(
        dataloader,
        desc="Train",
        dynamic_ncols=True,
        leave=False,
    )

    for images, labels in progress:
        optimizer.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type = 'cuda'):
            outputs = model(images, labels=labels)
            logits = outputs["out"]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        loss_meter.update(batch_loss)

        with torch.no_grad():
            batch_acc = topk_accuracy(logits, labels, topk=(1,))[0].item()
            acc_meter.update(batch_acc)

        progress.set_postfix(
            loss=f"{batch_loss:.4f} ({loss_meter.avg:.4f})",
            acc=f"{batch_acc:.2f}% ({acc_meter.avg:.2f}%)",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
        )

    if scheduler is not None:
        scheduler.step()

    return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def valid_epoch_cls(
    model,
    dataloader,
    device,
    criterion,
):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(
        dataloader,
        desc="Val (Cls)",
        dynamic_ncols=True,
        leave=False,
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images, labels=None)
        logits = outputs["out"]
        loss = criterion(logits, labels)

        batch_loss = loss.item()
        loss_meter.update(batch_loss)

        batch_acc = topk_accuracy(logits, labels, topk=(1,))[0].item()
        acc_meter.update(batch_acc)

        progress.set_postfix(
            loss=f"{batch_loss:.4f} ({loss_meter.avg:.4f})",
            acc=f"{batch_acc:.2f}% ({acc_meter.avg:.2f}%)",
        )

    return acc_meter.avg, loss_meter.avg


gc.collect()
torch.cuda.empty_cache()


@torch.no_grad()
def valid_epoch_ver(model, pair_dataloader, device, fpr_targets=None):
    model.eval()
    scores = []
    match_labels = []

    progress = tqdm(
        pair_dataloader,
        desc="Val (Veri)",
        dynamic_ncols=True,
        leave=False,
    )

    for images1, images2, labels in progress:
        images1 = images1.to(device, non_blocking=True)
        images2 = images2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images = torch.cat([images1, images2], dim=0)
        outputs = model(images, return_feats=True)
        feats = F.normalize(outputs["feats"], dim=1)

        feats1, feats2 = feats.chunk(2, dim=0)
        similarity = F.cosine_similarity(feats1, feats2, dim=1)

        scores.append(similarity.cpu().numpy())
        match_labels.append(labels.cpu().numpy())

    scores = np.concatenate(scores)
    match_labels = np.concatenate(match_labels)

    if fpr_targets is None:
        fpr_targets = [1e-4, 5e-4, 1e-3, 5e-3, 5e-2]

    metric_dict = verification_metrics(match_labels, scores, fpr_targets)
    print("Verification Metrics:", metric_dict)
    return metric_dict


wandb.login(key="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")

run = wandb.init(
    name = "Fine Tuning with ArcFace",
    project = "HW2P2",
    config = config
)


checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

def save_model(model, optimizer, scheduler, metrics, epoch, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


def load_model(model, optimizer=None, scheduler=None, path="./checkpoint.pth", device=None):
    print(f"Loading checkpoint from {path}...")
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    return model, optimizer, scheduler, epoch, metrics

def load_for_finetune(model, path, device):
    """
    Special load function for transitioning to ArcFace.
    Loads ONLY the model weights, uses strict=False, and ignores old optimizers.
    """
    print(f"Loading pre-trained backbone from: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # strict=False is the magic keyword here!
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print("Backbone weights loaded successfully! ArcFace head is randomly initialized.")
    return model

# optimizer = optim.SGD([
#     {'params': [p for n, p in model.named_parameters() if 'arc' not in n], 'lr': config['lr'] * 0.1},
#     {'params': model.arc.parameters(), 'lr': config['lr']}
# ], momentum=0.9, weight_decay=config['weight_decay'], nesterov=True)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

model = load_for_finetune(model, "/home/avid/Intro_Deep_Learning/hw2_checkpoint/best_ret.pth", DEVICE)
# model, optimizer, scheduler, epoch, metrics = load_model(model, optimizer, scheduler, path="/home/avid/Intro_Deep_Learning/hw2_finetuning_checkpoint/best_ret.pth", device=DEVICE)
start_epoch = 0
best_cls_acc = 0.0
best_ret_acc = 0.0
best_eer = float('inf')
eval_cls = True

for epoch in range(start_epoch, config["epochs"]):
    print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

    if epoch == 0:
        print("[Phase 1] Freezing backbone. Training ArcFace head only...")
        for name, param in model.named_parameters():
            if 'arc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'], nesterov=True)
        scheduler = get_scheduler(optimizer, warmup_epochs=config['warmup_epochs'], total_epochs=config['epochs'])
                
    elif epoch == 2:
        print("[Phase 2] Unfreezing backbone. Dropping learning rate for full fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.SGD([
            {'params': [p for n, p in model.named_parameters() if 'arc' not in n], 'lr': config['lr'] * 0.1},
            {'params': model.arc.parameters(), 'lr': config['lr']}
        ], momentum=0.9, weight_decay=config['weight_decay'], nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    curr_lr = optimizer.param_groups[0]["lr"]
    train_cls_acc, train_loss = train_epoch(
        model, cls_train_loader, optimizer, scheduler, scaler, DEVICE, criterion
    )

    metrics = {
        "train_cls_acc": train_cls_acc,
        "train_loss": train_loss,
        "lr": curr_lr,
    }

    if eval_cls:
        valid_cls_acc, valid_loss = valid_epoch_cls(model, cls_val_loader, DEVICE, criterion)
        metrics.update({
            "valid_cls_acc": valid_cls_acc,
            "valid_loss": valid_loss,
        })

    valid_ret_metrics = valid_epoch_ver(model, ver_val_loader, DEVICE)
    valid_ret_acc = valid_ret_metrics["ACC"]
    metrics.update({"valid_ret_acc": valid_ret_acc})

    checkpoint_path = os.path.join(config["checkpoint_dir"], "last.pth")
    save_model(model, optimizer, scheduler, metrics, epoch, checkpoint_path)

    if eval_cls and valid_cls_acc >= best_cls_acc:
        best_cls_acc = valid_cls_acc
        best_cls_path = os.path.join(config["checkpoint_dir"], "best_cls.pth")
        save_model(model, optimizer, scheduler, metrics, epoch, best_cls_path)
        if "wandb" in globals():
            wandb.save(best_cls_path)

    if valid_ret_acc >= best_ret_acc:
        best_ret_acc = valid_ret_acc
        best_ret_path = os.path.join(config["checkpoint_dir"], "best_ret.pth")
        save_model(model, optimizer, scheduler, metrics, epoch, best_ret_path)
        if "wandb" in globals():
            wandb.save(best_ret_path)

    if "run" in globals() and run is not None:
        run.log(metrics)

    if epoch == 10:
        break


@torch.no_grad()
def test_epoch_ver(model, pair_dataloader, device):
    model.eval()
    scores = []

    progress = tqdm(
        pair_dataloader,
        desc="Test (Veri)",
        dynamic_ncols=True,
        leave=False,
    )

    for images1, images2 in progress:
        images1 = images1.to(device, non_blocking=True)
        images2 = images2.to(device, non_blocking=True)

        images = torch.cat([images1, images2], dim=0)
        outputs = model(images, return_feats=True)
        feats = F.normalize(outputs["feats"], dim=1)

        feats1, feats2 = feats.chunk(2, dim=0)
        similarity = F.cosine_similarity(feats1, feats2, dim=1)

        scores.append(similarity.cpu().numpy())

    scores = np.concatenate(scores)
    return scores


scores = test_epoch_ver(model, ver_test_loader, DEVICE)

if "run" in globals() and run is not None:
    run.finish()


df_submission = pd.DataFrame({
    "ID": range(len(scores)),
    "Label": scores,
})
submission_path = "verification_early_submission.csv"
df_submission.to_csv(submission_path, index=False)


api.competition_submit(file_name="verification_early_submission.csv", message="Updated Architecture Submission", competition="11785-hw-2-p-2-face-verification-spring-2026")