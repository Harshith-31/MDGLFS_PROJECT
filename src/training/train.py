import os
import sys
import json
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.dataset import MultiDomainForensicDataset
from src.models.router import ViTRouter, DOMAINS
from src.models.detectors.face_detector import FaceDetector
from src.models.detectors.med_detector import MedDetector
from src.models.detectors.sat_detector import SatDetector
from src.models.detectors.gen_detector import GeneralDetector
from src.training.losses import CompoundForensicLoss

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def compute_metrics(preds_list, labels_list, masks_pred_list=None, masks_gt_list=None):
    """
    Computes genuine classification & localization metrics without mock values:
    Accuracy, Precision, Recall, F1, IoU, Dice.
    """
    preds = np.array(preds_list)
    labels = np.array(labels_list)
    
    bin_preds = (preds >= 0.5).astype(int)
    bin_labels = (labels >= 0.5).astype(int)
    
    tp = int(np.sum((bin_preds == 1) & (bin_labels == 1)))
    fp = int(np.sum((bin_preds == 1) & (bin_labels == 0)))
    fn = int(np.sum((bin_preds == 0) & (bin_labels == 1)))
    tn = int(np.sum((bin_preds == 0) & (bin_labels == 0)))
    
    acc = (tp + tn) / max(1, len(labels))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * (prec * rec) / max(1e-8, (prec + rec))
    
    mean_iou = 0.0
    mean_dice = 0.0
    if masks_pred_list is not None and len(masks_pred_list) > 0:
        ious = []
        dices = []
        for p_m, g_m in zip(masks_pred_list, masks_gt_list):
            p_bin = (p_m > 0.35).astype(float)
            g_bin = (g_m > 0.35).astype(float)
            inter = np.sum(p_bin * g_bin)
            union = np.sum(np.clip(p_bin + g_bin, 0, 1))
            total = np.sum(p_bin) + np.sum(g_bin)
            if union == 0:
                ious.append(1.0 if np.sum(g_bin) == 0 else 0.0)
                dices.append(1.0 if total == 0 else 0.0)
            else:
                ious.append(float(inter / union))
                dices.append(float((2.0 * inter) / (total + 1e-8)))
        mean_iou = float(np.mean(ious))
        mean_dice = float(np.mean(dices))
        
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "iou": round(float(mean_iou), 4),
        "dice": round(float(mean_dice), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }

def train_detector(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path, domain_name):
    print(f"\n==================================================")
    print(f" Training Expert Detector: {domain_name}")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {train_loader.batch_size}")
    print(f"==================================================")
    
    best_val_loss = float("inf")
    best_metrics = {}
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_seg_loss = 0.0
        train_box_loss = 0.0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            bboxes = batch["bbox"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            targets = {"mask": masks, "bbox": bboxes, "label": labels}
            loss_dict = criterion(preds, targets)
            loss = loss_dict["loss_total"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_cls_loss += loss_dict["loss_cls"].item()
            train_seg_loss += loss_dict["loss_seg"].item()
            train_box_loss += loss_dict["loss_box"].item()
            
        n_train = max(1, len(train_loader))
        train_loss /= n_train
        train_cls_loss /= n_train
        train_seg_loss /= n_train
        train_box_loss /= n_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        preds_list, labels_list = [], []
        masks_pred_list, masks_gt_list = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                bboxes = batch["bbox"].to(device)
                labels = batch["label"].to(device)
                
                preds = model(imgs)
                targets = {"mask": masks, "bbox": bboxes, "label": labels}
                loss_dict = criterion(preds, targets)
                val_loss += loss_dict["loss_total"].item()
                
                probs = preds["prob"].detach().cpu().numpy().flatten().tolist()
                lbls = labels.cpu().numpy().flatten().tolist()
                preds_list.extend(probs)
                labels_list.extend(lbls)
                
                p_masks = preds["mask"].squeeze(1).detach().cpu().numpy()
                g_masks = masks.squeeze(1).cpu().numpy()
                for i in range(len(p_masks)):
                    masks_pred_list.append(p_masks[i])
                    masks_gt_list.append(g_masks[i])
                    
        n_val = max(1, len(val_loader))
        val_loss /= n_val
        val_metrics = compute_metrics(preds_list, labels_list, masks_pred_list, masks_gt_list)
        val_metrics["val_loss"] = round(float(val_loss), 4)
        val_metrics["train_loss"] = round(float(train_loss), 4)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Loss (Train/Val): {train_loss:.4f} / {val_loss:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f}")
        
        if val_loss < best_val_loss or epoch == epochs:
            best_val_loss = val_loss
            best_metrics = val_metrics
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved checkpoint: {save_path} (Val Loss: {val_loss:.4f})")
            
    print(f"✓ {domain_name} training completed. Best F1: {best_metrics['f1']} | IoU: {best_metrics['iou']}")
    return best_metrics

def train_router(model, train_loader, val_loader, optimizer, device, epochs, save_path):
    print(f"\n==================================================")
    print(f" Training Vision Transformer Router (ViTRouter)")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {train_loader.batch_size}")
    print(f"==================================================")
    
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_metrics = {}
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["domain_idx"].to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            logits = out["logits"]
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["domain_idx"].to(device)
                out = model(imgs)
                logits = out["logits"]
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= max(1, len(val_loader))
        val_acc = correct / max(1, total)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Loss (Train/Val): {train_loss:.4f} / {val_loss:.4f} | "
              f"Routing Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss or epoch == epochs:
            best_val_loss = val_loss
            best_metrics = {
                "val_loss": round(float(val_loss), 4),
                "accuracy": round(float(val_acc), 4)
            }
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved checkpoint: {save_path} (Val Acc: {val_acc:.4f})")
            
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description="Train MD-DGFLS Multi-Domain Forensics Suite")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "face", "med", "sat", "gen", "router"])
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Initializing MD-DGFLS Training Pipeline on: {device}")
    
    train_manifest = "datasets/manifests/train_manifest.json"
    val_manifest = "datasets/manifests/validation_manifest.json"
    
    if not os.path.exists(train_manifest) or not os.path.exists(val_manifest):
        raise FileNotFoundError("Dataset manifests missing! Run src/datasets/build_dataset.py first.")
        
    os.makedirs("models", exist_ok=True)
    registry_path = "models/registry.json"
    registry = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
        except Exception:
            registry = {}
            
    criterion = CompoundForensicLoss()
    
    # 1. ViTRouter Training
    if args.domain in ["all", "router"]:
        ds_train = MultiDomainForensicDataset.from_manifest(train_manifest)
        ds_val = MultiDomainForensicDataset.from_manifest(val_manifest)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
        
        router_model = ViTRouter(img_size=256, num_domains=5).to(device)
        save_path = "models/router_best.pth"
        if os.path.exists(save_path):
            try:
                router_model.load_state_dict(torch.load(save_path, map_location=device), strict=False)
                print(f"Loaded existing router checkpoint from {save_path} for fine-tuning.")
            except Exception:
                pass
        optimizer = torch.optim.AdamW(router_model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        metrics = train_router(router_model, train_loader, val_loader, optimizer, device, epochs=args.epochs, save_path=save_path)
        registry["router"] = {
            "model_type": "ViTRouter",
            "checkpoint": save_path,
            "trained_at": datetime.datetime.now().isoformat(),
            "device": str(device),
            "parameters": sum(p.numel() for p in router_model.parameters()),
            "metrics": metrics
        }
        
    # Domain specific configs
    detector_configs = [
        {
            "key": "face",
            "name": "Face",
            "manifest_filter": "Face",
            "model_cls": FaceDetector,
            "save_path": "models/face_detector_best.pth"
        },
        {
            "key": "med",
            "name": "Medical Radiograph",
            "manifest_filter": "Medical Radiograph",
            "model_cls": MedDetector,
            "save_path": "models/xray_detector_best.pth"
        },
        {
            "key": "sat",
            "name": "Satellite Terrain",
            "manifest_filter": "Satellite Terrain",
            "model_cls": SatDetector,
            "save_path": "models/sat_detector_best.pth"
        },
        {
            "key": "gen",
            "name": "General Scene & Out-of-Domain",
            "manifest_filter": None, # Will use Regular Out-of-Domain + Natural Scene
            "model_cls": GeneralDetector,
            "save_path": "models/general_detector_best.pth"
        }
    ]
    
    for cfg in detector_configs:
        if args.domain != "all" and args.domain != cfg["key"]:
            continue
            
        print(f"\nLoading data for {cfg['name']}...")
        if cfg["manifest_filter"]:
            ds_train = MultiDomainForensicDataset.from_manifest(train_manifest, domain_filter=cfg["manifest_filter"])
            ds_val = MultiDomainForensicDataset.from_manifest(val_manifest, domain_filter=cfg["manifest_filter"])
        else:
            # For general detector, combine Natural Scene + Regular Out-of-Domain
            with open(train_manifest) as f:
                d_train = [d for d in json.load(f) if d["domain"] in ["Natural Scene", "Regular Out-of-Domain"]]
            with open(val_manifest) as f:
                d_val = [d for d in json.load(f) if d["domain"] in ["Natural Scene", "Regular Out-of-Domain"]]
            ds_train = MultiDomainForensicDataset(d_train)
            ds_val = MultiDomainForensicDataset(d_val)
            
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
        
        model = cfg["model_cls"]().to(device)
        if os.path.exists(cfg["save_path"]):
            try:
                model.load_state_dict(torch.load(cfg["save_path"], map_location=device), strict=False)
                print(f"Loaded existing checkpoint from {cfg['save_path']} for fine-tuning.")
            except Exception as e:
                print(f"Starting from base weights: {e}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        metrics = train_detector(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            save_path=cfg["save_path"],
            domain_name=cfg["name"]
        )
        
        registry[cfg["key"]] = {
            "domain_name": cfg["name"],
            "model_type": cfg["model_cls"].__name__,
            "checkpoint": cfg["save_path"],
            "trained_at": datetime.datetime.now().isoformat(),
            "device": str(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_samples": len(ds_train),
            "val_samples": len(ds_val),
            "metrics": metrics
        }
        
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\n[✓] Registry saved to {registry_path}")
    print("All training passes successfully finished!")

if __name__ == "__main__":
    main()
