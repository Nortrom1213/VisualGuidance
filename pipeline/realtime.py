#!/usr/bin/env python
"""
Realtime Overlay for Game Navigation Hints with Late Fusion

Usage:
  python realtime_infer.py \
    --detector_model_path model1_stp_detector.pth \
    --selector_model_path model2_mstp_selector.pth \
    [--use_retrieval --retrieval_bank_file feature_bank.pkl --alpha 0.6] \
    [--frame_interval 200]
"""

import sys, time, argparse, pickle, os
from PIL import Image
import numpy as np
import cv2
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from PyQt5 import QtWidgets, QtCore, QtGui
import mss

# -------------------------
# Shared Adapter Module
# -------------------------
class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_dim=256):
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.relu = nn.ReLU()
        self.up   = nn.Linear(bottleneck_dim, in_features)
    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))

# -------------------------
# CombinedPredictor for STP detector
# -------------------------
class CombinedPredictor(nn.Module):
    def __init__(self, orig_predictor, in_features, adapter_dim=256):
        super().__init__()
        self.predictor = orig_predictor
        self.adapter   = Adapter(in_features, adapter_dim)
    def forward(self, x):
        x = self.adapter(x)
        return self.predictor(x)

# -------------------------
# STP Detector Definition
# -------------------------
def get_stp_detector(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    new_p = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_f, num_classes)
    model.roi_heads.box_predictor = CombinedPredictor(new_p, in_f, adapter_dim=256)
    return model

def remap_and_load_detector(det, ckpt_path, device):
    raw     = torch.load(ckpt_path, map_location=device)
    remapped = {}
    for k, v in raw.items():
        if k.startswith("roi_heads.box_predictor.cls_score"):
            nk = k.replace("roi_heads.box_predictor.", "roi_heads.box_predictor.predictor.")
        elif k.startswith("roi_heads.box_predictor.bbox_pred"):
            nk = k.replace("roi_heads.box_predictor.", "roi_heads.box_predictor.predictor.")
        else:
            nk = k
        remapped[nk] = v
    missing, unexpected = det.load_state_dict(remapped, strict=False)
    print(f"→ Detector loaded from {ckpt_path}\n   Missing keys: {missing}\n   Unexpected: {unexpected}")

det_transform = T.Compose([T.ToTensor()])

def run_stp_detector(pil_img, model, device, thr):
    t = det_transform(pil_img).to(device)
    model.eval()
    with torch.no_grad():
        out = model([t])[0]
    boxes  = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep   = np.where(scores >= thr)[0]
    return boxes[keep]

# -------------------------
# MSTP Selector Definition
# -------------------------
class MSTPSelectorNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(pretrained=True)
        self.cand = nn.Sequential(*list(r.children())[:-1])
        self.glob = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(64,512), nn.ReLU()
        )
        self.adapter = Adapter(1024,256)
        self.cls     = nn.Sequential(nn.Linear(1024,256), nn.ReLU(), nn.Linear(256,1))

    def forward(self, cands, glob):
        f1 = self.cand(cands).view(cands.size(0),-1)
        f2 = self.glob(glob.unsqueeze(0)).expand(f1.size(0),-1)
        fused = torch.cat([f1,f2],1)
        fused = self.adapter(fused)
        return self.cls(fused).squeeze(1)

crop_t = T.Compose([T.Resize((224,224)), T.ToTensor()])
glob_t = T.Compose([T.Resize((64,64)),   T.ToTensor()])

# -------------------------
# Retrieval Helpers
# -------------------------
def load_bank(path):
    if os.path.exists(path):
        with open(path,"rb") as f: return pickle.load(f)
    return []

def cosine(a,b):
    a,b = np.array(a),np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

def retr_score(feat, bank):
    sims = [cosine(feat,e["feature"]) for e in bank]
    return max(sims) if sims else 0.0

def get_feature_extractor(device):
    r = torchvision.models.resnet18(pretrained=True)
    seq = nn.Sequential(*list(r.children())[:-1]).to(device)
    seq.eval()
    return seq

def extract_feat(pil_crop, ext, device):
    t = T.Compose([T.Resize((224,224)),T.ToTensor()])(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        f = ext(t).view(-1).cpu().numpy()
    return f.tolist()

# -------------------------
# Overlay Widget
# -------------------------
class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint|
                            QtCore.Qt.WindowStaysOnTopHint|
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.boxes=[]

    def update_boxes(self, boxes):
        self.boxes = boxes
        super().update()

    def paintEvent(self, e):
        painter = QtGui.QPainter(self)
        for x1,y1,x2,y2,label,score in self.boxes:
            # 红色粗线标 MSTP，绿色细线标 STP
            color = QtGui.QColor(255,0,0) if label=="MSTP" else QtGui.QColor(0,255,0)
            width = 3 if label=="MSTP" else 2
            pen   = QtGui.QPen(color, width)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            # 在框上方绘制标签＋两位小数的分数
            text = f"{label} {score:.2f}"
            painter.drawText(x1, y1-5, text)

# -------------------------
# Worker Thread
# -------------------------
class Worker(QtCore.QThread):
    sig = QtCore.pyqtSignal(list)
    def __init__(self, det, sel, dev, thr, interval, bank, alpha):
        super().__init__()
        self.det, self.sel, self.dev = det, sel, dev
        self.thr, self.intv = thr, interval
        self.bank, self.alpha = bank, alpha
        self.ext = get_feature_extractor(dev)
        self.running = True

    def run(self):
        with mss.mss() as sct:
            mon = sct.monitors[1]
            while self.running:
                t0 = time.time()
                shot = sct.grab(mon)
                img  = np.array(shot)[:,:,:3]
                pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = run_stp_detector(pil, self.det, self.dev, self.thr)
                out=[]
                if len(boxes)>0:
                    crops = [crop_t(pil.crop(b)) for b in boxes]
                    ct    = torch.stack(crops).to(self.dev)
                    gt    = glob_t(pil).to(self.dev)
                    with torch.no_grad():
                        ms = self.sel(ct,gt).cpu().numpy()
                    feats = [
                        extract_feat(
                            Image.fromarray((c.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)),
                            self.ext, self.dev
                        ) for c in crops
                    ]
                    rs = np.array([retr_score(f,self.bank) for f in feats])
                    fs = self.alpha*ms + (1-self.alpha)*rs
                    idx = int(np.argmax(fs))
                    for i,b in enumerate(boxes):
                        lbl = "MSTP" if i==idx else "STP"
                        score = float(fs[i])
                        out.append((int(b[0]),int(b[1]),int(b[2]),int(b[3]),lbl,score))
                self.sig.emit(out)
                dt = (time.time()-t0)*1000
                self.msleep(max(0,int(self.intv-dt)))

    def stop(self):
        self.running=False
        self.quit()
        self.wait()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_model_path", required=True)
    parser.add_argument("--selector_model_path", required=True)
    parser.add_argument("--use_retrieval",       action="store_true")
    parser.add_argument("--retrieval_bank_file", default="feature_bank.pkl")
    parser.add_argument("--alpha",               type=float, default=0.5)
    parser.add_argument("--score_threshold",     type=float, default=0.5)
    parser.add_argument("--frame_interval",      type=float,   default=1)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    over = Overlay()
    over.setGeometry(app.primaryScreen().geometry())
    over.show()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    # Load detector
    det = get_stp_detector().to(dev)
    remap_and_load_detector(det, args.detector_model_path, dev)

    # Load selector
    sel = MSTPSelectorNet().to(dev)
    ck = torch.load(args.selector_model_path, map_location=dev)
    sel.load_state_dict(ck, strict=False)
    print("→ Selector loaded.")

    bank = load_bank(args.retrieval_bank_file) if args.use_retrieval else []

    worker = Worker(det, sel, dev,
                    thr=args.score_threshold,
                    interval=args.frame_interval,
                    bank=bank,
                    alpha=args.alpha)
    worker.sig.connect(over.update_boxes)   # <-- connect to update_boxes
    worker.start()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# python realtime_infer.py --detector_model_path bmw_model1_stp_detector.pth --selector_model_path model2_mstp_selector.pth  --use_retrieval --retrieval_bank_file feature_bank.pkl --alpha 0.6 --frame_interval 0.2
