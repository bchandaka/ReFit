import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob
from torch.utils.data import default_collate

from lib.core.config import update_cfg
from lib import get_model
from lib.datasets.custom_dataset import custom_dataset
from lib.renderer.renderer_img import Renderer as Renderer_img
from lib.yolo import Yolov7

# from pytorch3d.transforms import matrix_to_axis_angle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='configs/config.yaml')
    parser.add_argument("--ckpt",  type=str, default='data/pretrain/refit_all/checkpoint_best.pth.tar')
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--viz_results", action='store_true')
    args = parser.parse_args()


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo = Yolov7(device=DEVICE, weights='data/pretrain/yolov7-e6e.pt', imgsz=1281)

    dir = 'C:\projects\imu_human\ReFit\data\examples\long_bedroom_sample'
    camera_pose_dir = 'C:\projects\imu_human\ReFit\data\examples\long_bedroom_sample\camera_pose'
    img_paths = [os.path.join(dir, imgfile) for imgfile in sorted(os.listdir(dir)) if (imgfile.endswith('.png') or imgfile.endswith('.jpg')) ]

    ### --- Detection ---
    all_boxes = []
    for img_path in img_paths:
        img = cv2.imread(img_path)[:,:,::-1].copy()
        with torch.no_grad():
            boxes = yolo(img, conf=0.50, iou=0.45)
            boxes = boxes.cpu().numpy()
            all_boxes.append(boxes)
    individual_cam_data = [np.load(os.path.join(camera_pose_dir, pose_file)) for pose_file in sorted(os.listdir(camera_pose_dir))] 
    camera_data = {}
    keys = ['cam_R', 'cam_t', 'img_focal', 'img_center', 'trans']
    for key in keys:

        camera_data[key] = np.array([cam_data[key] for cam_data in individual_cam_data])
    # camera_data = np.load('/home/bchan/projects/imu_human/ReFit/data/examples/multiview_examples/multiview_examples.npz')

    # multiview examples
    db = custom_dataset(img_paths, all_boxes, camera_data, is_train=False, use_augmentation=False, 
                     normalization=True, cropped=False, crop_size=256)

    # refit
    DEVICE = args.device
    cfg = update_cfg(args.cfg)
    cfg.DEVICE = DEVICE
    cfg.MODEL.VERSION = 'mv'
    
    model = get_model(cfg).to(DEVICE)
    state_dict = torch.load(args.ckpt, map_location=cfg.DEVICE)
    _ = model.load_state_dict(state_dict['model'], strict=False)
    _ = model.eval()

    # Rendering
    renderer_img = Renderer_img(model.smpl.faces, color=(0.40,  0.60,  0.9, 1.0))

    # Load 4 views
    items = []
    for i in range(len(img_paths)):
        item = db[i]
        items.append(item)
    batch = default_collate(items)
    for k,v in batch.items():
        if type(v)==torch.Tensor:
            batch[k] = v.float().to(DEVICE)

    # multiview refitex
    with torch.no_grad():
        out, iter_preds = model(batch, 10)
        smpl_out = model.smpl.query(out)

    for k in range(len(img_paths)):
        # imgfile = batch['imgname'][k]
        img_full = cv2.imread(img_paths[k])[:,:,::-1]

        vert = smpl_out.vertices[k]
        trans = out['trans_full'][k]
        vert_full = (vert + trans).cpu()

        focal = batch['img_focal'][k]
        center = batch['img_center'][k]
        img_render = renderer_img(vert_full, [0,0,0], img_full, focal, center)

        os.makedirs(f'mv_refit', exist_ok=True)
        cv2.imwrite(f'mv_refit/img_{i}_{k}.png', img_full[:,:,::-1])
        cv2.imwrite(f'mv_refit/mesh_{i}_{k}.png', img_render[:,:,::-1])














