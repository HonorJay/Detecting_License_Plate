
import torch
import sys
import os
import traceback
import json

sys.path.append('/work/Detecting_License_Plate/yolov5')

from utils.general import non_max_suppression, scale_coords
# from ai_core.object_detection.yolov5_custom.od.data.datasets import letterbox
# from typing import List
# from dynaconf import settings
from models.experimental import attempt_load
import cv2
import pandas as pd
import numpy as np
from pororo import Pororo
import logging

class Detection:
    """Handles the object detection tasks."""

    def __init__(self, weights_path='.pt',size=(640,640),device='cpu',iou_thres=None,conf_thres=None):
        self.device=device
        self.char_model, self.names = self.load_model(weights_path)
        self.size=(640,640)
        self.iou_thres=iou_thres
        self.conf_thres=conf_thres

    def detect(self, frame):
        """
        1. Detects license plates in a frame.
        2. Detects characters in a bbox of license plates.
        """
        results, resized_img = self.char_detection_yolo(frame)

        return results, resized_img
    
    def preprocess_image(self, original_image):

        resized_img = self.ResizeImg(original_image,size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()
        image = image / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img
    
    def char_detection_yolo(self, image, classes=None, \
                            agnostic_nms=True, max_det=1000):

        img,resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        
        detections = non_max_suppression(pred, conf_thres=self.conf_thres,
                                            iou_thres=self.iou_thres,
                                            classes=classes,
                                            agnostic=agnostic_nms,
                                            multi_label=True,
                                            labels=(),
                                            max_det=max_det)
        results=[]
        for i, det in enumerate(detections):
            # det[:, :4]=scale_coords(resized_img.shape,det[:, :4],image.shape).round()
            det=det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    # xc,yc,w_,h_=(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])
                    result=[self.names[int(cls)], str(conf*100)[:4]+"%", (xyxy[0],xyxy[1],xyxy[2],xyxy[3])]
                    results.append(result)
        # print(results)
        return results, resized_img
        
    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        # print(h1, w1, _)
        h, w = size
        if w1 < h1 * (w / h):
            # print(w1/h1)
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        
    def load_model(self,path, train = False):
        # print(self.device)
        model = attempt_load(path, map_location=self.device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if train:
            model.train()
        else:
            model.eval()
        return model, names


class Arguments:
    def __init__(self, video_path="test_video.mp4"):
        self.lp_weights = '/work/Detecting_License_Plate/Vietnamese/object.pt'
        self.ch_weights = '/work/Detecting_License_Plate/Vietnamese/char.pt'
        self.source = video_path
        self.lp_imgsz = [1280]
        self.ch_imgsz = [128]
        self.conf_thres = 0.1
        self.iou_thres = 0.5
        self.max_det = 1000
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main(video_path):
    ocr = Pororo(task='ocr', lang='ko')
    logging.warning("call pororo framework")

    opt = Arguments(video_path)
    lp_model=Detection(size=opt.lp_imgsz,weights_path=opt.lp_weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
    logging.warning("load model")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = {}
    logging.warning("total frame : {}".format(num_frames))
    try:
        for frame_id in range(num_frames):
            logging.warning("now frame : {}".format(frame_id))
            ret, frame = cap.read()
            if not ret:
                break
            lp_results, resized_img = lp_model.detect(frame.copy()) 
            logging.warning("found {} license plate".format(len(lp_results)))
            frame_results = []
            for lp_result in lp_results:
                if lp_result[0] in ['square license plate', 'rectangle license plate']:
                    bbox = lp_result[-1]
                    lp_result_dict = {"class": lp_result[0], "confidence": lp_result[1], "bbox": lp_result[2]}
                    lp_image = resized_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if lp_image.size == 0:
                        print("\nEmpty bounding box for frame: ", frame_id, f"bbox: {int(bbox[0])}:{int(bbox[2])}, {int(bbox[1])}:{int(bbox[3])}")
                        continue
                    # save the temporary image for pororo OCR
                    tmp = './tmp.jpg'
                    cv2.imwrite(tmp, lp_image)
                    logging.warning("save temporary image file")
                    if os.path.isfile(tmp):
                        recognized_text = ocr('./tmp.jpg')
                        os.remove(tmp)
                    else:
                        logging.warning("counldn't get any image file")
                        continue

                    frame_results.append({"license_plate": lp_result_dict, "recognized_text": recognized_text})

            results[frame_id] = frame_results  # Store results for this frame
    
    except Exception as e:
        print(f"Error processing frame {frame_id}: {e}")
        print(traceback.print_exc())  
    
    finally:
        file_path = "/work/result.json"
        with open(file_path, 'w') as f:
           json_object = json.dump(results, f)
        
    cap.release()
    cv2.destroyAllWindows()
    return json_object, results


    
