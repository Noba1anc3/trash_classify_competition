# -*- coding: utf-8 -*-
import json
import os
import codecs
from collections import OrderedDict
from PIL import Image

from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
import log
logger = log.getLogger(__name__)

mmcv_path = os.path.join(os.path.dirname(__file__), 'mmcv-0.5.9-cp36-cp36m-linux_x86_64.whl')
mmdet_path = os.path.join(os.path.dirname(__file__), 'mmdet-2.0.0+unknown-cp36-cp36m-linux_x86_64.whl')
os.system(f'pip install {mmcv_path}; pip install {mmdet_path}')
from mmdet.apis import inference_detector, init_detector
from mmcv.runner import load_checkpoint
import numpy as np


class ObjectDetectionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # make sure these files exist
        self.model_name = model_name
        self.checkpoint = os.path.join(os.path.dirname(__file__), 'epoch_17.pth')
        self.config = os.path.join(os.path.dirname(__file__), 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
        self.label_map = parse_classify_rule(os.path.join(os.path.dirname(__file__), 'classify_rule.json'))

        self.model = init_detector(self.config, checkpoint=self.checkpoint, device='cpu')
        load_checkpoint(self.model, self.checkpoint, map_location='cpu')
        self.model.eval()

        self.classes = self.model.CLASSES
        self.input_image_key = 'images'
        self.score = 0.3

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                preprocessed_data[k] = image
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        tick = time.time()
        img = data[self.input_image_key]

        result = inference_detector(self.model, img)
        tock = time.time()
        print('-------------1', tock-tick)
        tick = time.time()
        bboxes = np.vstack(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)

        inds = np.where(bboxes[:, -1] > self.score)
        bboxes = bboxes[inds]
        labels = labels[inds]
        tock = time.time()
        print('-------------2', tock - tick)
        tick = time.time()
        result = OrderedDict()
        if len(bboxes) > 0:
            out_classes = labels
            out_scores = bboxes[:, 4]
            out_boxes = bboxes[:, 0:4]

            detection_class_names = []
            for class_id in out_classes:
                class_name = self.classes[int(class_id)]
                class_name = self.label_map[class_name] + '/' + class_name
                detection_class_names.append(class_name)

            out_boxes_list = []
            for box in out_boxes:
                out_boxes_list.append([round(float(v), 1) for v in box])

            result['detection_classes'] = detection_class_names
            result['detection_scores'] = [round(float(v), 4) for v in out_scores]
            result['detection_boxes'] = out_boxes_list
        else:
            result['detection_classes'] = []
            result['detection_scores'] = []
            result['detection_boxes'] = []
        tock = time.time()
        print('-------------3', tock - tick)
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data


def parse_classify_rule(json_path=''):
    with codecs.open(json_path, 'r', 'utf-8') as f:
        rule = json.load(f)
    label_map = {}
    for super_label, labels in rule.items():
        for label in labels:
            label_map[label] = super_label
    return label_map
