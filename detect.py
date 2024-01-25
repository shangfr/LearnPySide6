# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:47:07 2024

@author: shangfr
"""
import time
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_imshow
from ultralytics.cfg import get_cfg
from ultralytics import YOLO

from PySide6.QtCore import Signal, QObject

class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)   # raw image signal
    yolo2main_res_img = Signal(np.ndarray)   # test result signal
    # Detecting/pausing/stopping/testing complete/error reporting signal
    yolo2main_status_msg = Signal(str)
    yolo2main_fps = Signal(str)              # fps
    # Detected target results (number of each category)
    yolo2main_labels = Signal(dict)
    yolo2main_progress = Signal(int)         # Completeness
    yolo2main_class_num = Signal(int)        # Number of categories detected
    yolo2main_target_num = Signal(int)       # Targets detected

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(
            SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(
            Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI args
        self.used_model_name = None      # The detection model name to use
        self.new_model_name = None       # Models that change in real time
        self.source = ''                 # input source
        self.stop_dtc = False            # Termination detection
        self.continue_dtc = True         # pause
        self.save_res = False            # Save test results
        self.save_txt = False            # save label(txt) file
        self.iou_thres = 0.45            # iou
        self.conf_thres = 0.25           # conf
        self.speed_thres = 10            # delay, ms
        self.labels_dict = {}            # return a dictionary of results
        self.progress_value = 0          # progress bar

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        # self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.yolomodel = None
        self.callbacks = defaultdict(
            list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    # main for detect
    @smart_inference_mode()
    def run(self):
        try:
            if self.args.verbose:
                LOGGER.info('')

            # set model
            self.yolo2main_status_msg.emit('Loding Model...')

            if not self.yolomodel or self.used_model_name != self.new_model_name:
                self.yolomodel = YOLO(self.new_model_name)
                self.used_model_name = self.new_model_name
                # method defaults
                custom = {'conf': self.conf_thres,
                          'iou': self.iou_thres, 'save': False}
                # highest priority args on the right
                args = {**self.yolomodel.overrides,
                        **custom, 'mode': 'predict'}
                self.predictor = (self.yolomodel._smart_load('predictor'))(overrides=args,
                                                                           _callbacks=self.callbacks)
                self.predictor.setup_model(
                    model=self.yolomodel.model, verbose=False)
            self.predictor.setup_source(
                self.source if self.source is not None else self.predictor.args.source)

            if self.predictor.args.save or self.predictor.args.save_txt:
                (self.predictor.save_dir /
                 'labels' if self.predictor.args.save_txt else self.predictor.save_dir).mkdir(parents=True, exist_ok=True)
            # Warmup model
            if not self.predictor.done_warmup:
                self.predictor.model.warmup(imgsz=(
                    1 if self.predictor.model.pt or self.predictor.model.triton else self.predictor.dataset.bs, 3, *self.predictor.imgsz))
                self.predictor.done_warmup = True

            self.seen, self.windows, self.batch, self.dt = 0, [
            ], None, (ops.Profile(), ops.Profile(), ops.Profile())

            self.dataset = self.predictor.dataset
            count = 0                       # run location frame
            start_time = time.time()        # used to calculate the frame rate
            batch = iter(self.dataset)
            while True:
                # Termination detection
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        # release final video writer
                        self.vid_writer[-1].release()
                    self.yolo2main_status_msg.emit('Detection terminated!')
                    break

                # Change the model midway
                if self.used_model_name != self.new_model_name:
                    self.yolo2main_status_msg.emit('Change Model...')
                    self.yolomodel = YOLO(self.new_model_name)
                    self.used_model_name = self.new_model_name
                    # method defaults
                    custom = {'conf': self.conf_thres,
                              'iou': self.iou_thres, 'save': False}
                    args = {**self.yolomodel.overrides, **custom,
                            'mode': 'predict'}  # highest priority args on the right
                    self.predictor = (self.yolomodel._smart_load('predictor'))(overrides=args,
                                                                               _callbacks=self.callbacks)
                    self.predictor.setup_model(
                        model=self.yolomodel.model, verbose=False)
                    self.predictor.setup_source(
                        self.source if self.source is not None else self.predictor.args.source)
                    self.dataset = self.predictor.dataset
                    batch = iter(self.dataset)

                # pause switch
                if self.continue_dtc:
                    # time.sleep(0.001)
                    self.yolo2main_status_msg.emit('Detecting...')
                    batch = next(self.dataset)  # next data

                    self.predictor.batch = batch

                    # path, im, im0s, vid_cap, s = batch
                    path, im0s, vid_cap, s = batch

                    count += 1              # frame count +1

                    # all_count = 1
                    # results =  self.yolomodel(self.source,stream= True)

                    if vid_cap:
                        all_count = vid_cap.get(
                            cv2.CAP_PROP_FRAME_COUNT)   # total frames
                    elif self.predictor.source_type.webcam:
                        all_count = -1
                    else:
                        all_count = 1
                    # progress bar(0~1000)
                    self.progress_value = int(count/all_count*1000)
                    if count % 5 == 0 and count >= 5:                     # Calculate the frame rate every 5 frames
                        self.yolo2main_fps.emit(
                            str(int(5/(time.time()-start_time))))
                        start_time = time.time()
                    # # preprocess
                    with self.dt[0]:
                        im = self.predictor.preprocess(im0s)
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # inference
                    with self.dt[1]:
                        visualize = increment_path(self.predictor.save_dir / Path(self.predictor.batch[0][0]).stem,
                                                   mkdir=True) if self.predictor.args.visualize and (
                            not self.predictor.source_type.tensor) else False
                        preds = self.predictor.model(
                            im, augment=self.predictor.args.augment, visualize=visualize)
                    # postprocess
                    with self.dt[2]:
                        self.results = self.predictor.postprocess(
                            preds, im, im0s)

                    # visualize, save, write results
                    n = len(im0s)     # To be improved: support multiple img
                    for i in range(n):
                        self.results[i].speed = {
                            'preprocess': self.dt[0].dt * 1E3 / n,
                            'inference': self.dt[1].dt * 1E3 / n,
                            'postprocess': self.dt[2].dt * 1E3 / n}
                        p, im0 = path[i], None if self.predictor.source_type.tensor else im0s[i].copy(
                        )
                        p = Path(p)     # the source dir
                        # labels   /// original :s +=
                        label_str = self.write_results(
                            i, self.results, (p, im, im0))
                    #
                        # labels and nums dict
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}
                        if 'no detections' in label_str:
                            pass
                        else:
                            for ii in label_str.split(',')[:-1]:
                                nums, label_name = ii.split('~')

                                self.labels_dict[label_name] = int(nums)
                                target_nums += int(nums)
                                class_nums += 1

                        # save img or video result
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(
                                self.predictor.save_dir / p.name))

                        # Send test results
                        # self.yolo2main_res_img.emit(im0) # after detection
                        self.yolo2main_res_img.emit(
                            self.plotted_img)  # after detection

                        self.yolo2main_pre_img.emit(im0s if isinstance(
                            im0s, np.ndarray) else im0s[0])   # Before testing
                        # webcam need to change the def write_results
                        self.yolo2main_labels.emit(self.labels_dict)
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres/1000)   # delay , ms

                    self.yolo2main_progress.emit(
                        self.progress_value)   # progress bar

                # Detection completed
                if count == all_count:
                    if isinstance(self.predictor.vid_writer[-1], cv2.VideoWriter):
                        # release final video writer
                        self.predictor.vid_writer[-1].release()
                    self.yolo2main_status_msg.emit('Detection completed')
                    break

        except Exception as e:
            print(e)
            self.yolo2main_status_msg.emit('%s' % e)

    def get_annotator(self, img):
        return Annotator(img, line_width=None, example=str(self.predictor.model.names))

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.predictor.source_type.webcam or self.predictor.source_type.from_img or self.predictor.source_type.tensor:  # batch_size >= 1
            # log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + (
            '' if self.dataset.mode == 'image' else f'_{frame}')
        result = results[idx]

        det = results[idx].boxes  # TODO: make boxes inherit from tensors

        if len(det) == 0:
            return f'{log_string}(no detections), '  # if no, send this~~

        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            # {'s' * (n > 1)}, "   # don't add 's'
            log_string += f"{n}~{self.predictor.model.names[int(c)]},"

        self.annotator = self.get_annotator(im0)

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.predictor.args.line_width,
                'boxes': det,  # self.predictor.args.boxes,
                'conf': self.predictor.args.show_conf,
                'labels': self.predictor.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt',
                            save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops',
                             file_name=self.data_path.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))

        return log_string


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

