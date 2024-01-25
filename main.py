# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:47:07 2024

@author: shangfr
"""

import sys
import cv2
import os
import json
import numpy as np

from PySide6.QtGui import QAction # using by exec 
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QPoint, Qt
from PySide6.QtCore import QParallelAnimationGroup
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QEvent
from PySide6.QtWidgets import QGraphicsDropShadowEffect
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu


from ui.custom_grips import CustomGrip
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow

from utils.capnums import Camera
from utils.rtsp_win import Window


from detect import YoloPredictor,letterbox

GLOBAL_STATE = False    # max min flag
GLOBAL_TITLE_BAR = True


class MainWindow(QMainWindow, Ui_MainWindow):
    # The main window sends an execution signal to the yolo instance
    main2yolo_begin_sgl = Signal()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        # Set window flag: hide window borders
        self.setWindowFlags(Qt.FramelessWindowHint)
        UIFunctions.uiDefinitions(self)
        # Show module shadows
        UIFunctions.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFunctions.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFunctions.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFunctions.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize(
            './models/' + x))   # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox = QTimer(self)
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        # self.yolo_predict = YOLO()
        # Create a Yolo instance
        self.yolo_predict = YoloPredictor()
        self.select_model = self.model_box.currentText()                   # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        # Create yolo thread
        self.yolo_thread = QThread()
        self.yolo_predict.yolo2main_pre_img.connect(
            lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_res_img.connect(
            lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(
            lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(
            lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)
        self.yolo_predict.yolo2main_class_num.connect(
            lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(
            lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(
            lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(
            lambda x: self.change_val(x, 'iou_slider'))      # iou scroll bar
        self.conf_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(
            lambda x: self.change_val(x, 'conf_slider'))    # conf scroll bar
        self.speed_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(
            lambda x: self.change_val(x, 'speed_slider'))  # speed scroll bar

        # Prompt window initialization
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # Select detection source
        self.src_file_button.clicked.connect(
            self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(self.chose_cam)  # chose_cam
        # self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_rtsp

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)   # pause/start
        self.stop_button.clicked.connect(self.stop)             # termination

        # Other function buttons
        self.save_res_button.toggled.connect(
            self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(
            self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(
            lambda: UIFunctions.toggleMenu(self, True))   # left navigation button
        self.settings_button.clicked.connect(
            lambda: UIFunctions.settingBox(self, True))   # top right settings button

        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status(
                'Please select a video source before starting detection...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)    # start button
                # It is forbidden to check and save after starting the detection
                self.save_txt_button.setEnabled(False)
                self.save_res_button.setEnabled(False)
                self.show_status('Detecting...')
                self.yolo_predict.continue_dtc = True   # Control whether Yolo is paused
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)    # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
            self.pre_video.clear()           # clear image display
            self.res_video.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(
            self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Load File：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    # Select camera source----  have one bug
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='Note', text='loading camera...', time=2000, auto=True).exec()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                # cap = cv2.VideoCapture(cam)
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置分辨率
                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()
            y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.yolo_predict.source = action.text()

                self.show_status('Loading camera：{}'.format(action.text()))

        except Exception as e:
            self.show_status('%s' % e)

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(
            lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.yolo_predict.save_res = True

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.yolo_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()         # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)    # start key recovery
        # Ability to use the save button
        self.save_res_button.setEnabled(True)
        # Ability to use the save button
        self.save_txt_button.setEnabled(True)
        self.pre_video.clear()           # clear image display
        self.res_video.clear()           # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            # The box value changes, changing the slider
            self.iou_slider.setValue(int(x*100))
        elif flag == 'iou_slider':
            # The slider value changes, changing the box
            self.iou_spinbox.setValue(x/100)
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.iou_thres = x/100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms

    # change model
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (
            0 if self.save_res_button.checkState() == Qt.Unchecked else 2)
        config['save_txt'] = (
            0 if self.save_txt_button.checkState() == Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


class UIFunctions(MainWindow):
    # Expand left menu
    def toggleMenu(self, enable):
        if enable:
            standard = 68
            maxExtend = 180
            width = self.LeftMenuBg.width()

            if width == 68:
                widthExtended = maxExtend
            else:
                widthExtended = standard

            # animation
            self.animation = QPropertyAnimation(
                self.LeftMenuBg, b"minimumWidth")
            self.animation.setDuration(500)  # ms
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QEasingCurve.InOutQuint)
            self.animation.start()

    # Expand the settings menu on the right
    def settingBox(self, enable):
        if enable:
            # GET WIDTH
            widthRightBox = self.prm_page.width()           # right set column width
            widthLeftBox = self.LeftMenuBg.width()  # left column length
            maxExtend = 220
            standard = 0

            # SET MAX WIDTH
            if widthRightBox == 0:
                widthExtended = maxExtend
            else:
                widthExtended = standard

            # ANIMATION LEFT BOX
            self.left_box = QPropertyAnimation(
                self.LeftMenuBg, b"minimumWidth")
            self.left_box.setDuration(500)
            self.left_box.setStartValue(widthLeftBox)
            self.left_box.setEndValue(68)
            self.left_box.setEasingCurve(QEasingCurve.InOutQuart)

            # ANIMATION RIGHT BOX
            self.right_box = QPropertyAnimation(self.prm_page, b"minimumWidth")
            self.right_box.setDuration(500)
            self.right_box.setStartValue(widthRightBox)
            self.right_box.setEndValue(widthExtended)
            self.right_box.setEasingCurve(QEasingCurve.InOutQuart)

            # GROUP ANIMATION
            self.group = QParallelAnimationGroup()
            self.group.addAnimation(self.left_box)
            self.group.addAnimation(self.right_box)
            self.group.start()

    # maximize window
    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status == False:
            GLOBAL_STATE = True
            self.showMaximized()    # maximize
            # self.Main_QW.setContentsMargins(0, 0, 0, 0)   # appMargins = Main_QW
            self.max_sf.setToolTip("Restore")
            self.frame_size_grip.hide()         # Hide the interface size adjustment button
            # Up, down, left, and right adjustments are prohibited
            self.left_grip.hide()
            self.right_grip.hide()
            self.top_grip.hide()
            self.bottom_grip.hide()
        else:
            GLOBAL_STATE = False
            self.showNormal()       # minimize
            self.resize(self.width()+1, self.height()+1)
            # self.Main_QW.setContentsMargins(10, 10, 10, 10)
            self.max_sf.setToolTip("Maximize")
            self.frame_size_grip.show()
            self.left_grip.show()
            self.right_grip.show()
            self.top_grip.show()
            self.bottom_grip.show()

    # window control
    def uiDefinitions(self):
        # Double-click the title bar to maximize
        def dobleClickMaximizeRestore(event):
            if event.type() == QEvent.MouseButtonDblClick:
                QTimer.singleShot(
                    250, lambda: UIFunctions.maximize_restore(self))
        self.top.mouseDoubleClickEvent = dobleClickMaximizeRestore

        # MOVE WINDOW / MAXIMIZE / RESTORE
        def moveWindow(event):
            if GLOBAL_STATE:                        # IF MAXIMIZED CHANGE TO NORMAL
                UIFunctions.maximize_restore(self)
            if event.buttons() == Qt.LeftButton:    # MOVE
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
        self.top.mouseMoveEvent = moveWindow
        # CUSTOM GRIPS
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)

        # MINIMIZE
        self.min_sf.clicked.connect(lambda: self.showMinimized())
        # MAXIMIZE/RESTORE
        self.max_sf.clicked.connect(lambda: UIFunctions.maximize_restore(self))
        # CLOSE APPLICATION
        self.close_button.clicked.connect(self.close)

    # Control the stretching of the four sides of the window
    def resize_grips(self):
        self.left_grip.setGeometry(0, 10, 10, self.height())
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
        self.top_grip.setGeometry(0, 0, self.width(), 10)
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # Show module to add shadow
    def shadow_style(self, widget, Color):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setOffset(8, 8)  # offset
        shadow.setBlurRadius(38)    # shadow radius
        shadow.setColor(Color)    # shadow color
        widget.setGraphicsEffect(shadow)


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, self.cap

    def __len__(self):
        return 0


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
