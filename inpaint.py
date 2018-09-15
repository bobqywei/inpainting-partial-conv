"""
PyQt App that leverages completed model for image inpainting
"""

import sys
import os
import random
import torch
import argparse

from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms

from partial_conv_net import PartialConvUNet
from places2_train import unnormalize, MEAN, STDDEV

def exceeds_bounds(y):
    if y >= 250:
        return True
    else:
        return False

class Drawer(QWidget):
    newPoint = pyqtSignal(QPoint)
    def __init__(self, image_path, parent=None):
        QWidget.__init__(self, parent)
        self.path = QPainterPath()
        self.image_path = image_path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(QRect(0, 0, 256, 256), QPixmap(self.image_path))
        painter.setPen(QPen(Qt.black, 12))
        painter.drawPath(self.path)

    def mousePressEvent(self, event):
        if exceeds_bounds(event.pos().y()):
            return
        
        self.path.moveTo(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        if exceeds_bounds(event.pos().y()):
            return
        
        self.path.lineTo(event.pos())
        self.newPoint.emit(event.pos())
        self.update()

    def sizeHint(self):
        return QSize(256, 256)

    def resetPath(self):
        self.path = QPainterPath()
        self.update()

class InpaintApp(QWidget):

    def __init__(self, image_num):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.title = 'Inpaint Application'
        self.width = 276
        self.height = 350
        self.cwd = os.getcwd()

        image_num = str(image_num).zfill(8)
        image_path = self.cwd + "/val_256/Places365_val_{}.jpg".format(image_num)

        self.save_path = self.cwd + "/test.jpg"
        self.open_and_save_img(image_path, self.save_path)
        self.drawer = Drawer(self.save_path, self)

        self.setWindowTitle(self.title)
        self.setGeometry(200, 200, self.width, self.height)

        self.layout().addWidget(self.drawer)
        self.layout().addWidget(QPushButton("Inpaint!", clicked=self.inpaint))
        
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STDDEV)])
        self.mask_transform = transforms.ToTensor()
        self.device = torch.device("cpu")

        model_dict = torch.load(self.cwd + "/model_e1_i56358.pth", map_location="cpu")
        model = PartialConvUNet()
        model.load_state_dict(model_dict["model"])
        model = model.to(self.device)

        self.model = model
        self.model.eval()
        
        self.show()

    def open_and_save_img(self, path, dest):
        img = Image.open(path)
        img.save(dest)

    def inpaint(self):
        mask = QImage(256, 256, QImage.Format_RGB32)
        mask.fill(qRgb(255, 255, 255))

        painter = QPainter()
        painter.begin(mask)
        painter.setPen(QPen(Qt.black, 12))
        painter.drawPath(self.drawer.path)
        painter.end()

        mask.save("mask.png", "png")

        # open image and normalize before forward pass
        mask = Image.open(self.cwd + "/mask.png")
        mask = self.mask_transform(mask.convert("RGB"))
        gt_img = Image.open(self.save_path)
        gt_img = self.img_transform(gt_img.convert("RGB"))
        img = gt_img * mask

        # adds dimension of 1 (batch) to image
        img.unsqueeze_(0)
        gt_img.unsqueeze_(0)
        mask.unsqueeze_(0)

        # forward pass
        with torch.no_grad():
            output = self.model(img.to(self.device), mask.to(self.device))

        # unnormalize the image and output
        output = mask * img + (1 - mask) * output
        grid = make_grid(unnormalize(output))
        save_image(grid, "test.jpg")

        self.drawer.resetPath()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=int, default=1)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = InpaintApp(args.img)
    sys.exit(app.exec_())