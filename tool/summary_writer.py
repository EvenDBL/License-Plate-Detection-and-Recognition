from tensorboardX import SummaryWriter
import logging
import os
class Writer():
    def __init__(self, path='summary'):
        self.summary_path = path
        self.writer = SummaryWriter(self.summary_path)
        if not os.path.exists(self.summary_path):
            os.mkdir(self.summary_path)

    def add_scalar(self, scope_name, y, x):
        self.writer.add_scalar(scope_name, y, x)

    def add_scalars(self, scope_name, y, x):
        self.writer.add_scalars(scope_name, y, x)

    def add_image(self, scope_name, image):
            self.writer.add_image(scope_name, image, dataformats='HWC') #

    def add_figure(self, scope_name, figure):
            self.writer.add_figure(tag=scope_name, figure=figure) #

    def add_images(self, scope_name, batch):
        self.writer.add_images(scope_name, batch)