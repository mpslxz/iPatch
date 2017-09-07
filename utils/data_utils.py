import os
import tarfile
import cPickle
from urllib import urlretrieve


class DataFactory(object):
    def __init__(self, dataset, aspect_ratio='original', resize_ratio=1):
        self.aspect_ratio = aspect_ratio
        self.resize_ratio = resize_ratio
        if dataset == 'CIFAR100':
            self.file_name = 'cifar-100-python'
            self.image_size = (32, 32)
        else:
            self.file_name = None

    def download(self, source, file_name):
        urlretrieve(source, file_name)

    def download_dataset(self):
        if self.file_name == 'cifar-100-python':
            if not os.path.exists(os.path.join('.', self.file_name+'.tar.gz')):
                print('Downloading dataset..')
                self.download('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', self.file_name+'.tar.gz')
            if not os.path.isdir(self.file_name):
                tar = tarfile.open(self.file_name+'.tar.gz')
                tar.extractall()

    def get_images(self):
        if self.file_name == 'cifar-100-python':
            with open(os.path.join(self.file_name, 'train'), 'rb') as f:
                data_dict = cPickle.load(f)
                data =  data_dict['data']
                data =  data.reshape((-1, 3, self.image_size[0], self.image_size[1]))
                return data.swapaxes(1,2).swapaxes(2,3)
