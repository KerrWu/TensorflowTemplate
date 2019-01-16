import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

        if config.mode == "train":
            self.DataGen = ImageDataGenerator(rotation_range=40,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              fill_mode='nearest',
                                              data_format='channels_last')
        elif config.mode == "test":
            self.DataGen = ImageDataGenerator( fill_mode='nearest',
                                              data_format='channels_last')
        else:
            print("mode must be train or test!")
            raise ValueError

        self.Generator = self.DataGen.flow_from_directory('../data',
                                                          target_size=(config.image_height, config.image_width),
                                                          color_mode='rgb',
                                                          classes=config.class_list, class_mode='categorical',
                                                          batch_size=config.batch_size, shuffle=True, seed=None,
                                                          save_to_dir=None,
                                                          save_prefix=config.exp_name,
                                                          save_format='jpeg',
                                                          follow_links=False)

        #返回batch总个数，用于tqdm输出
        self.imageNum = self.Generator.__len__()

    def next_batch(self):

        # keras的generator的输出label是one-hot形式的
        return next(self.Generator)