import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.initGlobalStep()
        # init the epoch counter
        self.initCurEpoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.globalStepTensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def initCurEpoch(self):
        with tf.variable_scope('cur_epoch'):
            self.curEpochTensor = tf.Variable(0, trainable=False, name='curEpoch')
            self.incrementCurEpochTensor = tf.assign(self.curEpochTensor, self.curEpochTensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def initGlobalStep(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.globalStepTensor = tf.Variable(0, trainable=False, name='globalStep')

    def initSaver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def buildModel(self):
        raise NotImplementedError