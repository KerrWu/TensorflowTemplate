import tensorflow as tf


class BaseTester:
    def __init__(self, sess, model, data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data

        # restore test模型
        (self.model).saver.restore(self.sess, self.config.model_path)

    def test(self):
        self.testEpoch()

    def testEpoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the test step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def testStep(self):
        """
        implement the logic of the test step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def testFunc(self, *args, **kwargs):
        """
        不同任务用到的不同test函数
        """
        raise NotImplementedError