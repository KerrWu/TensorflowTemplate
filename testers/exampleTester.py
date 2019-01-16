from base.baseTester import BaseTester
from tqdm import tqdm
import numpy as np


class ExampleTester(BaseTester):
    def __init__(self, sess, model, data, config):
        super(ExampleTester, self).__init__(sess, model, data, config)

    def testEpoch(self):
        # tqdm显示循环进度条
        loop = tqdm(range(self.data.imageNum))
        losses = []
        accs = []

        # loop中一次循环就是一个batchSize
        for _ in loop:
            correstNum = self.testStep()

            #使用tqdm时print失效，需要用tqdm.write()在终端打印输出
            loop.write(correstNum)

    def testStep(self):
        batch_x, batch_y = self.data.next_batch()
        print(batch_y)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_Testing: self.config.mode=="Test"}

        # 可选输出包括 1.网络最后一层输出，2.softmax输出，3.交叉熵损失，4.对每个样本的预测，5.准确率 共5项
        # self.model.modelOutput
        # self.model.softmaxOutput
        # self.model.cross_entropy
        # self.model.prediction
        # self.model.accuracy
        # 这里取softmax输出和prediction, 以softmax输出作为testFunc的例子
        softmaxOutput, prediction = self.sess.run([self.model.softmaxOutput, self.model.prediction],feed_dict=feed_dict)
        correctNum = self.testFunc(softmaxOutput, prediction)

        return correctNum

    def testFunc(self, softmaxOutput, prediction):
        #不同任务需用到不同的testFunc，这里以比较softmax输出和网络prediction是否一致为例
        softmaxResult = np.argmax(softmaxOutput, 1)
        correct = np.equal(softmaxResult, prediction)
        return np.sum(correct)