from base.baseTrainer import BaseTrainer
from tqdm import tqdm
import numpy as np


class ExampleTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config,logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config,logger)

    def trainEpoch(self):
        # tqdm显示循环进度条
        loop = tqdm(range(self.data.imageNum))
        losses = []
        accs = []

        # loop中一次循环就是一个batchSize
        for _ in loop:
            loss, acc = self.trainStep()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("loss = {0}, acc = {1}".format(loss,acc))
        cur_it = self.model.globalStepTensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def trainStep(self):
        batch_x, batch_y = self.data.next_batch()
        print(batch_y)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: self.config.mode=="train"}
        _, loss, acc = self.sess.run([self.model.trainStep, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc