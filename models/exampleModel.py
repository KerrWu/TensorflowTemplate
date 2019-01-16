from base.baseModel import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.numClass = int(len(self.config.class_list))
        self.buildModel()
        self.initSaver()

    def buildModel(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, self.config.image_height, self.config.image_width, 3])
        self.y = tf.placeholder(tf.int32, shape=[None])

        #定义网络结构
        d0 = tf.layers.flatten(self.x)
        d1 = tf.layers.dense(d0, 10, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, self.numClass, name="dense2")

        #softmax_cross_entropy_with_logits要求logits和labels的shape一致，故如果标签不为one-hot形式，则转换为one-hot
        #本实现使用的是keras的generator，会自动输出one-hot形式的标签，故可以省略
        if d2.get_shape() != (self.y).get_shape():
            self.y = tf.one_hot(self.y, self.numClass, 1, 0)

        #如果是训练模式，定义损失函数及训练op，并计算准确率
        if self.config.mode == "train":
            with tf.name_scope("loss"):
                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.trainStep = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.globalStepTensor)
                correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #如果是测试模式,输出 网络最后一层输出，softmax输出，交叉熵损失，对每个样本的预测，准确率 共5项，由用户自己选择使用哪些
        if self.config.mode == "test":
            with tf.name_scope("test"):
                self.modelOutput = d2
                self.softmaxOutput = tf.nn.softmax(logits=d2)
                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
                self.prediction = tf.argmax(d2, 1)
                correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            print("invalid mode! ")
            exit(0)


    def initSaver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

