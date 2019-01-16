# 导入通用模块
import tensorflow as tf
from dataLoader.DataLoader import DataGenerator
from utils.parse import process_config
from utils.makeDirs import makeDirs
from utils.logger import Logger
from utils.getArgs import getArgs

# 导入需重写的模块
from models.exampleModel import ExampleModel
from trainers.exampleTrainer import ExampleTrainer
from testers.exampleTester import ExampleTester

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = getArgs()
        print("args got")
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    if config.mode == "train":

        # create the experiments dirs
        makeDirs([config.summary_dir, config.checkpoint_dir])
        # create tensorflow session
        sess = tf.Session()
        # create your data generator
        data = DataGenerator(config)
        # create an instance of the model
        model = ExampleModel(config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = ExampleTrainer(sess, model, data, config, logger)
        # load model if exists
        model.load(sess)
        # train model
        trainer.train()


    if config.mode == "test":
        # create the experiments dirs
        makeDirs([config.summary_dir, config.checkpoint_dir])
        # create tensorflow session
        sess = tf.Session()
        # create your data generator
        data = DataGenerator(config)
        # create an instance of the model
        model = ExampleModel(config)
        # create trainer and pass all the previous components to it
        tester = ExampleTester(sess, model, data, config)
        # train model
        tester.test()

    else:
        print("invalid mode")
        exit(0)


if __name__ == '__main__':
    main()