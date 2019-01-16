# 模版内容：



## 1. 配置信息读取器--Configuration

parser在./utils/parse.py中

配置信息默认存储在./config/config.json文件中，格式如下

```
{

 "exp_name": "xxx",

  "num_epochs": 20,

  "learning_rate": 0.001,

  "batch_size": 5,

  "max_to_keep":5,

  "image_height": 300,

  "image_width": 300,

  "class_list": ["bcc", "lue", "ccc"],

   "mode": "train",
   
   "model_path":"../.."
}
```



## 2. 数据装载器--Data Loader

包括数据的载入和预处理，直接与trainer对接

利用Keras的DataGenerator和flow_from_directory方法实现

数据放在./data文件夹下，不同类别按文件夹存放，类别名即文件夹名



## 3. 模型--Model

**BaseModel**

定义在./base中

一个抽象类，实现了基础功能，所有模型类都要继承

其中包含了这些通用数据成员和方法成员：

数据成员：

1. curEpoch: keep track 当前的epoch数
2. globalStep: keep track 当前step数

方法成员：

1. save: 将checkpoint存入disk
2. load: 将disk中的checkpoint导入
3. initSaver: 抽象类，需要在定义模型类的时候被override，其中需要定义self.saver成员
4. buildModel: 抽象类，需要在定义模型类的时候被override，作用是定义模型结构



**my model**

定义在./models中

实现自己的模型时需注意：

1. 继承baseModel
2. 重写InitSaver和BuildModel
3. 在initialize模型时调用重写的InitSaver和BuildModel、



## 4. 记录器--Logger

定义在./utils中

与tensorboard的summary对接，在trainer中创建一个dict，将需要记录的所有variables写入，再将其传入logger.summarize()

## 

## 5.训练器--Trainer

**BaseTrainer**

定义在./base中

抽象类，将所有训练过程wrap在一起



**my trainer**

定义在./trainers中

实现自己的trainer时需注意：

1. 继承baseTrainer
2. 重写trainStep和trainEpoch两个方法



## 6. 测试器--Tester

**BaseTester**

定义在./base中

抽象类，将所有训练过程wrap在一起



**my tester**

定义在./testers中

实现自己的tester时需注意：

1. 继承baseTester
2. 重写testEpoch, testStep, testFunc 3个方法



## 7. 主函数--Main

导入所有模块，其中model和trainer模块为重写后的版本

其流程包括：

1. parse配置信息
2. 创建tensorflow session
3. 创建所需用到的实例，包括：Model, DataGenerator, Logger,并将parse得到的参数分别传入这些实例
4. 创建实例Trainer，将之前创建的实例Model, DataGenerator, Logger传入Trainer实例中
5. 调用Trainer.train()开始训练















​	

