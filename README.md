# 猫狗分类器

基于残差网络Resnet34实现的猫狗分类器，包含了使用flask搭建的网站源代码，按照GPL v3.0开源。

可以访问我的网站测试模型识别准确率，对于大部分正常的猫狗图片都能实现准确的分类。

[猫狗分类器](http://47.115.209.13:5000/)

⚠注意**学术诚信**，切勿将此代码抄袭作为自己的课设作业提交，否则**后果自负**。

## 模型部署Demo

![](https://raw.githubusercontent.com/Dianashiba/piclib/main/Dog_result.png)

![](img/Cat_result.png)

## 环境信息

* Python版本： 3.10.11
* Pytorch版本：2.1.1+cu121
* 英伟达驱动版本：546.01
* CUDA版本：12.3
* 开发工具： PyCharm + Vscode

## 项目结构

```txt
|-- CNN_Cats&Dogs_classification
	|-- app
	|-- dataset
        |-- train
            |-- cat.0.jpg
            |-- cat.1.jpg
            |-- ······
        |-- test
            |-- 1.jpg
            |-- 2.jpg
            |-- ······
    |-- result
        |-- result.csv
    |-- model
    |-- dataset.py
    |-- evaluate.py
    |-- model.py
    |-- train.py
        
```

## app

基于flask实现的模型部署，可部署到云服务器上进行测试

```bash
nohup flask run -h 0.0.0.0 -p 80  > log.txt 2>&1 &
```

## model

存放部分已经训练好的模型

## dataset.py

数据集预处理

## evaluate.py

模型效果评估

## model.py

模型搭建

## train.py

训练模块

## 作者

22 级 [Dianashiba](https://dianashiba.github.io/)

email：avad1anash1ba@gmail.com