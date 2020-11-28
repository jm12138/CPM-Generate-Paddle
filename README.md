# CPM-Generate-Paddle
本Repo将模型转换为PaddlePaddle版本，原Repo https://github.com/TsinghuaAI/CPM-Generate

原项目首页：https://cpm.baai.ac.cn/

原项目介绍文章：https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw

## 项目说明
参考[CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)、[CPM-LM-TF2](https://github.com/qhduan/CPM-LM-TF2)、[gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch)等项目开发

感谢上述项目的开源代码和模型

## 快速试用
可以使用百度AIStudio平台简单快速的体验试用这个模型，[项目链接](https://aistudio.baidu.com/aistudio/projectdetail/1279908)

## 使用说明
* 克隆本项目代码
```shell
$ git clone https://github.com/jm12138/CPM-Generate-Paddle
$ cd CPM-Generate-Paddle
```

* 准备模型文件
  * 如果你没有原版模型可以直接下载：[下载链接](http://bj.bcebos.com/v1/ai-studio-online/ffb6bed9360147f4bf513c5970ad5a5e742cabeb298e4f51b16a3e2d21dde837?responseContentDisposition=attachment%3B%20filename%3DCPM.tar.gz&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-11-28T08%3A39%3A10Z%2F-1%2F%2F59c43785123712fa76ea11ce46d9348ccfb2739ccbdb1d6fcaa09a87cf2ce17f)
  * 如果已经下载了原版模型可以自己进行转换：请参考[转换代码](https://github.com/jm12138/CPM-Generate-Paddle/blob/main/convert.py)
  * 将模型文件放置于./GPT2/pretrained_model

* 安装如下依赖
```
sentencepiece 
jieba 
regex 
paddlepaddle==2.0.0rc0
```

* 运行测试Demo
```shell
$ python demo.py
```

## 引用
> @article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Cao, Jiannan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
