# 细粒度语义注入机制基于的指称分割

## 使用方法

### 安装

系统环境与软件版本

- OS: Ubuntu 18.04 LTS
- CUDA: 12.0
- PyTorch 1.11.0
- Python 3.7.0

安装需要的依赖Package：`pip install -r requirements.txt`

### 数据集

RefCOCO & RefCOCO+ & RefCOCOG & ReferIt 数据集。
生成工具见https://github.com/fengguang94/CEFNet，详细指令如下：

python build_batches.py -d unc -t train
python build_batches.py -d unc -t val
python build_batches.py -d unc -t testA
python build_batches.py -d unc -t testB
python build_batches.py -d unc+ -t train
python build_batches.py -d unc+ -t val
python build_batches.py -d unc+ -t testA
python build_batches.py -d unc+ -t testB
python build_batches.py -d referit -t trainval
python build_batches.py -d referit -t test
python build_batches.py -d Gref -t train
python build_batches.py -d Gref -t val

### 模型训练以及推理

1. 在data/config中指定数据的路径，格式按照参考代码进行更改
2. 在train中指定所需要的GPU参数（一块或者多块24g GeForce RTX3090s）
3. 使用python train.py进行模型训练
4. 使用python eval.py进行模型评估，在程序相关代码提示处指定模型ckpt路径
5. 使用python eval_miou.py来展示模型的Mean IoU得分。


## Citation
Please cite the following paper if you use this repository in your research.
```
@ARTICLE{10149830,
  author={Yang, Jiaxing and Zhang, Lihe and Lu, Huchuan},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Referring Image Segmentation With Fine-Grained Semantic Funneling Infusion}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TNNLS.2023.3281372}}
 ```


