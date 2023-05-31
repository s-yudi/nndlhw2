# nndlhw2 
ResNet-18
训练和测试步骤同时进行，主要修改参数如下：
python train.py 
--device cuda 
--lr 0.1 
--argumentation cutout
--batch-size 128
最终得到一个csv文件，里面存放了每个Epoch的train_loss, test_loss以及test_accuracy

Faster R-CNN
训练步骤：
运行train.py。
测试步骤：
在frcnn.py中修改相关参数，运行predict，提示下输入图片地址，结果保存在相应的输出文件夹中。

FCOS
训练步骤：
运行 train_voc.py
测试步骤：
运行 eval_voc.py
检测步骤：
运行 detect.py
