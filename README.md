# 机器学习测试小合集

## 合成特征和离群值
源码：synthetic_features_and_outliers.py
根据加州房价数据，建立SGD模型。合成特征作为单一输入，预测房价中位数，截去离群值样本后的预测对比。
### 1.导入数据  
### 2.定义输入函数  
### 3.合成特征  
创建一个名为 rooms_per_person 的合成特征,即 total_rooms 与 population 的比例，并将其用作 train_model() 的 input_feature.探索街区人口密度与房屋价值中位数之间的关系。  

### 4.识别离群值  
通过创建预测值与目标值的散点图来可视化模型效果。 
理想情况下，这些值将位于一条完全相关的对角线上。  
重点关注偏离这条线的点。我们注意到这些点的数量相对较少。  
查看 rooms_per_person 中值的分布情况，将这些异常情况追溯到源数据。  
如果我们绘制 rooms_per_person 的直方图，则会发现我们的输入数据中有少量离群值。  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/Figure_1_old.png)
（未处理离群值前）

### 5.截取离群值  
创建的直方图显示，大多数值都小于 5。  
我们将 rooms_per_person 的值截取为 5，然后绘制直方图以再次检查结果。  
为了验证截取是否有效，我们再训练一次模型，并再次输出校准数据。  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/Figure_2_new.png)
（处理离群值后）






