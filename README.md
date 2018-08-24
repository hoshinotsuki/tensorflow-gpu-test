# 机器学习测试小合集

## 一、合成特征和离群值
根据加州房价数据，建立SGD模型。合成特征作为单一输入，预测房价中位数，截去离群值样本后的预测对比。  
源码：[synthetic_features_and_outliers.py](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/synthetic_features_and_outliers.py "查看源码")  

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
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/synthetic_features_and_outliers/Figure_1_old.png)
（未处理离群值前）

### 5.截取离群值  
创建的直方图显示，大多数值都小于 5。  
我们将 rooms_per_person 的值截取为 5，然后绘制直方图以再次检查结果。  
为了验证截取是否有效，我们再训练一次模型，并再次输出校准数据。  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/synthetic_features_and_outliers/Figure_2_new.png)
（处理离群值后）
</br></br></br>

## 二、Validation: Another Partition
we're working with the California housing data set,to try and predict median_house_value at the city block level from 1990 census data. 

![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Validation/%E6%8D%95%E8%8E%B7.PNG)</br>

# Validation

**Learning Objectives:**
  * Use multiple features, instead of a single feature, to further improve the effectiveness of a model
  * Debug issues in model input data
  * Use a test data set to check if a model is overfitting the validation data  
  
As in the prior exercises, we're working with the California housing data set, to try and predict `median_house_value` at the city block level from 1990 census data.  
 

## Setup

First off, let's load up and prepare our data. This time, we're going to work with multiple features, so we'll modularize the logic for preprocessing the features a bit:  

* 加上随机化处理，否则训练集和验证集的分布不一致  

For the training set, we'll choose the first 12000 examples, out of the total of 17000.  
For the validation set, we'll choose the last 5000 examples, out of the total of 17000.
 
## Task 1: Examine the Data 
Let's check our data against some baseline expectations:  
* For some values, like `median_house_value`,we can check to see if these values fall within reasonable ranges (keeping in mind this was 1990 data — not today!).  
* For other values, like `latitude` and `longitude`,we can do a quick check to see if these `line up with位于合理的范围内`expected values from a quick Google search.
If you look closely, you may see some oddities:  
* `median_income` is on a scale from about 3 to 15.It's not at all clear what this scale refers to—looks like maybe some log scale?It's not documented anywhere; all we can assume is that higher values correspond to higher income.  
* The maximum `median_house_value` is 500,001. This looks like an `artificial cap of some kind人为上限`.  
* Our `rooms_per_person` feature is generally on a `sane scale正常范围`, with a 75th percentile value of about 2.But there are some very large values, like 18 or 55, which may show some amount of corruption in the data.  
We'll use these features as given for now.But hopefully these kinds of examples can help to build a little intuition about how to check data that comes to you from an unknown source.

## Task 2: Plot Latitude/Longitude vs. Median House Value
Let's take a close look at two features in particular:  
**`latitude`** and **`longitude`**. These are geographical coordinates of the city block in question.
This might make a nice visualization -let's plot `latitude` and `longitude`, and use color to show the `median_house_value`.

* Wait a second...this should have given us a nice map of the state of California,with red showing up in expensive areas like the San Francisco and Los Angeles.红色表示高房价，像洛杉矶  
* The training set sort of does, compared to a real map, but the validation set clearly doesn't.  
训练集（12/17）比验证集（5/17）更像一个真正的地图，因为没有随机化    
* Looking at the tables of summary stats above, it's easy to wonder how anyone would do a useful data check.  
如何做一个有效的数据检查  
* The key thing to notice is that for any given feature or column, 对于每个特征和特征列the distribution of values between the train and validation splits should be roughly equal.训练集和验证集的划分应该一致  
* The fact that this is not the case is a real worry, 真正担心的是事实不是这样and shows that we likely have a fault in the way that our train and validation split was created.说明区分训练集和验证集时有错误  
  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Validation/%E6%9C%AA%E9%9A%8F%E6%9C%BA%E5%8C%96.PNG)
（unrandomized training sample and valuating sample）  

## Task 3: Return to the Data Importing and Pre-Processing Code, and See if You Spot Any Bugs  
* There's an important lesson here.Debugging in ML is often data debugging rather than code debugging.  
重要：ML的调试是数据调试，不是代码调试  
* If the data is wrong, even the most advanced ML code can't save things.    
* Take a look at how the data is randomized when it's read in.数据读入的时候是否随机化  
* If we don't randomize the data properly before creating training and validation splits,then we may be in trouble if the data is given to us in some sorted order, which appears to be the case here.  

![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Validation/Figure_1.png)  
（randomized training sample and valuating sample）

## Task 4: Train and Evaluate a Model  
Next, we'll train a linear regressor using all the features in the data set, and see how well we do.  
Let's define the same input function we've used previously for loading the data into a TensorFlow model.  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Validation/Figure_3.png)  
（Train and Evaluate）

## Task 5: Evaluate on Test Data

**load in the test data set and evaluate your model on it.**  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Validation/Figure_2.png)  
（randomized test sample）  
We've done a lot of iteration on our validation data. Let's make sure we haven't overfit to the pecularities of that particular sample.  
How does your test performance compare to the validation performance?   
What does this say about the generalization performance of your model?    
**my ourput:Final RMSE (on test data): 161.66**



