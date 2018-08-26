# 机器学习测试小合集

## contents  

[1.synthetic_features_and_outliers](https://github.com/hoshinotsuki/tensorflow-gpu-test#%E4%B8%80%E5%90%88%E6%88%90%E7%89%B9%E5%BE%81%E5%92%8C%E7%A6%BB%E7%BE%A4%E5%80%BC)  
[2.validation](https://github.com/hoshinotsuki/tensorflow-gpu-test#2validation-another-partition)  
[3.feature-sets](https://github.com/hoshinotsuki/tensorflow-gpu-test#3feature-sets)  
[4.feature-crosses](https://github.com/hoshinotsuki/tensorflow-gpu-test#4feature-crosses)  

## 1.synthetic features and outliers
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

## 2.Validation: Another Partition  
we're working with the California housing data set,to try and predict median_house_value at the city block level from 1990 census data.  

code:[validation.py](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/validation.py "查看源码")  

### workflow  
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
  
    
      
      
    
    
# 3.Feature Sets

**Learning Objective:** Create a minimal set of features that performs just as well as a more complex feature set

So far, we've thrown all of our features into the model. Models with fewer features use fewer resources and are easier to maintain. Let's see if we can build a model on a minimal set of housing features that will perform equally as well as one that uses all the features in the data set.

## Setup

As before, let's load and prepare the California housing data.

## Task 1: Develop a Good Feature Set

**What's the best performance you can get with just 2 or 3 features?**

A **correlation matrix** shows **pairwise correlations**, both for each feature compared to the target and for each feature compared to other features.

Here, correlation is defined as the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)**皮尔逊相关系数**.  You don't have to understand the mathematical details for this exercise.
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Feature_Set/df_corr().PNG)   
（correlation matrix）

Correlation values have the following meanings:

  * `-1.0`: perfect negative correlation
  * `0.0`: no correlation
  * `1.0`: perfect positive correlation

Features that have **strong positive or negative correlations** with the **target** will add information to our model. We can use the correlation matrix to find such **strongly correlated features**.

We'd also like to have features that aren't so strongly correlated with each other, so that they add **independent information**.

Use this information to try removing features.  You can also try developing additional **synthetic features**, such as ratios of two raw features.

For convenience, we've included the training code from the previous exercise.

Spend 5 minutes searching for a good set of features and training parameters. Then check the solution to see what we chose. Don't forget that different features may require different learning parameters.  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Feature_Set/unbinning.png)  
(unbinning)

## Task 2: Make Better Use of Latitude

Plotting `latitude` vs. `median_house_value` shows that there really **isn't a linear relationship** there.

Instead, there are a couple of peaks, which roughly correspond to Los Angeles and San Francisco.  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Feature_Set/Isn't%20a%20linear%20relationship.png)   
(Isn't a linear relationship)

**Try creating some synthetic features that do a better job with latitude.**

For example, you could have a feature that maps `latitude` to a value of `|latitude - 38|`, and call this `distance_from_san_francisco`.

Or you could break the space into 10 different buckets.  `latitude_32_to_33`, `latitude_33_to_34`, etc., each showing a value of `1.0` if `latitude` is within that bucket range and a value of `0.0` otherwise.

Use the correlation matrix to help guide development, and then add them to your model if you find something that looks good.

What's the best validation performance you can get?

### Solution

Aside from `latitude`, we'll also keep `median_income`, to compare with the previous results.

We decided to bucketize the latitude. This is fairly straightforward in Pandas using `Series.apply`.
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/Feature_Set/binning.png)  
（binning）  
  
  

# 4.Feature Crosses

**Learning Objectives:**
  * Improve a linear regression model with the addition of additional synthetic features (this is a continuation of the previous exercise)
  * Use an input function to convert pandas `DataFrame` objects to `Tensors` and invoke the input function in `fit()` and `predict()` operations
  * Use the FTRL optimization algorithm for model training
  * Create new synthetic features through one-hot encoding, binning, and feature crosses

## Setup

First, as we've done in previous exercises, let's define the input and create the data-loading code.

## FTRL Optimization Algorithm

High dimensional linear models benefit from using a variant of gradient-based optimization called FTRL. 
This algorithm has the benefit of scaling the learning rate differently for different coefficients, 
which can be useful if some features **rarely** take **non-zero values** 
(it also is well suited to support L1 regularization).  
高维度线性模型可受益于使用一种基于梯度的优化方法，叫做 FTRL。
该算法的优势是针对不同系数以不同方式调整学习速率，
如果某些特征很少采用非零值，该算法可能比较实用（也非常适合支持 L1 正则化）。 

We can apply FTRL using the [FtrlOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer).

## One-Hot Encoding for Discrete Features

Discrete (i.e. strings, enumerations, integers) features are usually converted into families of binary features before training a logistic regression model.

For example, suppose we created a synthetic feature that can take any of the values `0`, `1` or `2`, and that we have a few training points:

| # | feature_value |
|---|---------------|
| 0 |             2 |
| 1 |             0 |
| 2 |             1 |

For each possible categorical value, we make a new **binary** feature of **real values** that can take one of just two possible values: 1.0 if the example has that value, and 0.0 if not. In the example above, the categorical feature would be converted into three features, and the training points now look like:

| # | feature_value_0 | feature_value_1 | feature_value_2 |
|---|-----------------|-----------------|-----------------|
| 0 |             0.0 |             0.0 |             1.0 |
| 1 |             1.0 |             0.0 |             0.0 |
| 2 |             0.0 |             1.0 |             0.0 |

## Bucketized (Binned) Features

Bucketization is also known as binning.

We can bucketize `population` into the following 3 buckets (for instance):
- `bucket_0` (`< 5000`): corresponding to less populated blocks
- `bucket_1` (`5000 - 25000`): corresponding to mid populated blocks
- `bucket_2` (`> 25000`): corresponding to highly populated blocks

Given the preceding bucket definitions, the following `population` vector:

    [[10001], [42004], [2500], [18000]]

becomes the following bucketized feature vector:

    [[1], [2], [0], [1]]

The feature values are now the `bucket indices`. Note that these indices are considered to be `discrete features`. Typically, these will be further converted in `one-hot representations` as above, but this is done transparently.

To define feature columns for bucketized features, instead of using `numeric_column`, we can use [`bucketized_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column), which takes a `numeric column` as input and transforms it to a `bucketized feature` using the bucket boundaries specified in the `boundardies` argument. The following code defines bucketized feature columns for `households` and `longitude`; the `get_quantile_based_boundaries` function calculates boundaries based on quantiles, so that each bucket contains an `equal number of elements`.  

## Task 1: Train the Model on Bucketized Feature Columns
**Bucketize all the real valued features in our example, train the model and see if the results improve.**

In the preceding code block, two real valued columns (namely `households` and `longitude`) have been transformed into `bucketized feature columns`. Your task is to bucketize the rest of the columns, then run the code to train the model. There are various heuristics to find the range of the buckets. This exercise uses a `quantile-based` technique, which chooses the bucket boundaries in such a way that each bucket has the same number of examples.

### Solution

You may be wondering how to determine how many buckets to use. That is of course data-dependent. Here, we just selected arbitrary values so as to obtain a not-too-large model.  

## Feature Crosses

Crossing two (or more) features is a clever way to learn `non-linear relations` using a `linear model`. In our problem, if we just use the feature `latitude` for learning, the model might learn that city blocks at a particular latitude (or within a particular range of latitudes since we have bucketized it) are more likely to be expensive than others. Similarly for the feature `longitude`. However, if we cross `longitude` by `latitude`, the crossed feature represents a well defined city block. If the model learns that certain city blocks (within range of latitudes and longitudes) are more likely to be more expensive than others, it is a stronger signal than two features considered individually.

Currently, the feature columns API only supports `discrete features` for crosses. To cross two continuous values, like `latitude` or `longitude`, we can `bucketize them`.

If we cross the `latitude` and `longitude` features (supposing, for example, that `longitude` was bucketized into `2` buckets, while `latitude` has `3` buckets), we actually get six crossed binary features. Each of these features will get its own separate weight when we train the model.

## Task 2: Train the Model Using Feature Crosses

**Add a feature cross of `longitude` and `latitude` to your model, train it, and determine whether the results improve.**

Refer to the TensorFlow API docs for [`crossed_column()`](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column) to build the feature column for your cross. Use a `hash_bucket_size` of `1000`.


## Optional Challenge: Try Out More Synthetic Features

So far, we've tried simple bucketized columns and feature crosses, but there are many more combinations that could potentially improve the results. For example, you could cross multiple columns. What happens if you vary the number of buckets? What other synthetic features can you think of? Do they improve the model?
