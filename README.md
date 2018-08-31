# Machine Learning Test Collection

## contents  

[1.synthetic_features_and_outliers](https://github.com/hoshinotsuki/tensorflow-gpu-test#%E4%B8%80%E5%90%88%E6%88%90%E7%89%B9%E5%BE%81%E5%92%8C%E7%A6%BB%E7%BE%A4%E5%80%BC)  
[2.validation](https://github.com/hoshinotsuki/tensorflow-gpu-test#2validation-another-partition)  
[3.feature sets](https://github.com/hoshinotsuki/tensorflow-gpu-test#3feature-sets)  
[4.feature crosses](https://github.com/hoshinotsuki/tensorflow-gpu-test#4feature-crosses)  
[5.logistic regression](https://github.com/hoshinotsuki/tensorflow-gpu-test#5logistic-regression)  
[6.sparsity and l1 regularization](https://github.com/hoshinotsuki/tensorflow-gpu-test#6sparsity-and-l1-regularization)  

## 1.synthetic features and outliers
根据加州房价数据，建立SGD模型。合成特征作为单一输入，预测房价中位数，截去离群值样本后的预测对比。  
源码：[synthetic_features_and_outliers.py](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/synthetic_features_and_outliers.py "查看源码")  

### 1.import datas 
### 2.define input function  
### 3.synthetic features
创建一个名为 rooms_per_person 的合成特征,即 total_rooms 与 population 的比例，并将其用作 train_model() 的 input_feature.探索街区人口密度与房屋价值中位数之间的关系。  
### 4.identify outliers
通过创建预测值与目标值的散点图来可视化模型效果。 
理想情况下，这些值将位于一条完全相关的对角线上。  
重点关注偏离这条线的点。我们注意到这些点的数量相对较少。  
查看 rooms_per_person 中值的分布情况，将这些异常情况追溯到源数据。  
如果我们绘制 rooms_per_person 的直方图，则会发现我们的输入数据中有少量离群值。  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/synthetic_features_and_outliers/Figure_1_old.png)
（未处理离群值前）
### 5.clip outliers  
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

So far, we've tried simple bucketized columns and feature crosses, but there are many more combinations that could potentially improve the results. For example, you could cross multiple columns. What happens if you vary the number of buckets? What other synthetic features can you think of? Do they improve the model?<br><br><br>

# 5.Logistic Regression

**Learning Objectives:**
  * Reframe the median house value predictor (from the preceding exercises) as a binary classification model
  * Compare the effectiveness of logisitic regression vs linear regression for a binary classification problem

As in the prior exercises, we're working with the [California housing data set](https://developers.google.com/machine-learning/crash-course/california-housing-data-description), but this time we will turn it into a binary classification problem by predicting whether a city block is a high-cost city block. We'll also revert to the default features, for now.

## Frame the Problem as Binary Classification

The target of our dataset is `median_house_value` which is a numeric (continuous-valued) feature. We can create a boolean label by applying a threshold to this continuous value.

Given features describing a city block, we wish to predict if it is a high-cost city block. To prepare the targets for train and eval data, we define a classification threshold of the 75%-ile for median house value (a value of approximately 265000). All house values above the threshold are labeled `1`, and all others are labeled `0`.

## Setup

Run the cells below to load the data and prepare the input features and targets.

Note how the code below is slightly different from the previous exercises. Instead of using `median_house_value` as target, we create a new binary target, `median_house_value_is_high`. <br>

```python
def preprocess_targets(california_housing_dataframe):  
  output_targets = pd.DataFrame()
  output_targets["median_house_value_is_high"] = (
    california_housing_dataframe["median_house_value"] > 265000).astype(float)
  return output_targets
```

## How Would Linear Regression Fare?
To see why logistic regression is effective, let us first train a naive model that uses linear regression. This model will use labels with values in the set `{0, 1}` and will try to predict a continuous value that is as close as possible to `0` or `1`. Furthermore, we wish to interpret the output as a probability, so it would be ideal if the output will be within the range `(0, 1)`. We would then apply a threshold of `0.5` to determine the label.
To train the linear regression model using [LinearRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor).
```python
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
```
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/classification/Figure_1.png)  
（Linear Regression ）  

## Task 1: Can We Calculate LogLoss for These Predictions?

**Examine the predictions and decide whether or not we can use them to calculate LogLoss.**

`LinearRegressor` uses the L2 loss, which doesn't do a great job at penalizing misclassifications when the output is interpreted as a probability.  For example, there should be a huge difference whether a negative example is classified as positive with a probability of 0.9 vs 0.9999, but L2 loss doesn't strongly differentiate these cases.

In contrast, `LogLoss` penalizes these "confidence errors" much more heavily.  Remember, `LogLoss` is defined as:
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/classification/logloss.PNG)  

But first, we'll need to obtain the prediction values. We could use `LinearRegressor.predict` to obtain these.
Given the predictions and the targets, can we calculate `LogLoss`?

### Solution
```python
validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
_ = plt.hist(validation_predictions)
```  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/classification/Figure_2.png)  
（hist(validation_predictions)）  

## Task 2: Train a Logistic Regression Model and Calculate LogLoss on the Validation Set

To use logistic regression, simply use [LinearClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier) instead of `LinearRegressor`. 

**NOTE**: When running `train()` and `predict()` on a `LinearClassifier` model, you can access the real-valued predicted probabilities via the `"probabilities"` key in the returned dict—e.g., `predictions["probabilities"]`. Sklearn's [log_loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) function is handy for calculating LogLoss using these probabilities.

### Solution
```python
# Take a break and compute predictions.    
training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
training_log_loss = metrics.log_loss(training_targets, training_probabilities)
validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
```
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/classification/Figure_3.png)  
（LogLoss on the Validation Set）  


## Task 3: Calculate Accuracy and plot a ROC Curve for the Validation Set

A few of the metrics useful for classification are the model [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification), the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and the area under the ROC curve (AUC). We'll examine these metrics.

`LinearClassifier.evaluate` calculates useful metrics like accuracy and AUC.  

```python
evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)  
print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
```
AUC on the validation set: 0.70  
Accuracy on the validation set: 0.75  

You may use class probabilities, such as those calculated by `LinearClassifier.predict`,
and Sklearn's [roc_curve](http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics) to
obtain the true positive and false positive rates needed to plot a ROC curve.  

```python
validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
# Get just the probabilities for the positive class.
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
```
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/classification/Figure_4.png)  
（LROC Curve for the Validation Set）  

**See if you can tune the learning settings of the model trained at Task 2 to improve AUC.**

Often times, certain metrics improve at the detriment of others, and you'll need to find the settings that achieve a good compromise.

**Verify if all metrics improve at the same time.**

![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/classification/Figure_5.png)  
（improve AUC ）  
```python
evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
```
AUC on the validation set: 0.75  
Accuracy on the validation set: 0.77

### Solution
One possible solution that works is to just train for longer, as long as we don't overfit. 
We can do this by increasing the number the steps, the batch size, or both.
All metrics improve at the same time, so our loss metric is a good proxy for both AUC and accuracy.
Notice how it takes many, many more iterations just to squeeze a few more units of AUC. This commonly happens. But often even this small gain is worth the costs.  
![image](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/figures/classification/Figure_6.png)  
（LogLoss on the Validation Set）  
```python
evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
```
AUC on the validation set: 0.79  
Accuracy on the validation set: 0.78  

# 6.Sparsity and L1 Regularization
[code](https://github.com/hoshinotsuki/tensorflow-gpu-test/blob/master/6.sparsity_and_l1_regularization.py)
**Learning Objectives:**
  * Calculate the size of a model
  * Apply L1 regularization to reduce the size of a model by increasing sparsity

One way to reduce complexity is to use a regularization function that encourages weights to be exactly zero. For linear models such as regression, a zero weight is equivalent to not using the corresponding feature at all. In addition to avoiding overfitting, the resulting model will be more efficient.

L1 regularization is a good way to increase sparsity.

## Setup
```python
def preprocess_features(california_housing_dataframe):
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features
```  

```python
def preprocess_targets(california_housing_dataframe):
  output_targets = pd.DataFrame()
  # Create a boolean categorical feature representing whether the
  # median_house_value is above a set threshold.
  output_targets["median_house_value_is_high"] = (
    california_housing_dataframe["median_house_value"] > 265000).astype(float)
  return output_targets
```  

```python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                            
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
```  

```python
def get_quantile_based_buckets(feature_values, num_buckets):
  quantiles = feature_values.quantile(
    [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
  return [quantiles[q] for q in quantiles.keys()]
```  

```python
def construct_feature_columns():
  bucketized_households = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("households"),
    boundaries=get_quantile_based_buckets(training_examples["households"], 10))
  bucketized_longitude = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("longitude"),
    boundaries=get_quantile_based_buckets(training_examples["longitude"], 50))
  bucketized_latitude = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("latitude"),
    boundaries=get_quantile_based_buckets(training_examples["latitude"], 50))
  bucketized_housing_median_age = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("housing_median_age"),
    boundaries=get_quantile_based_buckets(
      training_examples["housing_median_age"], 10))
  bucketized_total_rooms = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("total_rooms"),
    boundaries=get_quantile_based_buckets(training_examples["total_rooms"], 10))
  bucketized_total_bedrooms = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("total_bedrooms"),
    boundaries=get_quantile_based_buckets(training_examples["total_bedrooms"], 10))
  bucketized_population = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("population"),
    boundaries=get_quantile_based_buckets(training_examples["population"], 10))
  bucketized_median_income = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("median_income"),
    boundaries=get_quantile_based_buckets(training_examples["median_income"], 10))
  bucketized_rooms_per_person = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column("rooms_per_person"),
    boundaries=get_quantile_based_buckets(
      training_examples["rooms_per_person"], 10))

  long_x_lat = tf.feature_column.crossed_column(
    set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)

  feature_columns = set([
    long_x_lat,
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_total_rooms,
    bucketized_total_bedrooms,
    bucketized_population,
    bucketized_households,
    bucketized_median_income,
    bucketized_rooms_per_person])
  
  return feature_columns
```  

## Calculate the Model Size

To calculate the model size, we simply count the number of parameters that are non-zero. We provide a helper function below to do that. The function uses intimate knowledge of the Estimators API - don't worry about understanding how it works.  

```python
def model_size(estimator):
  variables = estimator.get_variable_names()
  size = 0
  for variable in variables:
    if not any(x in variable 
               for x in ['global_step',
                         'centered_bias_weight',
                         'bias_weight',
                         'Ftrl']
              ):
      size += np.count_nonzero(estimator.get_variable_value(variable))
  return size
```  

## Reduce the Model Size

Your team needs to build a highly accurate Logistic Regression model on the *SmartRing*, a ring that is so smart it can sense the demographics of a city block ('median_income', 'avg_rooms', 'households', ..., etc.) and tell you whether the given city block is high cost city block or not.

Since the SmartRing is small, the engineering team has determined that it can only handle a model that has **no more than 600 parameters**. On the other hand, the product management team has determined that the model is not launchable unless the **LogLoss is less than 0.35** on the holdout test set.

Can you use your secret weapon—L1 regularization—to tune the model to satisfy both the size and accuracy constraints?

### Task 1: Find a good regularization coefficient.

**Find an L1 regularization strength parameter which satisfies both constraints — model size is less than 600 and log-loss is less than 0.35 on validation set.**

The following code will help you get started. There are many ways to apply regularization to your model. Here, we chose to do it using `FtrlOptimizer`, which is designed to give better results with L1 regularization than standard gradient descent.

Again, the model will train on the entire data set, so expect it to run slower than normal.  
```python
def train_linear_classifier_model(
    learning_rate,
    regularization_strength,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    regularization_strength: A `float` that indicates the strength of the L1
       regularization. A value of `0.0` means no regularization.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    feature_columns: A `set` specifying the input feature columns to use.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearClassifier` object trained on the training data.
  """

  periods = 7
  steps_per_period = steps / periods

  # Create a linear classifier object.
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["median_house_value_is_high"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["median_house_value_is_high"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["median_house_value_is_high"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
  
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("LogLoss (on validation data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
    # Compute training and validation loss.
    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, validation_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()
  plt.show()

  return linear_classifier
```  

### Solution
A regularization strength of 0.1 should be sufficient. Note that there is a compromise to be struck:
stronger regularization gives us smaller models, but can affect the classification loss.  
```python
linear_classifier = train_linear_classifier_model(
    learning_rate=0.1,
    regularization_strength=0.1,
    steps=300,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
print("Model size:", model_size(linear_classifier))
```  
