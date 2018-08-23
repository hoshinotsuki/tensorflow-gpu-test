# we're working with the California housing data set,
# to try and predict median_house_value at the city block level from 1990 census data.

# Setup
# First off, let's load up and prepare our data.
from __future__ import print_function

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# bug 1:import 进去的时候 优化import即可

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("D:\datas\california_housing_train.csv", sep=",")

# 加上随机化处理，否则训练集和验证集的分布不一致
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


# This time, we're going to work with multiple features,
# so we'll modularize the logic for preprocessing the features a bit:
def preprocess_features(california_housing_dataframe):  # 返回一个dataframe
    """Prepares input features from California housing data set.
  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """

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


def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.
  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


# For the training set, we'll choose the first 12000 examples, out of the total of 17000.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# For the validation set, we'll choose the last 5000 examples, out of the total of 17000.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))


# Task 1: Examine the Data
# Okay, let's look at the data above. We have 9 input features that we can use.
# Take a quick skim over the table of values. Everything look okay?
# See how many issues you can spot.
# Don't worry if you don't have a background in statistics; common sense will get you far.
# After you've had a chance to look over the data yourself,
# check the solution for some additional thoughts on how to verify data.

# Solution
# Let's check our data against some baseline expectations:
# 1.For some values, like median_house_value,
# we can check to see if these values fall within reasonable ranges (keeping in mind this was 1990 data — not today!).
# 2.For other values, like latitude and longitude,
# we can do a quick check to see if these line up with位于合理的范围内expected values from a quick Google search.
# If you look closely, you may see some oddities:
# 3.median_income is on a scale from about 3 to 15.
# It's not at all clear what this scale refers to—looks like maybe some log scale?
# It's not documented anywhere; all we can assume is that higher values correspond to higher income.
# 4.The maximum median_house_value is 500,001. This looks like an artificial cap of some kind人为上限.
# 5.Our rooms_per_person feature is generally on a sane scale正常范围, with a 75th percentile value of about 2.
# But there are some very large values, like 18 or 55, which may show some amount of corruption in the data.
# We'll use these features as given for now.
# But hopefully these kinds of examples can help to build a little intuition about how to check data that comes to you from an unknown source.


# Task 2: Plot Latitude/Longitude vs. Median House Value
# Let's take a close look at two features in particular:
# latitude and longitude. These are geographical coordinates of the city block in question.
# This might make a nice visualization —
# let's plot latitude and longitude, and use color to show the median_house_value.

# 封装成一个验证数据集的函数
def Examine_Data(validation_examples, training_examples, num):
    plt.figure(num=num, figsize=(13, 16))
    ax = plt.subplot(2, 2, 1)
    ax.set_title("Validation Data")
    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.tight_layout()
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
    # 颜色代表房价
    ax = plt.subplot(2, 2, 2)
    ax.set_title("Training Data")
    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.tight_layout()
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    _ = plt.plot()


Examine_Data(
    validation_examples=validation_examples,
    training_examples=training_examples,
    num=1)


# Wait a second...this should have given us a nice map of the state of California,
# with red showing up in expensive areas like the San Francisco and Los Angeles.红色表示高房价，像洛杉矶
# The training set sort of does, compared to a real map, but the validation set clearly doesn't.训练集比验证集更像一个真正的地图
# Go back up and look at the data from Task 1 again.
# Do you see any other differences in the distributions of features or targets between the training and validation data?'''
#
# Solution
# Looking at the tables of summary stats above, it's easy to wonder how anyone would do a useful data check.如何做一个有效的数据检查
# What's the right 75th percentile value for total_rooms per city block?
# The key thing to notice is that for any given feature or column, 对于每个特征和特征列
# the distribution of values between the train and validation splits should be roughly equal.训练集和验证集的划分应该一致
# The fact that this is not the case is a real worry, 真正担心的是事实不是这样
# and shows that we likely have a fault in the way that our train and validation split was created.说明区分训练集和验证集时有错误
#
# Task 3: Return to the Data Importing and Pre-Processing Code, and See if You Spot Any Bugs
# If you do, go ahead and fix the bug. Don't spend more than a minute or two looking. If you can't find the bug, check the solution.
# When you've found and fixed the issue, re-run latitude / longitude plotting cell above and confirm that our sanity checks look better.
# By the way, there's an important lesson here.重要：ML的调试是数据调试，不是代码调试
# Debugging in ML is often data debugging rather than code debugging.
# If the data is wrong, even the most advanced ML code can't save things.
#
# Solution
# Take a look at how the data is randomized when it's read in.数据读入的时候是否随机化#
# If we don't randomize the data properly before creating training and validation splits,
# then we may be in trouble if the data is given to us in some sorted order, which appears to be the case here.
#
#
# Task 4: Train and Evaluate a Model
# Spend 5 minutes or so trying different hyperparameter settings.
# Try to get the best validation performance you can.
# Next, we'll train a linear regressor using all the features in the data set, and see how well we do.
# Let's define the same input function we've used previously for loading the data into a TensorFlow model.

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# Because we're now working with multiple input features,
# let's modularize our code for configuring feature columns into a separate function.
# (For now, this code is fairly simple, as all our features are numeric,
# but we'll build on this code as we use other types of features in future exercises.)
def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


# Next, go ahead and complete the train_model() code below to set up the input functions and calculate predictions.
# NOTE: It's okay to reference the code from the previous exercises,
# but make sure to call predict() on the appropriate data sets.

# Compare the losses on training data and validation data.
# With a single raw feature, our best root mean squared error (RMSE) was of about 180.
# See how much better you can do now that we can use multiple features.
#
# Check the data using some of the methods we've looked at before. These might include:
# Comparing distributions of predictions and actual target values
# Creating a scatter plot of predictions vs. target values
# Creating two scatter plots of validation data using latitude and longitude:
# One plot mapping color to actual target median_house_value
# A second plot mapping color to predicted median_house_value for side-by-side comparison.

def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,
        num):
    """Trains a linear regression model of multiple features.
    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.
    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.
    Returns:
      A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # 1. Create input functions.
    training_input_fn = lambda: my_input_fn(
        training_examples,  # pandas DataFrame of features样本集dataframe格式的
        training_targets["median_house_value"],
        batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets["median_house_value"],
        num_epochs=1,
        shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
        validation_examples,
        validation_targets["median_house_value"],
        num_epochs=1,
        shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # 2. Take a break and compute predictions.
        training_predictions = linear_regressor.predict(
            input_fn=predict_training_input_fn)  # 输入要预测的样本集。函数封装。Tuple of (features, labels) for next data batch
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.figure(num, figsize=(13, 16))
    ax2 = plt.subplot2grid((13, 16), (0, 7), colspan=9)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.tight_layout()
    return linear_regressor


linear_regressor = train_model(
    # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
    learning_rate=0.0001,
    steps=100,
    batch_size=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets,
    num=1)

# Task 5: Evaluate on Test Data
# In the cell below, load in the test data set and evaluate your model on it.
# We've done a lot of iteration on our validation data.
# Let's make sure we haven't overfit to the pecularities of that particular sample.
# How does your test performance compare to the validation performance?
# What does this say about the generalization performance of your model?比较rmse
california_housing_test_data = pd.read_csv("D:\datas\california_housing_test.csv", sep=",")
# 加上随机化处理，否则训练集和验证集的分布不一致
california_housing_test_data = california_housing_test_data.reindex(
    np.random.permutation(california_housing_test_data.index))
# 检查数据
# For the test set,
testing_examples = preprocess_features(california_housing_test_data)
testing_targets = preprocess_targets(california_housing_test_data)

# 验证数据集,用封装的Examine_Data函数
# Examine_Data(
#     validation_examples=testing_examples,
#     training_examples=validation_examples,
#     num=2)


# 预测 不是训练，所以用API.predict
predict_test_input_fn = lambda: my_input_fn(
    testing_examples,
    testing_targets["median_house_value"],
    num_epochs=1,
    shuffle=False)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)  # 训练要很多次。预测就只要一次
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, testing_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

plt.show()
