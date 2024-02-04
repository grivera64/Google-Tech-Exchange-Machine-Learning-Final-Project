# Twitter Disasters Final Report | April 2023

## Authors

- [Giovanni Rivera](https://github.com/grivera64); Intro to Machine Learning, Section A
- [Oscar Jesus Zambrano](https://github.com/osc-zam22); Intro to Machine Learning, Section B

#### Setup Dependencies



```python
# Ensure that we have the newest version of pip installed
%pip install -q --upgrade pip

# Install necessary libraries
%pip install -q numpy
%pip install -q pandas
%pip install -q matplotlib
%pip install -q seaborn
%pip install -q plotly

# Helps avoid showing plots in a separate line
%matplotlib inline

%pip install -q scikit-learn
%pip install -q tensorflow

# Helps run plot_model from keras
%pip install pydot

```

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.



```python
# Import the modules
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objs as plotly
from sklearn import datasets

# Set the styling of the plt plots to darkgrid
sns.set_style('darkgrid')

# Removes error messsages and sets precision to 3 decimal places
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)
```

## 1. Introduction

### Our Story:


In light of recent events around the globe, we have seen the rise of misinformation being spread on social media. In order to combat this we are using real tweets from real users on Twitter to find a way to stop misinformation from spreading, and promote real useful information to those living in or have relatives in affected areas.

### Our Task:

We are designing machine learning models using Tensorflow/Keras to identify whether a given tweet is a natural distaster.

Throughout the notebook, we will refer to tweets about a natural distaster as part of the `postive class`, while tweets that aren't about a natural disaster as part of the `negative class`.

#### 1.1 Load the Data in a Colab Notebook

We use a dataset from the Kaggle competition [Tech Exchange 2023 ML Project](https://www.kaggle.com/competitions/techexchange-2023-ml-project).

You may find the dataset by navigating to the 'Data' tab in the link aforementioned. Under 'Data Explorer', you can download the following files:

- `train.csv`: Contains the training data; we will use this dataset for our training and validation data.
- `test.csv`: Contains our testing data; we will use this dataset for making our submissions to Kaggle.


```python
# Please update the Path here to the location of your train.csv and test.csv files
path_to_csv = 'Data/techexchange-2023-ml-project'

# Load the Data Frames from the Training and Testing Data Frame
train_df = pd.read_csv(f'{path_to_csv}/train.csv')
test_df = pd.read_csv(f'{path_to_csv}/test.csv')
display(train_df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Separate Data Frame for input and outputs

input_names = ['id', 'keyword', 'location', 'text']
input_df = train_df[input_names]
display(input_df.head())

output_names = ['target']
output_df = train_df[output_names]
display(output_df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### 1.2 Convert our data into numpy arrays for usage in our ML models.

Numpy will help us take full advantage of our GPU power to quickly perform training operations.


```python
# Convert into numpy data
X_data = input_df.to_numpy()
Y_data = output_df.to_numpy().flatten()

print(X_data[:5])
print(Y_data[:5])

print(X_data.shape)
print(Y_data.shape)
```

    [[1 nan nan
      'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all']
     [4 nan nan 'Forest fire near La Ronge Sask. Canada']
     [5 nan nan
      "All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected"]
     [6 nan nan
      '13,000 people receive #wildfires evacuation orders in California ']
     [7 nan nan
      'Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school ']]
    [1 1 1 1 1]
    (7613, 4)
    (7613,)


### 1.3 Split the training data into 90% training and 10% for validation.

We use Sklearn's train_test_split to split our `train.csv` dataset to create a 90:10 testing/validation split to use further down this notebook.


```python
from sklearn.model_selection import train_test_split

# Split the data into train and test
X_train, X_validation, Y_train, Y_validation = train_test_split(X_data, Y_data, train_size=0.90)
```


```python
# Going through a few examples in the training split

for index in range(2):
    print('Text')
    print(X_train[index])
    print()
    
    print('Is Natural Disaster?')
    print(Y_train[index], 'Yes' if Y_train[index] == 1 else 'No')
    print('======================')
    print()
```

    Text
    [4917 'exploded' nan
     'Im Dead!!! My two Loves in 1 photo! My Heart exploded into a Million Pieces!!!  ?????????????? @BrandonSkeie @samsmithworld http://t.co/yEtagC2d8A']
    
    Is Natural Disaster?
    0 No
    ======================
    
    Text
    [8280 'rioting' 'heart of darkness, unholy ?'
     "@Georgous__ what alternatives? Legal alternatives? Protesting? Rioting may not be the most peaceful thing but it's a demonstration of how"]
    
    Is Natural Disaster?
    0 No
    ======================
    


## 2. Baseline

### 2.1 Create a Simple Baseline


```python
# Returns a positive result, regardless of the input
def baseline_model(text_inputs):
    return 1

# Vectorized version of the method to apply to numpy arrays properly
baseline_model_np = np.vectorize(baseline_model, signature='(n) -> ()')
```


```python
# Testing the baseline on the 
baseline_predictions_train = baseline_model_np(X_train)

for i in range(5):
    print('Input:')
    print(X_train[i])
    print()

    print('Output')
    print(Y_train[i])
    print()

    print('Prediction')
    print(baseline_predictions_train[i])
    print('==================')
    print()
```

    Input:
    [4917 'exploded' nan
     'Im Dead!!! My two Loves in 1 photo! My Heart exploded into a Million Pieces!!!  ?????????????? @BrandonSkeie @samsmithworld http://t.co/yEtagC2d8A']
    
    Output
    0
    
    Prediction
    1
    ==================
    
    Input:
    [8280 'rioting' 'heart of darkness, unholy ?'
     "@Georgous__ what alternatives? Legal alternatives? Protesting? Rioting may not be the most peaceful thing but it's a demonstration of how"]
    
    Output
    0
    
    Prediction
    1
    ==================
    
    Input:
    [4606 'emergency%20services' 'London, UK'
     'I am not an American but I have family who have served in the military work in the emergency services and work in... http://t.co/Pl2VzLrKVK']
    
    Output
    1
    
    Prediction
    1
    ==================
    
    Input:
    [7668 'panic' 'Topeka, KS'
     "The good thing is that the #Royals won't face a newbie in the playoffs. No real reason to panic."]
    
    Output
    0
    
    Prediction
    1
    ==================
    
    Input:
    [7397 'obliterated' 'Valparaiso '
     'RIZZO IS ON ???????? THAT BALL WAS OBLITERATED']
    
    Output
    0
    
    Prediction
    1
    ==================
    


### 2.2 Calculate the Log Loss of our baseline model

Since our baseline is simple, we can use this loss value to determine whether our models are more accurate than a naive approach.


```python
# Calculates Log Loss
def calculate_loss(labels, predictions):
    epsilon = 0.000001  # Prevents taking the natural log of non-positive values
    ce_values = -labels * np.log(predictions + epsilon) - (1 - labels) * np.log(1 - predictions + epsilon)
    loss = ce_values.mean()
    return loss
```


```python
# Calculate the loss on the training portion of our train data
training_loss = calculate_loss(Y_train, baseline_predictions_train)
print('Training Loss:', training_loss)

# Calculate the loss on the validation portion our our train data
baseline_predictions_validation = baseline_model_np(X_validation)
testing_loss = calculate_loss(Y_validation, baseline_predictions_validation)
print('Validation Loss:', testing_loss)
```

    Training Loss: 7.87469942823704
    Validation Loss: 7.923067964344554



```python
training_accuracy = (baseline_predictions_train == Y_train).sum() / len(Y_train)
print("Training Accuracy:", training_accuracy)

validation_accuracy = (baseline_predictions_validation == Y_validation).sum() / len(Y_validation)
print('Validation Accurracy:', validation_accuracy)
```

    Training Accuracy: 0.4300102174864983
    Validation Accurracy: 0.42650918635170604


### 2.3 Upload the baseline model to Kaggle

We can then upload a `submission.csv` file to Kaggle to find the F1 score of our baseline model.


```python
# Create the submission CSV file for our Kaggle submission
def save_to_submissions_csv(text_inputs, prediction_labels, name='submission.csv'):
    print(f'Generating "{name}" file...')

    # Extract the ids of the text inputs and flatten to a 1D ndarray
    test_ids = text_inputs[:,0].flatten()

    # Write the submission file and save to 'submission.csv'
    np.savetxt(
        name,
        np.rec.fromarrays([test_ids, prediction_labels]),
        fmt=['%s', '%d'],
        delimiter=',',
        header='id,target',
        comments=''
    )

    # Show success!
    print(f'Successfully created "{name}"')
```


```python
# Reformat the single training dataframe to an input dataframe
input_names = ['id', 'keyword', 'location', 'text']
test_input_df = test_df[input_names]

# Reformat the input dataframe into a numpy array for running through our model
test_input_np = test_input_df.to_numpy()

# Predict by using the baseline model on the test input and save to a .csv
baseline_predictions_test = baseline_model_np(test_input_np)
save_to_submissions_csv(test_input_np, baseline_predictions_test, 'baseline_submission.csv')
```

    Generating "baseline_submission.csv" file...
    Successfully created "baseline_submission.csv"



```python
# Look at the first few predictions to ensure things went smoothly
pd.read_csv('baseline_submission.csv').head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Data Analysis

### 3.1 Explore the frequency of common words

Tweets can have all sorts of words inside of them, but we can always find common words that we can use for our vocabulary

The motivation for this graph to see what type of words are commonly used within our tweets dataset to see what we should and should not include in our vocabulary.


```python
from collections import Counter

# Plots the frequency of the top commonly used words in the provided tweets in
# the positive and negative class
def plot_frequency(tweets_np, labels_np, max_word_num=20):
    # Creates a counter that keeps track of the frequency of words (similar to defaultdict)
    pos_counter = Counter()
    neg_counter = Counter()
    tweets_np = np.copy(tweets_np) # Deep Copy of input

    # Total Counter Variables
    pos_counter_all = 0
    neg_counter_all = 0

    # Go through the tweets dataset
    total_words = set()
    for entry_index in range(tweets_np.shape[0]):
        # Flatten all of the features into a single string
        words = ' '.join([str(feature).lower() for feature in tweets_np[entry_index]])
        # Count the frequency of each word
        for word in words.split():
            # Group all links as 1 token
            if word.startswith('http'):
                word = '<LINK>'
            # Group all articles as 1 token
            elif word in ['the', 'a', 'an']:
                word = '<ARTICLE>'
            
            if labels_np[entry_index]:
                pos_counter[word] += 1  # Positive entry
                pos_counter_all += 1
            else:
                neg_counter[word] += 1  # Negative entry
                neg_counter_all += 1

            # For Debug purposes, saving all of the words we encounter
            total_words.add(word)
    
    # Extract at most max_word_num words that are the most common words
    # for both classes (and removes overlap)
    top_pos_words = [word for word, _ in pos_counter.most_common(max_word_num // 2)]
    top_neg_words = [word for word, _ in neg_counter.most_common(max_word_num // 2)]
    top_words = set(top_pos_words + top_neg_words)

    # Create a Data Frame for the collected data
    result = {
        'word': [word for word in top_words],
        'pos count': [pos_counter[word] for word in top_words],
        'neg count': [neg_counter[word] for word in top_words],
        '% chance is pos': [(pos_counter[word] / (pos_counter[word] + neg_counter[word])) * 100 for word in top_words]
    }
    word_count_df = pd.DataFrame(data=result, columns=result.keys())
    word_count_df = word_count_df.set_index('word')
    display(word_count_df)
    
    # Plot a bar graph that groups pos and neg count for a few of the most used words
    pd.concat([word_count_df[['pos count']], word_count_df[['neg count']]], axis=1).plot.bar()

    print(f'DEBUG: Total Words Len: {len(total_words)}')
    print(f'DEBUG: Total Positive Examples: {pos_counter_all}')
    print(f'DEBUG: Total Negative Examples: {neg_counter_all}')
```


```python
# Display and plot at most 40 words from the X_data set
plot_frequency(X_data, Y_data, max_word_num=40)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pos count</th>
      <th>neg count</th>
      <th>% chance is pos</th>
    </tr>
    <tr>
      <th>word</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>with</th>
      <td>188</td>
      <td>382</td>
      <td>32.982456</td>
    </tr>
    <tr>
      <th>for</th>
      <td>399</td>
      <td>489</td>
      <td>44.932432</td>
    </tr>
    <tr>
      <th>to</th>
      <td>761</td>
      <td>1195</td>
      <td>38.905930</td>
    </tr>
    <tr>
      <th>-</th>
      <td>419</td>
      <td>396</td>
      <td>51.411043</td>
    </tr>
    <tr>
      <th>and</th>
      <td>512</td>
      <td>927</td>
      <td>35.580264</td>
    </tr>
    <tr>
      <th>&lt;ARTICLE&gt;</th>
      <td>2426</td>
      <td>3372</td>
      <td>41.842014</td>
    </tr>
    <tr>
      <th>i</th>
      <td>292</td>
      <td>1061</td>
      <td>21.581670</td>
    </tr>
    <tr>
      <th>my</th>
      <td>134</td>
      <td>566</td>
      <td>19.142857</td>
    </tr>
    <tr>
      <th>that</th>
      <td>181</td>
      <td>357</td>
      <td>33.643123</td>
    </tr>
    <tr>
      <th>of</th>
      <td>957</td>
      <td>935</td>
      <td>50.581395</td>
    </tr>
    <tr>
      <th>you</th>
      <td>125</td>
      <td>629</td>
      <td>16.578249</td>
    </tr>
    <tr>
      <th>in</th>
      <td>1186</td>
      <td>854</td>
      <td>58.137255</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>1117</td>
      <td>1479</td>
      <td>43.027735</td>
    </tr>
    <tr>
      <th>new</th>
      <td>132</td>
      <td>292</td>
      <td>31.132075</td>
    </tr>
    <tr>
      <th>after</th>
      <td>181</td>
      <td>75</td>
      <td>70.703125</td>
    </tr>
    <tr>
      <th>&lt;LINK&gt;</th>
      <td>2510</td>
      <td>2196</td>
      <td>53.336167</td>
    </tr>
    <tr>
      <th>is</th>
      <td>348</td>
      <td>590</td>
      <td>37.100213</td>
    </tr>
    <tr>
      <th>at</th>
      <td>305</td>
      <td>237</td>
      <td>56.273063</td>
    </tr>
    <tr>
      <th>it</th>
      <td>164</td>
      <td>342</td>
      <td>32.411067</td>
    </tr>
    <tr>
      <th>as</th>
      <td>171</td>
      <td>160</td>
      <td>51.661631</td>
    </tr>
    <tr>
      <th>this</th>
      <td>179</td>
      <td>287</td>
      <td>38.412017</td>
    </tr>
    <tr>
      <th>from</th>
      <td>247</td>
      <td>183</td>
      <td>57.441860</td>
    </tr>
    <tr>
      <th>by</th>
      <td>279</td>
      <td>242</td>
      <td>53.550864</td>
    </tr>
    <tr>
      <th>on</th>
      <td>418</td>
      <td>444</td>
      <td>48.491879</td>
    </tr>
    <tr>
      <th>be</th>
      <td>113</td>
      <td>287</td>
      <td>28.250000</td>
    </tr>
  </tbody>
</table>
</div>


    DEBUG: Total Words Len: 33580
    DEBUG: Total Positive Examples: 61864
    DEBUG: Total Negative Examples: 80205



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_32_2.png)
    


#### 3.1.1

After reviewing this information, there appears to be a lot of words that are partially helpful, but we don't see any keywords that we expected such as "fire", or other natural disaster words.

As a result, we will need to have a lot of words in our vocabulary to include these helpful words as well, as the expected keywords that we expected.

### 3.2 Explore the length of the input

We were thinking that the number of tokens in a tweet might help with determining whether a tween is a disaster or not.


```python
from collections import defaultdict

def length_plot(tweets_np , labels_np, interval_list):

    # initializes maps to count based on intervals of words
    pos_intervals = defaultdict(int)
    neg_intervals = defaultdict(int)

    # Convert the interval list into ranges for use below
    interval_map = {}
    for interval in interval_list:
        # Parses interval strings into useable ranges
        if '-' in interval:
            start, end = map(lambda x: int(x), interval.split('-'))
        else:
            start, end = interval.split('+')[0], 285

        interval_map[interval] = range(int(start), int(end) + 1)

    # Track the counts of positive and negative inputs for each range from above
    for entry_index in range(tweets_np.shape[0]):
        # Flatten all of the features into a single string
        words = ' '.join([str(feature) for feature in tweets_np[entry_index]]).split()

        for interval in interval_list:
            # Ignore words counts outside of our intervals
            if len(words) not in interval_map[interval]:
                continue

            # Updates the counts of positive and negative entries in the dictionary
            # based on the interval they are in
            if labels_np[entry_index]:
                pos_intervals[interval] += 1
            else:
                neg_intervals[interval] += 1

    # Create a Data Frame for the collected data
    result= {
        'interval' : [ interval for interval in interval_list],
        'pos intervals' : [pos_intervals[interval] for interval in interval_list],
        'neg intervals' : [neg_intervals[interval] for interval in interval_list],
        '% chance is pos': [(pos_intervals[interval] / (pos_intervals[interval] + neg_intervals[interval])) * 100 for interval in interval_list],
    }
    word_count_df = pd.DataFrame(data = result , columns=result.keys())
    word_count_df = word_count_df.set_index('interval')

    # Plot and Display the collected data
    pd.concat([word_count_df[['pos intervals']], word_count_df[['neg intervals']]], axis=1).plot.bar()
    display(word_count_df)

    # Debugging code
    print(neg_intervals)
    print(pos_intervals)
```


```python
# Plots the count of positive and negative classes of X_data based on the given intervals
intervals = ['0-5', '6-10' , '11-15' , '16-20' , '21-25' , '26-30' , '31-35', '36+']
length_plot(X_data, Y_data, intervals)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pos intervals</th>
      <th>neg intervals</th>
      <th>% chance is pos</th>
    </tr>
    <tr>
      <th>interval</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0-5</th>
      <td>2</td>
      <td>28</td>
      <td>6.666667</td>
    </tr>
    <tr>
      <th>6-10</th>
      <td>164</td>
      <td>463</td>
      <td>26.156300</td>
    </tr>
    <tr>
      <th>11-15</th>
      <td>740</td>
      <td>1017</td>
      <td>42.117245</td>
    </tr>
    <tr>
      <th>16-20</th>
      <td>1041</td>
      <td>1165</td>
      <td>47.189483</td>
    </tr>
    <tr>
      <th>21-25</th>
      <td>973</td>
      <td>1010</td>
      <td>49.067070</td>
    </tr>
    <tr>
      <th>26-30</th>
      <td>317</td>
      <td>552</td>
      <td>36.478711</td>
    </tr>
    <tr>
      <th>31-35</th>
      <td>34</td>
      <td>106</td>
      <td>24.285714</td>
    </tr>
    <tr>
      <th>36+</th>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    defaultdict(<class 'int'>, {'6-10': 463, '0-5': 28, '16-20': 1165, '11-15': 1017, '21-25': 1010, '26-30': 552, '31-35': 106, '36+': 1})
    defaultdict(<class 'int'>, {'16-20': 1041, '6-10': 164, '21-25': 973, '11-15': 740, '26-30': 317, '31-35': 34, '0-5': 2, '36+': 0})



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_36_2.png)
    


#### 3.2.1

It looks like there is a fair distribution based on the number of words in the tweet.

After viewing this graph, we figured that it may be best to not include the number of words in our model inputs. Bucketing the tweets based on these values may not be worth the extra model complexity.

## 4. Our Experiments

We were able to come up with three differing models using different algorithms learned in class, each varying in complexity. various differing variations of each model were ran however we kept the best versions of each model along with an explanation of how we reached that point

### 4.1 Normalize our data

Since tweets can contain lots of punctuation, we want to specify a standardization to sanitize our inputs. In addition, all links are changed to <LINK> as a way of reducing the creation of unique features. 
> **NOTE** Though standarization helps ensure that our model works for unusual input, there may be a lost 
> of information when applying normalizations.

#### 4.1.1 Normalization Technique 1: Remove punctuation and Links


```python
# Stardardizes the input
def normalize_punctuation_and_links(tweets_text, show_debug=False):
    # Make all letters lowercase
    result_tensor = tf.strings.lower(tweets_text)

    # Replace links with <LINK> token
    link_regex = r'(https?:\/\/)([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)'
    result_tensor = tf.strings.regex_replace(result_tensor, link_regex, '<LINK>')

    # Remove punctuation (but don't remove <> from our Link tokens)
    punctuation_regex = r'[^\w\s<>]'
    result_tensor = tf.strings.regex_replace(result_tensor, punctuation_regex, ' ')

    # Remove extra spaces
    multi_space_regex = r'\s{2,}'
    result_tensor = tf.strings.regex_replace(result_tensor, multi_space_regex, ' ')

    if show_debug:
        print('DEBUG: ', end='')
        tf.print(result_tensor)

    return result_tensor

normalize_punctuation_and_links('I.am.cool  http://www.example.com, https://github.com/example', show_debug=True)
```

    DEBUG: i am cool <LINK> <LINK>





    <tf.Tensor: shape=(), dtype=string, numpy=b'i am cool <LINK> <LINK>'>



### 4.2 Build Our Model(s)

#### 4.2.1 FFNN Model (Bag Of Embeddings)

##### 4.2.1.1
Our first model is a fairly simple FFNN. We used the concept of "Bag of Embeddings", where we use embeddings with dim=3 to try to identify which words are related to each other.

A problem with this model is that, as the name suggests, the words are not in any particular order (just in a messy "bag"), but there is a lot of information that the model can learn without word order.


```python
def build_ffnn_model(tweets_np, max_vocab, max_tokens, embedding_dim):
    # Remove randomness
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(0)
    
    # Build the normalization layer and vocabulary ahead of time
    norm_layer = tf.keras.layers.TextVectorization(
        max_tokens,
        standardize=normalize_punctuation_and_links,
        split='whitespace',
        output_mode='int',
        encoding='utf-8',
        name='Normalization_Layer',
        output_sequence_length=max_tokens,
    )
    norm_layer.adapt(tweets_np, batch_size=64)

    # display(norm_layer.get_vocabulary())

    # Build our FFNN Model using embeddings and average pooling
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(norm_layer)
    model.add(tf.keras.layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        input_length=max_tokens,
        name='Embeddings_Layer',
    ))
    # max pool layer
    model.add(tf.keras.layers.GlobalMaxPooling1D(
        name='Max_Pool_Layer'
    ))
    # two dense layer with a sigmoid activation
    model.add(tf.keras.layers.Dense(
        units=128,  
        activation='sigmoid',
        name='Dense_Layer_1',
    ))
    model.add(tf.keras.layers.Dense(
        units=64,
        activation='sigmoid',
        name='Dense_Layer_2',
    ))

    # output layer with a sigmoid activation 
    model.add(tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        name='Output_Layer',
    ))

    # Compile and return the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

A large vocabulary with seems to have a direct benefit to the model as more features can be identified. An input of 300 with a dimension of 12 was also found to have a good impact through tedious trial and error. The biggest obstacle was finding the correct amount of training. This model was very prone to overfitting data if it was trained past 11 epochs.


```python
ffnn_model = build_ffnn_model(X_train[:,3], 35000, 300, 12)
display(ffnn_model.summary())
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     Normalization_Layer (TextVe  (None, 300)              0         
     ctorization)                                                    
                                                                     
     Embeddings_Layer (Embedding  (None, 300, 12)          420000    
     )                                                               
                                                                     
     Max_Pool_Layer (GlobalMaxPo  (None, 12)               0         
     oling1D)                                                        
                                                                     
     Dense_Layer_1 (Dense)       (None, 128)               1664      
                                                                     
     Dense_Layer_2 (Dense)       (None, 64)                8256      
                                                                     
     Output_Layer (Dense)        (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 429,985
    Trainable params: 429,985
    Non-trainable params: 0
    _________________________________________________________________



    None


#### 4.2.2 CNN Model

##### 4.2.2.1
The second model made was a CNN. This model is also fairly standard. This model applies filters along the features to detect any patterns that are useful for our predictions. We still use embeddings since these is useful for identifying if words are related.

Giovanni put this together and got it running with originally 2 conv1D layers, which later concantenated together.


```python
def build_cnn_model(tweets_np, max_vocab, max_tokens, embedding_dim):
    # Remove randomness
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(0)
    
    # Build the normalization layer ahead of time
    norm_layer = tf.keras.layers.TextVectorization(
        max_tokens,
        standardize=normalize_punctuation_and_links,
        split='whitespace',
        output_mode='int',
        encoding='utf-8',
        name='Normalization_Layer',
        output_sequence_length=max_tokens,
    )
    norm_layer.adapt(tweets_np, batch_size=64)

    # Build our CNN Model using Keras' Functional API
    input_layer = tf.keras.Input(shape=(1,), dtype=tf.string, name='Input Layer')
    norm_layer = norm_layer(input_layer)
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        input_length=max_tokens,
        name='Embeddings_Layer',
    )(norm_layer)
    
    # three conv1D layers, filter sizes increase as it progresses
    conv_1 = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='Conv_Layer_1_1',
    )(embedding_layer)
    conv_2 = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=4,
        padding='same',
        activation='relu',
        name='Conv_Layer_1_2',
    )(embedding_layer)
    conv_3 = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        padding='same',
        activation='relu',
        name='Conv_Layer_1_3',
    )(embedding_layer)

    # combines layers into single layers (Dropout adds some randomness, while MaxPool1D helps shrink our layers)
    concat_layer = tf.keras.layers.Concatenate(name='Concatenate_Layer')([conv_1, conv_2, conv_3])
    dropout_layer = tf.keras.layers.Dropout(rate=0.05)(concat_layer)
    max_pool_layer = tf.keras.layers.MaxPool1D(pool_size=max_tokens, name='Max_Pool_Layer')(dropout_layer)
    flatten_layer = tf.keras.layers.Flatten(name='Flatten_Layer_1')(max_pool_layer)
    
    # hidden layer of size 32, sigmoid
    hidden_layer = tf.keras.layers.Dense(
        units=32,
        activation='sigmoid',
        name='Hidden_Layer_1',
    )(flatten_layer)

    # output layer
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        name='Output_Layer',
    )(hidden_layer)

    # adds input and output layer to model
    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output_layer,
        name='CNN_Model',
    )

    # Compile and return the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

##### 4.2.2.2

As in the previous model, a larger vocab size, and embedding Layer (input/dimension), had a huge boost. This may be due to using the same preprocessing funcion across the board. We first started with just 2 convolutional layers. After doing some research online, and seeing that common practice was to use a smaller filter size and increase through each of the convolutional layers, we had some better success. Our reasoning behind this choice was that the context of the input is widened as it is processed through the model. Our final change was adding one more convolutional layer, which brought the total of those type of layers to three.



```python
cnn_model = build_cnn_model(X_train[:,3], 20000, 300, 9)
display(cnn_model.summary())
```

    Model: "CNN_Model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     Input Layer (InputLayer)       [(None, 1)]          0           []                               
                                                                                                      
     Normalization_Layer (TextVecto  (None, 300)         0           ['Input Layer[0][0]']            
     rization)                                                                                        
                                                                                                      
     Embeddings_Layer (Embedding)   (None, 300, 9)       180000      ['Normalization_Layer[0][0]']    
                                                                                                      
     Conv_Layer_1_1 (Conv1D)        (None, 300, 16)      448         ['Embeddings_Layer[0][0]']       
                                                                                                      
     Conv_Layer_1_2 (Conv1D)        (None, 300, 16)      592         ['Embeddings_Layer[0][0]']       
                                                                                                      
     Conv_Layer_1_3 (Conv1D)        (None, 300, 16)      736         ['Embeddings_Layer[0][0]']       
                                                                                                      
     Concatenate_Layer (Concatenate  (None, 300, 48)     0           ['Conv_Layer_1_1[0][0]',         
     )                                                                'Conv_Layer_1_2[0][0]',         
                                                                      'Conv_Layer_1_3[0][0]']         
                                                                                                      
     dropout (Dropout)              (None, 300, 48)      0           ['Concatenate_Layer[0][0]']      
                                                                                                      
     Max_Pool_Layer (MaxPooling1D)  (None, 1, 48)        0           ['dropout[0][0]']                
                                                                                                      
     Flatten_Layer_1 (Flatten)      (None, 48)           0           ['Max_Pool_Layer[0][0]']         
                                                                                                      
     Hidden_Layer_1 (Dense)         (None, 32)           1568        ['Flatten_Layer_1[0][0]']        
                                                                                                      
     Output_Layer (Dense)           (None, 1)            33          ['Hidden_Layer_1[0][0]']         
                                                                                                      
    ==================================================================================================
    Total params: 183,377
    Trainable params: 183,377
    Non-trainable params: 0
    __________________________________________________________________________________________________



    None


#### 4.2.3 RNN model



##### 4.2.3.1

The last model we tried was an RNN. Redundant neural networks are still new to us, so it was really difficult to build this.

In comparison to the other models, RNNs process input data using a Bidirectional layer (bi meaning 2 directions, forwards and backwards). This ensures that the model can use information from previous training and future training to change its weights in the back propagation process.

Oscar was able to get this model working and we trust that this model will help us predict the best label.


```python
def build_rnn_model(tweets_np, max_vocab, max_tokens, embedding_dim):
    
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(0)


    # create normalized vocab
    norm = tf.keras.layers.TextVectorization(
        max_tokens,
        standardize=normalize_punctuation_and_links,
        split='whitespace',
        output_mode='int',
        encoding='utf-8',
        name='Normalization_Layer',
        output_sequence_length=max_tokens,
    )

    norm.adapt(tweets_np , batch_size=64)

    # Uses Sequential api with guidance from the tutorial https://www.tensorflow.org/text/tutorials/text_classification_rnn
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string, name='Input Layer'))
    model.add(norm)
    model.add(tf.keras.layers.Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        input_length=max_tokens,
        name='Embeddings_Layer',
    ))

    # bidirectional layers
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences= True), name='Bidirectional_Layer_1'))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True ), name='Bidirectional_Layer_2'))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), name='Bidirectional_Layer_3'))
    
    # hidden sigmoid layers
    model.add(tf.keras.layers.Dense(128, activation='sigmoid', name='hidden_layer_1'))
    model.add(tf.keras.layers.Dense(64, activation='sigmoid', name='hidden_layer_2'))
    
    # output layers
    model.add(tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        name='Output_Layer',
    ))

    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
```


```python
rnn_model = build_rnn_model(X_train[:,3], 30000, 300, 9)
display(rnn_model.summary())
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     Normalization_Layer (TextVe  (None, 300)              0         
     ctorization)                                                    
                                                                     
     Embeddings_Layer (Embedding  (None, 300, 9)           270000    
     )                                                               
                                                                     
     Bidirectional_Layer_1 (Bidi  (None, 300, 128)         37888     
     rectional)                                                      
                                                                     
     Bidirectional_Layer_2 (Bidi  (None, 300, 128)         98816     
     rectional)                                                      
                                                                     
     Bidirectional_Layer_3 (Bidi  (None, 128)              98816     
     rectional)                                                      
                                                                     
     hidden_layer_1 (Dense)      (None, 128)               16512     
                                                                     
     hidden_layer_2 (Dense)      (None, 64)                8256      
                                                                     
     Output_Layer (Dense)        (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 530,353
    Trainable params: 530,353
    Non-trainable params: 0
    _________________________________________________________________



    None


## 5. Error Analysis


```python
def plot_history(history, epochs):
    history = pd.DataFrame(history)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')

    plt.plot(list(range(1, epochs + 1)), history['loss'], label="Train")
    plt.plot(list(range(1, epochs + 1)), history['val_loss'], label="Validation")

    plt.legend(loc='best')
    plt.show()

    print('Loss:', history['loss'].iloc[-1])
    print('Val Loss:', history['val_loss'].iloc[-1])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (in %)')
    plt.title('Accuracy vs Epoch')

    plt.plot(list(range(1, epochs + 1)), history['accuracy'] * 100, label="Train")
    plt.plot(list(range(1, epochs + 1)), history['val_accuracy'] * 100, label="Validation")

    plt.legend(loc='best')
    plt.show()

    print('Accuracy:', history['accuracy'].iloc[-1])
    print('Val Accuracy:', history['val_accuracy'].iloc[-1])
```


```python
def train_and_analyze_model(model, tweet_features, labels, num_of_epochs=16, num_per_batch=16, validation=0.1):
    # Remove randomness
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(0)
        
    history = model.fit(
        tweet_features[:,3],
        labels,
        epochs=num_of_epochs,
        batch_size=num_per_batch,
        validation_split=validation,
        verbose=1,
    )

    plot_history(history.history, num_of_epochs)
```


```python
def plot_predictions(predictions, actual):
    from matplotlib.ticker import PercentFormatter
    plt.title('Prediction Distribution')
    plt.xlabel('Positive Confidence Level (in %)')
    plt.ylabel('# of predictions')
    plt.gca().xaxis.set_major_formatter(PercentFormatter())

    plt.hist([predictions[actual == 0] * 100, predictions[actual == 1] * 100], label=['negative', 'positive'])
    plt.legend(loc='best')
    plt.show()
```


```python
def plot_difference(predictions, actual):
    plt.title('Prediction Difference Distribution')
    plt.xlabel('Positive Confidence Level (in %)')
    plt.ylabel('Difference')

    diff = abs(predictions - actual)

    plt.hist([diff[actual == 0] * 100, diff[actual == 1] * 100], label=['negative', 'positive'])
    plt.show()
```


```python
def print_confusion_matrix(predictions, actual, threshold):
    from sklearn.metrics import confusion_matrix

    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0
    tn, fp, fn, tp = confusion_matrix(actual, predictions).ravel()

    print('True Positives:', tp)
    print('True Negatives:', tn)
    print('False Positives:', fp)
    print('False Negatives:', fn)
    print()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
```

### 5.1 Train and Anaylze the FFNN Model


```python
train_and_analyze_model(
    ffnn_model,
    X_train,
    Y_train,
    num_of_epochs=10,
    num_per_batch=64,
    validation=0.1,
)
```

    Epoch 1/10
    97/97 [==============================] - 2s 12ms/step - loss: 0.6862 - accuracy: 0.5689 - val_loss: 0.6785 - val_accuracy: 0.5802
    Epoch 2/10
    97/97 [==============================] - 1s 7ms/step - loss: 0.6715 - accuracy: 0.5732 - val_loss: 0.6377 - val_accuracy: 0.5875
    Epoch 3/10
    97/97 [==============================] - 1s 7ms/step - loss: 0.5872 - accuracy: 0.7155 - val_loss: 0.5403 - val_accuracy: 0.7391
    Epoch 4/10
    97/97 [==============================] - 1s 7ms/step - loss: 0.5152 - accuracy: 0.7543 - val_loss: 0.5100 - val_accuracy: 0.7522
    Epoch 5/10
    97/97 [==============================] - 1s 7ms/step - loss: 0.4819 - accuracy: 0.7739 - val_loss: 0.4972 - val_accuracy: 0.7711
    Epoch 6/10
    97/97 [==============================] - 1s 8ms/step - loss: 0.4651 - accuracy: 0.7849 - val_loss: 0.4946 - val_accuracy: 0.7682
    Epoch 7/10
    97/97 [==============================] - 1s 7ms/step - loss: 0.4538 - accuracy: 0.7893 - val_loss: 0.4952 - val_accuracy: 0.7726
    Epoch 8/10
    97/97 [==============================] - 1s 8ms/step - loss: 0.4459 - accuracy: 0.7942 - val_loss: 0.4987 - val_accuracy: 0.7595
    Epoch 9/10
    97/97 [==============================] - 1s 7ms/step - loss: 0.4386 - accuracy: 0.7958 - val_loss: 0.4976 - val_accuracy: 0.7726
    Epoch 10/10
    97/97 [==============================] - 1s 7ms/step - loss: 0.4332 - accuracy: 0.8016 - val_loss: 0.4995 - val_accuracy: 0.7580



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_65_1.png)
    


    Loss: 0.4331941604614258
    Val Loss: 0.4994606077671051



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_65_3.png)
    


    Accuracy: 0.8016220331192017
    Val Accuracy: 0.7580174803733826


#### 5.1.1
Here above we see that the model converges pretty well. One small issue that we have noticed is that it does not generalize for all data too well, but the results are acceptable for a simple model that is not too complex.

**Notes**

After multiple runs with different parameters and layers/layer sizes, we noticed that it trained and converged faster then the rest of the models. Adding more hidden layers with more nuerons did not have a significant impact on this model's accuracy, in fact it may have hurt it. As the writing of this, it is the best performing model we have however, it does not have much room to grow. One way that can maybe help this function is the using another process to pre-process the data that is far more complex that also can handle potential typos that can occur in online discourse.


```python
ffnn_train_predictions = ffnn_model.predict(X_train[:, 3]).flatten()
ffnn_train_loss = calculate_loss(Y_train, ffnn_train_predictions)
print(f'Train Loss: {ffnn_train_loss}')

ffnn_validation_predictions = ffnn_model.predict(X_validation[:, 3]).flatten()
ffnn_validation_loss = calculate_loss(Y_validation, ffnn_validation_predictions)
print(f'Val Loss: {ffnn_validation_loss}')
```

    215/215 [==============================] - 1s 2ms/step
    Train Loss: 0.43176578291667506
    24/24 [==============================] - 0s 2ms/step
    Val Loss: 0.522468330900813



```python
plot_predictions(ffnn_train_predictions, Y_train)
plot_difference(ffnn_train_predictions, Y_train)
```


    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_68_0.png)
    



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_68_1.png)
    



```python
print_confusion_matrix(ffnn_train_predictions, Y_train, 0.7)
```

    True Positives: 1587
    True Negatives: 3756
    False Positives: 149
    False Negatives: 1359
    
    Accuracy: 0.7798861480075902
    Precision: 0.9141705069124424
    Recall: 0.5386965376782077


#### 5.1.2
The performance of this model was middle of the pack after modifications were made. The data tables above show the distribution of predictions using two different metrics. Looking at the raw distribution of predictions, we can see that the model might fair best using a threshold between .7 and .8 to get the maximum results. In addition, when looking at the difference in distributions, we can see the model is making a lot of errors in which many false negatives are being predicted. This can be derived from both graphs. The prediciton distribution graph shows a fair amount of positives near the middle of the graph, which will all be missed with a high threshold. In the second graph we see larger amount of positives to the right of the graph signaling false negatives. 


```python
ffnn_predictions = ffnn_model.predict(test_input_np[:, 3]).flatten()

# Apply threshold
THRESHOLD = 0.7
ffnn_predictions[ffnn_predictions >= THRESHOLD] = 1
ffnn_predictions[ffnn_predictions < THRESHOLD] = 0

save_to_submissions_csv(test_input_np, ffnn_predictions, 'ffnn_submission.csv')
```

    102/102 [==============================] - 0s 2ms/step
    Generating "ffnn_submission.csv" file...
    Successfully created "ffnn_submission.csv"



```python
pd.read_csv('ffnn_submission.csv').head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 5.2 Train and Evaluate the CNN Model


```python
train_and_analyze_model(
    cnn_model,
    X_train,
    Y_train,
    num_of_epochs=10,
    num_per_batch=32,
    validation=0.1,
)
```

    Epoch 1/10
    193/193 [==============================] - 4s 14ms/step - loss: 0.7237 - accuracy: 0.5306 - val_loss: 0.6533 - val_accuracy: 0.5875
    Epoch 2/10
    193/193 [==============================] - 2s 13ms/step - loss: 0.6209 - accuracy: 0.6701 - val_loss: 0.5798 - val_accuracy: 0.7012
    Epoch 3/10
    193/193 [==============================] - 2s 12ms/step - loss: 0.5476 - accuracy: 0.7294 - val_loss: 0.5098 - val_accuracy: 0.7595
    Epoch 4/10
    193/193 [==============================] - 2s 12ms/step - loss: 0.4848 - accuracy: 0.7721 - val_loss: 0.4892 - val_accuracy: 0.7770
    Epoch 5/10
    193/193 [==============================] - 2s 13ms/step - loss: 0.4684 - accuracy: 0.7826 - val_loss: 0.4871 - val_accuracy: 0.7755
    Epoch 6/10
    193/193 [==============================] - 3s 13ms/step - loss: 0.4613 - accuracy: 0.7857 - val_loss: 0.4884 - val_accuracy: 0.7653
    Epoch 7/10
    193/193 [==============================] - 3s 13ms/step - loss: 0.4546 - accuracy: 0.7896 - val_loss: 0.4866 - val_accuracy: 0.7770
    Epoch 8/10
    193/193 [==============================] - 3s 13ms/step - loss: 0.4495 - accuracy: 0.7909 - val_loss: 0.4870 - val_accuracy: 0.7711
    Epoch 9/10
    193/193 [==============================] - 3s 13ms/step - loss: 0.4443 - accuracy: 0.7990 - val_loss: 0.4929 - val_accuracy: 0.7638
    Epoch 10/10
    193/193 [==============================] - 3s 14ms/step - loss: 0.4402 - accuracy: 0.7982 - val_loss: 0.4905 - val_accuracy: 0.7682



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_74_1.png)
    


    Loss: 0.44017520546913147
    Val Loss: 0.4905254542827606



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_74_3.png)
    


    Accuracy: 0.7982157468795776
    Val Accuracy: 0.7682215571403503


#### 5.2.1 
Using the scatter plots above, we can see that this model might actually be overfitting the data. During the validation, it does not generalize new data well and struggles a little.


```python
cnn_train_predictions = cnn_model.predict(X_train[:, 3]).flatten()
cnn_train_loss = calculate_loss(Y_train, cnn_train_predictions)
print(f'Train Loss: {cnn_train_loss}')

cnn_validation_predictions = cnn_model.predict(X_validation[:, 3]).flatten()
cnn_validation_loss = calculate_loss(Y_validation, cnn_validation_predictions)
print(f'Val Loss: {cnn_validation_loss}')
```

    215/215 [==============================] - 1s 4ms/step
    Train Loss: 0.43194361795525693
    24/24 [==============================] - 0s 5ms/step
    Val Loss: 0.5099057625701744



```python
plot_predictions(cnn_train_predictions, Y_train)
plot_difference(cnn_train_predictions, Y_train)
```


    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_77_0.png)
    



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_77_1.png)
    



```python
print_confusion_matrix(cnn_train_predictions, Y_train, 0.7)
```

    True Positives: 1613
    True Negatives: 3771
    False Positives: 134
    False Negatives: 1333
    
    Accuracy: 0.7858706758137498
    Precision: 0.9232970807097882
    Recall: 0.5475220638153429


### 5.2.2
This model has a similar problem to the first model, the same pattern appears here where an decent chunk of false negatives are being mislabeled. One reason for this may be the text processing techinque and use of embeddings in combination. In contrast, negative results are being predicted with a reasonable amount of accuracy.


```python
cnn_predictions = cnn_model.predict(test_input_np[:, 3]).flatten()

# Apply threshold
THRESHOLD = 0.7
cnn_predictions[cnn_predictions >= THRESHOLD] = 1
cnn_predictions[cnn_predictions < THRESHOLD] = 0

save_to_submissions_csv(test_input_np, cnn_predictions, 'cnn_submission.csv')
```

    102/102 [==============================] - 0s 4ms/step
    Generating "cnn_submission.csv" file...
    Successfully created "cnn_submission.csv"



```python
pd.read_csv('cnn_submission.csv').head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 5.3 Train and Evaluate the RNN Model

> **NOTE:** This RNN model has the longest training time. If you are in a time crunch and are only interested in viewing our results, please view the already saved output for the next few cells.


```python
                                                                   
train_and_analyze_model(
    rnn_model,
    X_train,
    Y_train,
    num_of_epochs=10,
    num_per_batch=32,
    validation=0.1,
)
```

    Epoch 1/10
    193/193 [==============================] - 119s 547ms/step - loss: 0.6519 - accuracy: 0.6076 - val_loss: 0.5534 - val_accuracy: 0.7362
    Epoch 2/10
    193/193 [==============================] - 99s 513ms/step - loss: 0.5288 - accuracy: 0.7457 - val_loss: 0.5017 - val_accuracy: 0.7566
    Epoch 3/10
    193/193 [==============================] - 99s 515ms/step - loss: 0.4947 - accuracy: 0.7674 - val_loss: 0.4997 - val_accuracy: 0.7566
    Epoch 4/10
    193/193 [==============================] - 97s 500ms/step - loss: 0.4834 - accuracy: 0.7734 - val_loss: 0.4887 - val_accuracy: 0.7697
    Epoch 5/10
    193/193 [==============================] - 94s 485ms/step - loss: 0.4741 - accuracy: 0.7810 - val_loss: 0.4818 - val_accuracy: 0.7799
    Epoch 6/10
    193/193 [==============================] - 98s 508ms/step - loss: 0.4710 - accuracy: 0.7815 - val_loss: 0.4940 - val_accuracy: 0.7609
    Epoch 7/10
    193/193 [==============================] - 97s 501ms/step - loss: 0.4586 - accuracy: 0.7899 - val_loss: 0.4816 - val_accuracy: 0.7799
    Epoch 8/10
    193/193 [==============================] - 95s 494ms/step - loss: 0.4538 - accuracy: 0.7953 - val_loss: 0.5022 - val_accuracy: 0.7653
    Epoch 9/10
    193/193 [==============================] - 97s 501ms/step - loss: 0.4498 - accuracy: 0.7908 - val_loss: 0.4894 - val_accuracy: 0.7566
    Epoch 10/10
    193/193 [==============================] - 97s 503ms/step - loss: 0.4435 - accuracy: 0.7968 - val_loss: 0.4822 - val_accuracy: 0.7741



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_83_1.png)
    


    Loss: 0.44350114464759827
    Val Loss: 0.48224204778671265



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_83_3.png)
    


    Accuracy: 0.796755850315094
    Val Accuracy: 0.7740525007247925


#### 5.3.1

This model is the most chaotic of all three. As we can see, it is not performing well in its current state. The model does not converge well, however it seems to generalize well. It is possible that if left training for a long enough time it will perform slightly better.

After several trial and errors while working on this model, we expanded our bidirectional layers to three. This however was not the best decision. The thought process was that adding more layers can help the model learn word contexts of the tweets, rather then making predictions on single words, such as the FFNN. Another unforseen effect was the amount of time it would take to fully train a model. Some variations lasted for up to 30 minutes. We also learned that more complexity does not mean better results. Three bidirectional layers performed worse then a model with two layers.

Out of all our experiments, this one is the worst in terms of accuracy and time taken to train. As we trained with different variations, it was noted that adding complexity does not necessarily add any benefit and may instead hinder the learning process. Despite these current results, we still think that using an RNN model as a basis may still help us.

As of writing this, the model is still a work in progess and has shown promise thus far. We will continue to work on this model to try to get the highest kaggle score possible as it still is promising.


```python
rnn_train_predictions = cnn_model.predict(X_train[:, 3]).flatten()
rnn_train_loss = calculate_loss(Y_train, rnn_train_predictions)
print(f'Train Loss: {rnn_train_loss}')

rnn_validation_predictions = cnn_model.predict(X_validation[:, 3]).flatten()
rnn_validation_loss = calculate_loss(Y_validation, rnn_validation_predictions)
print(f'Val Loss: {rnn_validation_loss}')
```

    215/215 [==============================] - 1s 4ms/step
    Train Loss: 0.43194361795525693
    24/24 [==============================] - 0s 4ms/step
    Val Loss: 0.5099057625701744



```python
plot_predictions(rnn_train_predictions, Y_train)
plot_difference(rnn_train_predictions, Y_train)
```


    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_86_0.png)
    



    
![png](Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_files/Giovanni_%26_Oscar_Twitter_Disasters_Final_Report_86_1.png)
    



```python
print_confusion_matrix(rnn_train_predictions, Y_train, 0.7)
```

    True Positives: 1613
    True Negatives: 3771
    False Positives: 134
    False Negatives: 1333
    
    Accuracy: 0.7858706758137498
    Precision: 0.9232970807097882
    Recall: 0.5475220638153429


#### 5.3.2
Here the same problem emerges, a large amount of false positves are being predicted. Interestingly however, it has done a very good job of prediciting negative values.


```python
rnn_predictions = rnn_model.predict(test_input_np[:, 3]).flatten()

# Apply threshold
THRESHOLD = 0.7
rnn_predictions[rnn_predictions >= THRESHOLD] = 1
rnn_predictions[rnn_predictions < THRESHOLD] = 0

save_to_submissions_csv(test_input_np, rnn_predictions, 'rnn_submission.csv')
```

    102/102 [==============================] - 19s 157ms/step
    Generating "rnn_submission.csv" file...
    Successfully created "rnn_submission.csv"



```python
pd.read_csv('rnn_submission.csv').head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 6. Conclusion

### 6.1 Overall Results

Overall, our models much better than the baseline, showing that they have learned something from the training process, rather than predicting 1 or 0 all the time. Below is a chart that shows the accuracy of all of our models.

> **Note:** Some of these values change due to some inner randomness of each layer (despite using a set global seed), but the values are still very close each training time.


Model | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | F1 Score
---|---|---|---|---|---
Baseline | 7.900914820187636 | 0.4308860020434973 | 7.687370260599765 | 0.41863517060367456 | 0.42966
FFNN | 0.43176037073135376 | 0.8029196858406067 | 0.5280683636665344 | 0.7434402108192444 | 0.75176
CNN | 0.4304109811782837 | 0.8103811740875244 | 0.5113658905029297 | 0.7594752311706543 | 0.74287
RNN | 0.44599589705467224 | 0.7956204414367676 | 0.5282750725746155 | 0.7448979616165161 | 0.72816

**Most General:** RNN

**Highest F1:** FFNN

**Smallest Validation Loss:** CNN

### 6.2 Future Work
After sifting through the graphs on all the models, the main hurdle we seem to struggle with the most is the amount of false negatives the models continuously miss. The root of the problem may be the use of custom made pre-processing function for this project. It is present in all models, so it may be the biggest factor that hinders the learning process. One possibility for a better pre-processing function is to break apart words into n-grams, that way links can be broken down to individual websites that may appear frequeantly (e.g.common news site).

