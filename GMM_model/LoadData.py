import pandas as pd
import graphmodel as model
import tensorflow as tf
import numpy as np
dataframe = pd.read_csv('./GMM_model/GMM_data/G_10M_4M_1000.csv', index_col=0)
# print(GMM_data['0'])
dataframe = dataframe.sample(frac=1)
train_data = dataframe[0:8000]
test_data = dataframe[8000:10000]
# print(train_data[0:20],test_data[0:20])
dataset = tf.data.Dataset.from_tensor_slices(
    (
        {
            '0': np.array(train_data['0']),
            '1': np.array(train_data['1']),
            '2': np.array(train_data['2']),
            '3': np.array(train_data['3']),
            '4': np.array(train_data['4']),
            '5': np.array(train_data['5']),
            '6': np.array(train_data['6']),
            '7': np.array(train_data['7']),
            '8': np.array(train_data['8']),
            '9': np.array(train_data['9']),
        },
        np.array(train_data['label'])
    )
)
# dataset = dataset.repeat(10).batch(100)
# train_data = dataset.range()
def train_input_fn(dataset, batch_size):
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
# a = train_input_fn(dataset=dataset,batch_size=100)
my_feature_columns = []
for key in range(100):
    my_feature_columns.append(tf.feature_column.numeric_column(key = str(key)))
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, n_classes=10, hidden_units = [10,10,10])
classifier.train(input_fn =lambda: train_input_fn(dataset=dataset, batch_size=100), steps=20000)
dataset_test = tf.data.Dataset.from_tensor_slices(
    (
        {
            '0': np.array(test_data['0']),
            '1': np.array(test_data['1']),
            '2': np.array(test_data['2']),
            '3': np.array(test_data['3']),
            '4': np.array(test_data['4']),
            '5': np.array(test_data['5']),
            '6': np.array(test_data['6']),
            '7': np.array(test_data['7']),
            '8': np.array(test_data['8']),
            '9': np.array(test_data['9']),
        },
        np.array(test_data['label'])
    )
)
# dataset_test = dataset_test.batch(100)
# train_data = dataset.range()
def test_input_fn(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
# input_e = dataset.shuffle(20000).batch(100)
# eval_result = classifier.evaluate
eval_result = classifier.evaluate(input_fn =lambda: test_input_fn(dataset=dataset_test, batch_size=100),steps=1000)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))