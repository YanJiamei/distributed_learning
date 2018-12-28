import pandas as pd
import tensorflow as tf
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
tf.logging.set_verbosity(tf.logging.INFO)
models_path = os.path.join(dir_path, 'logs/origin20div20ap20-2/')
checkpoint = os.path.join(models_path, 'checkpoint')

dataframe = pd.read_csv('./datas/origin20div20ap20.csv', index_col=0)
# print(GMM_data['0'])
dataframe = dataframe.sample(frac=1)
train_data = dataframe[0:8000]
# train_data = train_data.loc[(train_data.gmm_id==2) | (train_data.gmm_id==0) | (train_data.gmm_id==4) | (train_data.gmm_id==6)| (train_data.gmm_id==8),:]
# train_data = train_data.loc[(train_data.gmm_id==1) | (train_data.gmm_id==3) | (train_data.gmm_id==5) | (train_data.gmm_id==7)| (train_data.gmm_id==9),:]
test_data = dataframe[8000:10000]
# test_data = test_data.loc[(test_data.gmm_id==1) | (test_data.gmm_id==3) | (test_data.gmm_id==5) | (test_data.gmm_id==7)| (test_data.gmm_id==9),:]
# test_data = test_data.loc[(test_data.gmm_id==2) | (test_data.gmm_id==4) | (test_data.gmm_id==6) | (test_data.gmm_id==8)| (test_data.gmm_id==0),:]
FEATURE_NUM = 40
CLASSES = 10
# print(train_data[0:20],test_data[0:20])
feature_list = [(str(x), np.array(train_data[str(x)])) for x in range(FEATURE_NUM)]
feature_list = dict(feature_list)
dataset = tf.data.Dataset.from_tensor_slices(
    (
        feature_list,
        np.array(train_data['label'].astype(int))
    )
)
# dataset = dataset.repeat(10).batch(100)
# train_data = dataset.range()
def train_input_fn(dataset, batch_size):
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
# a = train_input_fn(dataset=dataset,batch_size=100)
my_feature_columns = []
for key,_ in feature_list.items():
    my_feature_columns.append(tf.feature_column.numeric_column(key = key))

# config = tf.ConfigProto(intra_op_parallelism_threads = 0,inter_op_parallelism_threads = 0)

# run_config = tf.ConfigProto()

classifier = tf.estimator.DNNClassifier(model_dir=models_path,batch_norm=False, feature_columns=my_feature_columns, n_classes=CLASSES, hidden_units = [100])
classifier.train(input_fn =lambda: train_input_fn(dataset=dataset, batch_size=100), steps=40000)
# classifier = tf.estimator.DNNClassifier(model_dir=)
feature_list_test = [(str(x), np.array(test_data[str(x)])) for x in range(FEATURE_NUM)]
feature_list_test = dict(feature_list_test)
dataset_test = tf.data.Dataset.from_tensor_slices(
    (
        feature_list_test,
        np.array(test_data['label'].astype(int))
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