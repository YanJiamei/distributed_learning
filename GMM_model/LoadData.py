import pandas as pd
import graphmodel as model
import tensorflow as tf
import numpy as np
GMM_data = pd.read_csv('./GMM_model/GMM_data/G_3M_4M_10000.csv', index_col=0)
print(GMM_data['0'])
dataset = tf.data.Dataset.from_tensor_slices(
    (
        {
            '0': np.array(GMM_data['0']),
            '1': np.array(GMM_data['1']),
            '2': np.array(GMM_data['2']),
        },
        # np.array(GMM_data['label'])
        np.append(np.zeros(10000),np.ones(10000))
    )
)
def train_input_fn(dataset, batch_size):
    dataset = dataset.shuffle(20000).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
# a = train_input_fn(dataset=dataset,batch_size=100)
my_feature_columns = []
for key in range(3):
    my_feature_columns.append(tf.feature_column.numeric_column(key = str(key)))
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, n_classes=2, hidden_units = [10,10])
classifier.train(input_fn =lambda: train_input_fn(dataset=dataset,batch_size=100), steps=10)
# input_e = dataset.shuffle(20000).batch(100)
# eval_result = classifier.evaluate
eval_result = classifier.evaluate(input_fn =lambda: train_input_fn(dataset=dataset,batch_size=1000))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))