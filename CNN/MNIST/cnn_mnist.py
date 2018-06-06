import tensorflow as tf
import numpy as np

#Logging level changed to INFO
from tensorflow.contrib.keras.api.keras import activations

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features,labels,mode):
    #Input feature vector
    input_vector = tf.reshape(features["x"],[-1,28,28,1])

    #Connvolution layer 1
    conv1 = tf.layers.conv2d(
        inputs = input_vector,
        filters = 32,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu)

    #Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)

    #Convolution layer 2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu)

    #Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)

    #Dense Layer
    pool2_flat = tf.reshape(pool2,[-1,7 * 7 *64])
    dense = tf.layers.dense(inputs = pool2_flat, units =1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs = dense, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    #Logits Layer
    logits = tf.layers.dense(inputs= dropout, units = 10)

    predictions= {
        #Generate predictions on the PREDICT and EVAL Mode
        "classes" : tf.argmax(input= logits, axis = 1),
        #Add Softmax tensor to the graph
        "probablilites" : tf.nn.softmax(logits, name ="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictons= predictions )

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    #Load MNIST data from the tensorflow repository
    mnistData = tf.contrib.learn.datasets.load_dataset("mnist")
    #Train data
    trainData = mnistData.train.images
    trainLabels = np.asarray(mnistData.train.labels, dtype= np.int32)
    #Evaluation Data
    evalData = mnistData.test.images
    evalLabels = np.asarray(mnistData.test.labels, dtype=np.int32)
    #Create estimator
    mnistClassifier = tf.estimator.Estimator( model_fn=cnn_model_fn, model_dir ='C:\\Users\\ankit\\PycharmProjects\\CNN_Tensorflow\\convnet_model')
    #Logging Hooks
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": trainData},
        y=trainLabels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnistClassifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": evalData},
        y=evalLabels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnistClassifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()