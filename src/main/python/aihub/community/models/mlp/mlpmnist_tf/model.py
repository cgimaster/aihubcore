from aihub.core.models.aihubmodel import AIModel
import tensorflow as tf
import keras

# TODO: Save/restore models

class MLPMnistTF(AIModel):

    def __init__(self):
        self.meta['framework']='tensorflow'

    def build(self):
        # Network Parameters
        n_hidden_1 = 256  # 1st layer number of neurons
        n_hidden_2 = 256  # 2nd layer number of neurons
        n_input = 784  # MNIST data input (img shape: 28*28)
        n_classes = 10  # MNIST total classes (0-9 digits)

        # tf Graph input
        X = tf.placeholder("float", [None, n_input], name='X')
        Y = tf.placeholder("float", [None, n_classes], name='Y')

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Create model
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'], name='hidden_layer_1')
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name='hidden_layer_2')
        # Output fully connected layer with a neuron for each class
        logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='logits')
        pred = tf.nn.softmax(logits, name='output_softmax')  # Apply softmax to logits

        self.model = {
            'inputs':{
                "X": X,
            },
            'expected': {
                "Y": Y,
            },
            'outputs':{
                "logits": logits,
                'output_softmax':pred
            }
        }

        # Define loss and optimizer

    def fit(self, dataset, rebuild=False, params=None):
        (xtr, ytr), (xt, yt) = dataset.load_data()

        # Parameters
        learning_rate = 0.001
        training_epochs = 15
        batch_size = 100
        display_step = 1

        if not self.model: self.build()

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model['outputs']['logits'], labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        # Initializing the variables
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(xtr[0] / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    ytr_ohe = keras.utils.to_categorical(ytr[i*batch_size:(i+1)*batch_size,:], 10)
                    batch_x, batch_y = xtr[i*batch_size:(i+1)*batch_size,:], ytr_ohe
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={self.model['inputs']['X']: batch_x, self.model['expected']['Y']: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

            print("Optimization Finished!")

            save_path = saver.save(sess, "/tmp/mlp-mnist-tf-model.ckpt")
            print("Model saved in path: %s" % save_path)

    def predict(self, x, batch_size=1):
        saver = tf.train.Saver()
        result = None
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, "/tmp/mlp-mnist-tf-model.ckpt")
            print("Model restored.")
            # Check the values of the variables
            result = sess.run(self.model['outputs']['output_softmax'], feed_dict={self.model['inputs']['X']: [x]})
            # # Calculate accuracy
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # print("Accuracy:", accuracy.eval({X: x, Y: y}))
        return result