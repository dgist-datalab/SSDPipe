import tensorflow as tf
import numpy as np
import time
import sys
from tensorflow.keras.mixed_precision import experimental as mixed_precision

class Classifier:
    def __init__(self, num_classes, feature_dim,
                train_feature_path, train_label_path,
                test_feature_path, test_label_path, lb, loop):

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.train_x = np.zeros((0, feature_dim), dtype=np.float16)
        self.train_y = np.zeros((0), dtype=np.uint32)
        self.test_x = np.zeros((0, feature_dim), dtype=np.float16)
        self.test_y = np.zeros((0), dtype=np.uint32)
        self.init_lr = 0.01
        self.lr_list = np.array([0.001, 0.0008, 0.0005, 0.0003, 0.0001])*15
        self.lr_list = self.lr_list * ((lb-(loop/2))/lb) # learning rate scheduling
        self.epoch = 1
        self.batch = 128
        jet_number = 0
        for path in train_feature_path:
            with open(path, "rb") as frp:
                jet_number += 1
                data = frp.read()
                num_data = len(data) // (2*feature_dim)
                print("the data is come from jetson", jet_number, path)
                print("number of data is", num_data)
                print("data size is", sys.getsizeof(data)/1024/1024/1024, "GB")
                self.train_x = np.vstack((self.train_x, np.frombuffer(data, dtype=np.float16).reshape(num_data, self.feature_dim)))
        for path in test_feature_path:
            with open(path, "rb") as frp:
                data = frp.read()
                num_data = len(data) // (2*feature_dim)
                self.test_x = np.vstack((self.test_x, np.frombuffer(data, dtype=np.float16).reshape(num_data, self.feature_dim)))
        for path in train_label_path:
            with open(path, "rb") as frp:
                self.train_y = np.concatenate((self.train_y, np.frombuffer(frp.read(), dtype=np.uint32)))
        for path in test_label_path:
            with open(path, "rb") as frp:
                self.test_y = np.concatenate((self.test_y, np.frombuffer(frp.read(), dtype=np.uint32)))

        print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)

    def scheduler(self, epoch, lr):
        print(lr)
        if epoch < 15:
            return self.lr_list[0]
        elif epoch >= 15 and epoch < 30:
            return self.lr_list[1]
        elif epoch >= 30 and epoch < 45:
            return self.lr_list[2]
        elif epoch >= 45 and epoch < 60:
            return self.lr_list[3]
        else:
            return self.lr_list[4]


    def train(self, parameter_path, model_path):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        regularizer = tf.keras.regularizers.l2(0.002)
        start = time.perf_counter()
        mirrored_strategy = tf.distribute.MirroredStrategy()
        #if model != "None":
        #    model = tf.keras.experimental.load_from_saved_model("save_model")

        with mirrored_strategy.scope():
            
            if model_path == None:
                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(self.feature_dim)),
                    tf.keras.layers.Dense(1000, activation=None,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer,
                        activity_regularizer=regularizer),
                    #tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Softmax(activity_regularizer=regularizer)
                ])
                optimizer = tf.keras.optimizers.Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            else:
                print("model is already exist!")
                model = tf.keras.models.load_model(model_path)
                optimizer = tf.keras.optimizers.Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            #optimizer = tf.keras.optimizers.Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            #temp_model = tf.keras.models.load_model('./stale_model')
            #layer1 = temp_model.get_layer('predictions')
            #model = tf.keras.Sequential()
            #model.add(tf.keras.layers.InputLayer(input_shape=(2048,), name='input'))
            #model.add(layer1)
            #model.add(tf.keras.layers.Softmax(activity_regularizer=regularizer))
            #model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
            print('\n\n\n\n\n\n\n\n\n\n\n')
            #for lay in model.layers:
            #    print(lay.dtype)
            model.fit(self.train_x, self.train_y, batch_size=self.batch, epochs=self.epoch, callbacks=[callback], verbose=1)

            loss, accuracy = model.evaluate(self.test_x, self.test_y, verbose=0)
            print("Loss:", loss, "Accuracy:", accuracy)

        end = time.perf_counter()

        print("Training time:", end - start)

        print(model.layers[0].get_weights()[0].shape, model.layers[0].get_weights()[1].shape)

        save_path = "./save_model"
        model.save(save_path)

        with open(parameter_path, "wb") as f:
            f.write(model.layers[0].get_weights()[0].tobytes())
            f.write(model.layers[0].get_weights()[1].tobytes())
        return save_path, accuracy

if __name__ == "__main__":
    classifier = Classifier(num_classes = 1000,
                            feature_dim = 2048,
                            train_feature_path = ["train_feature_5_0.dat"],
                            train_label_path = ["train_label_5_0.dat"],
                            test_feature_path = ["test_feature_5.dat"],
                            test_label_path = ["test_label_5.dat"],lb=1,loop=0)
    classifier.train("classifier.dat", "./stale_model")

