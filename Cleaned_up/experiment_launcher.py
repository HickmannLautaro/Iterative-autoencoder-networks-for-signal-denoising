from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import os


class MetricsLogger(tf.keras.callbacks.Callback):
    """Callback to log metrics to sacred database."""

    def __init__(self, run, name):
        super().__init__()
        self._run = run
        self.name = name

    def on_epoch_end(self, epoch, logs):
        for k, v in logs.items():
            self._run.log_scalar(k + '_' + str(self.name), float(v), step=epoch)


class DenseTranspose(keras.layers.Layer):  # Hands-on ml2 chapter 17
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")
        super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


def write_csv_as_text(history, name, _run):
    filename = "tmp/" + name + ".txt"
    with open(filename, "w") as handle:
        handle.write("accuracy, loss/n")
        for accuracy, loss in zip(history.history["accuracy"], history.history["loss"]):
            handle.write(f"{accuracy}, {loss}/n")

    _run.add_artifact(filename=filename, name=name)


ex = Experiment("autoencoder_test")
ex.observers.append(MongoObserver())
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def deep_autoencoder_generator_over_dim():
    act = tf.keras.activations.elu
    encoded_dim = 36
    # deep_input_img = keras.Input(shape=(784,))<

    stacked_encoder = Sequential([
        layers.Dense(1024, input_shape=(784,), activation=act, name='encode-1'),
        layers.Dense(529, activation=act, name='encode-2'),
        layers.Dense(256, activation=act, name='encode-3'),
        layers.Dense(encoded_dim, activation=act, name='encoded-features-4'), ], name='encoder')

    stacked_decoder = Sequential([
        layers.Dense(256, activation=act, name='decode-3'),
        layers.Dense(529, activation=act, name='decode-2'),
        layers.Dense(1024, activation=act, name='decode-1'),
        layers.Dense(784, activation='tanh', name='Output')], name='decoder')

    deep_autoencoder = Sequential([stacked_encoder, stacked_decoder], name='deep_autoencoder')

    # deep_input_img = keras.Input(shape=(784,))
    # x = layers.Dense(1024, activation=act, name='encode-1')(deep_input_img)
    # x = layers.Dense(529, activation=act, name='encode-2')(x)
    # x = layers.Dense(256, activation=act, name='encode-3')(x)
    # deep_encoder_output = layers.Dense(encoded_dim, activation=act, name='encoded-features-4')(x)
    #
    # deep_encoder = keras.Model(deep_input_img, deep_encoder_output, name='encoder')
    #
    # x = layers.Dense(256, activation=act, name='decode-3')(deep_encoder_output)
    # x = layers.Dense(529, activation=act, name='decode-2')(x)
    # x = layers.Dense(1024, activation=act, name='decode-1')(x)
    # deep_decoder_output = layers.Dense(784, activation='tanh', name='Output')(x)
    #
    # deep_autoencoder = keras.Model(deep_input_img, deep_decoder_output, name='deep_autoencoder')

    keras.utils.plot_model(deep_autoencoder, show_shapes=True, expand_nested=True, show_layer_names=True)

    return deep_autoencoder, stacked_encoder, encoded_dim


@ex.capture
def deep_autoencoder_generator_over_dim_tied():
    act = tf.keras.activations.elu
    encoded_dim = 36

    dense_1 = layers.Dense(1024, activation=act, name='encode-1')
    dense_2 = layers.Dense(529, activation=act, name='encode-2')
    dense_3 = layers.Dense(256, activation=act, name='encode-3')
    dense_4 = layers.Dense(encoded_dim, activation=act, name='encoded-features-4')

    tied_encoder = Sequential([keras.Input(shape=(784,)), dense_1, dense_2, dense_3, dense_4, ], name='encoder')

    tied_decoder = Sequential([
        DenseTranspose(dense_4, activation=act),
        DenseTranspose(dense_3, activation=act),
        DenseTranspose(dense_2, activation=act),
        DenseTranspose(dense_1, activation=act)], name='decoder')

    tied_autoencoder = Sequential([tied_encoder, tied_decoder], name='deep_autoencoder')

    keras.utils.plot_model(tied_autoencoder, show_shapes=True, expand_nested=True, show_layer_names=True)

    return tied_autoencoder, tied_encoder, encoded_dim


@ex.capture
def deep_autoencoder_generator_normal_dim():
    act = tf.keras.activations.elu
    encoded_dim = 30
    # deep_input_img = keras.Input(shape=(784,))

    stacked_encoder = Sequential([
        layers.Dense(100, input_shape=(784,), activation=act, name='encode-1'),
        layers.Dense(encoded_dim, activation=act, name='encoded-features-2'), ], name='encoder')

    stacked_decoder = Sequential([
        layers.Dense(100, activation=act, name='decode-'),
        layers.Dense(784, activation='tanh', name='Output')], name='decoder')

    deep_autoencoder = Sequential([stacked_encoder, stacked_decoder], name='deep_autoencoder')

    keras.utils.plot_model(deep_autoencoder, show_shapes=True, expand_nested=True, show_layer_names=True)

    return deep_autoencoder, stacked_encoder, encoded_dim


@ex.capture
def deep_autoencoder_generator_normal_dim_tied():
    act = tf.keras.activations.elu
    encoded_dim = 30

    dense_1 = layers.Dense(100, activation=act, name='encode-1')
    dense_2 = layers.Dense(encoded_dim, activation=act, name='encoded-features-2')

    tied_encoder = Sequential([keras.Input(shape=(784,)), dense_1, dense_2], name='encoder')

    tied_decoder = Sequential([
        DenseTranspose(dense_2, activation=act),
        DenseTranspose(dense_1, activation=act)], name='decoder')

    tied_autoencoder = Sequential([tied_encoder, tied_decoder], name='deep_autoencoder')

    keras.utils.plot_model(tied_autoencoder, show_shapes=True, expand_nested=True, show_layer_names=True)

    return tied_autoencoder, tied_encoder, encoded_dim


@ex.capture
def generate_classifiers(encoded_dim):
    act = tf.keras.activations.relu
    soft = tf.keras.activations.softmax

    feature_classifier = Sequential([
        layers.Dense(64, input_shape=(encoded_dim,), activation='relu'),
        layers.Dense(32, activation=act),
        layers.Dense(10, activation=soft)], name='encoder')

    in_classifier = Sequential([
        layers.Dense(64, input_shape=(784,), activation='relu'),
        layers.Dense(32, activation=act),
        layers.Dense(10, activation=soft)], name='input_classifier')

    out_classifier = Sequential([
        layers.Dense(64, input_shape=(784,), activation='relu'),
        layers.Dense(32, activation=act),
        layers.Dense(10, activation=soft)], name='output_classifier')

    # deep_encoded_img = keras.Input(shape=(encoded_dim,))
    # x = layers.Dense(64, activation='relu')(deep_encoded_img)
    # x = layers.Dense(32, activation='relu')(x)
    # classifier_output = layers.Dense(10, activation='softmax')(x)
    #
    # feature_classifier = keras.Model(deep_encoded_img, classifier_output, name='feature_classifier')
    #
    # input_img = keras.Input(shape=(784,))
    # y = layers.Dense(64, activation='relu')(input_img)
    # y = layers.Dense(32, activation='relu')(y)
    # classifier_output_2 = layers.Dense(10, activation='softmax')(y)
    #
    # in_classifier = keras.Model(input_img, classifier_output_2, name='input_classifier')
    # out_classifier = keras.Model(input_img, classifier_output_2, name='output_classifier')

    print(feature_classifier.summary())
    print(in_classifier.summary())
    return feature_classifier, in_classifier, out_classifier

  # for data in signal_data:
  # test_network(autoencoder, encoder, in_classifier, feature_classifier, out_classifier, data, test_data,  test_labels, index, _run)
  #       index = index + 10
@ex.capture
def test_network(autoencoder, encoder, in_classifier, feature_classifier, out_classifier, signal, target, target_class,
                 index, _run):

    predictions = [signal]
    evaluations_autoencoder = [autoencoder.evaluate(signal, target, verbose=0)]

    predicted_output_encoder = [encoder.predict(signal)]

    predicted_in_classifier = [in_classifier.predict(signal)]
    evaluations_in_classifier = [in_classifier.evaluate(signal, target_class, verbose=0)]

    predicted_feature_classifier = [feature_classifier.predict(predicted_output_encoder[0])]
    evaluations_feature_classifier = [feature_classifier.evaluate(predicted_output_encoder[0], target_class, verbose=0)]

    predicted_out_classifier = []
    evaluations_out_classifier = []

    print("testing loop")

    for i in range(0, 7):
        data = autoencoder.predict(predictions[i])

        predicted_out_classifier.append(out_classifier.predict(data))
        evaluations_out_classifier.append(out_classifier.evaluate(data, target_class, verbose=0))

        ### next round

        predictions.append(data)
        evaluations_autoencoder.append(autoencoder.evaluate(data, target, verbose=0))

        predicted_in_classifier.append(in_classifier.predict(data))
        evaluations_in_classifier.append(in_classifier.evaluate(data, target_class, verbose=0))


        predicted_output_encoder.append(encoder.predict(data))


        predicted_feature_classifier.append(feature_classifier.predict(predicted_output_encoder[-1]))
        evaluations_feature_classifier.append(feature_classifier.evaluate(predicted_output_encoder[-1], target_class, verbose=0))

    data = autoencoder.predict(predictions[7])

    predicted_out_classifier.append(out_classifier.predict(data))
    evaluations_out_classifier.append(out_classifier.evaluate(data, target_class, verbose=0))



    predictions_df = pd.DataFrame({'predictions': predictions,
                                   'evaluations_autoencoder': evaluations_autoencoder,
                                   'predicted_output_encoder': predicted_output_encoder,
                                   'predicted_in_classifier': predicted_in_classifier,
                                   'evaluations_in_classifier': evaluations_in_classifier,
                                   'predicted_feature_classifier': predicted_feature_classifier,
                                   'evaluations_feature_classifier': evaluations_feature_classifier,
                                   'predicted_out_classifier': predicted_out_classifier,
                                   'evaluations_out_classifier': evaluations_out_classifier})

    filename = "tmp/predictions_df_" + str(index) + ".pickle"
    predictions_df.to_pickle(filename, compression='gzip')
    _run.add_artifact(filename, name="predictions_df_" + str(index))


def make_predictions(signal, autoencoder, ):
    predicted_output = [autoencoder.predict(signal)]
    evaluations = [autoencoder.evaluate(signal)]

    for i in range(0, 6):
        predicted_output.append(autoencoder.predict(predicted_output[i]))
        evaluations.append(autoencoder.evaluate(predicted_output[i]))

    return np.array([predicted_output]), np.array([evaluations])


@ex.config
def my_config():
    batch_size = 256
    epochs = 250
    autoencoder_type = 'Over_dim'
    targets_type = 'Mnist'


@ex.command
def experiment(batch_size, epochs, autoencoder_type, targets_type, _run):
    tf.keras.backend.clear_session()  # Otherwise memory of gpu is overused

    save_special = False
    iteration = False

    loaded = np.load('data.npz')
    train_labels = loaded['train_labels_cat']
    test_labels = loaded['test_labels_cat']
    signal_data = loaded['noisy_data']
    train_data  = loaded['x_train']
    test_data = loaded['x_test']
    eval_data=loaded['x_test']

    if targets_type == 'Mnist':
        train_data_targets = loaded['x_train']
        test_data_targets = loaded['x_test']


    elif targets_type == '10_Targets':
        train_data_targets = loaded['x_train_targets']
        test_data_targets = loaded['x_test_targets']

    elif targets_type == 'Noisy':
        train_data = loaded['x_train_noisy']
        test_data = loaded['x_test_noisy']
        train_data_targets = loaded['x_train']
        test_data_targets = loaded['x_test']

    # Generate and train autoencoder and classifiers
    if autoencoder_type == 'Over_dim':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_over_dim()
    elif autoencoder_type == 'Over_dim_iteration':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_over_dim()
    elif autoencoder_type == 'Over_dim_tied':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_over_dim_tied()
        save_special = True


    elif autoencoder_type == 'Over_dim_tied_iteration':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_over_dim_tied()
        iteration = True
        save_special = True
    elif autoencoder_type == 'normal_dim':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_normal_dim()
    elif autoencoder_type == 'normal_dim_iteration':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_normal_dim()
        iteration = True

    elif autoencoder_type == 'nomal_dim_tied':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_normal_dim_tied()
        save_special = True
    elif autoencoder_type == 'nomal_dim_tied_iteration':
        autoencoder, encoder, encoded_dim = deep_autoencoder_generator_normal_dim_tied()
        iteration = True

        save_special = True
    else:
        print("Error")

    metrics_logger = MetricsLogger(_run, "autoencoder")

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse,
                        metrics=['accuracy', tf.keras.metrics.mae])

    encoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse,
                    metrics=['accuracy', tf.keras.metrics.mae])

    history_autoencoder = autoencoder.fit(train_data, train_data_targets,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          validation_data=(test_data, test_data_targets),
                                          verbose=1, callbacks=[metrics_logger,
                                                                keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                                                    patience=30,
                                                                                                    mode='auto',
                                                                                                    restore_best_weights=True)])
    if iteration:
        predicted_output = autoencoder.predict(train_data)
        predicted_test = autoencoder.predict(test_data)

        history_autoencoder_2 = autoencoder.fit(predicted_output, train_data_targets,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                validation_data=(predicted_test, test_data_targets),
                                                verbose=1, callbacks=[metrics_logger,
                                                                      keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                                                    patience=30,
                                                                                                    mode='auto',
                                                                                                    restore_best_weights=True)])
        write_csv_as_text(history_autoencoder_2, "history_autoencoder_iteration", _run)

    write_csv_as_text(history_autoencoder, "history_autoencoder", _run)

    # To save space dont save the model & error with tied weighs
    filename_autoencoder = "tmp/autoencoder.hdf5"
    if save_special:
        # TODO produces error
        #autoencoder.save(filename_autoencoder,save_format='tf')
        a=2
    else:
        autoencoder.save(filename_autoencoder)
        _run.add_artifact(filename_autoencoder)

    encoded_input = encoder.predict(train_data)
    encoded_test = encoder.predict(test_data)

    predicted_output = autoencoder.predict(train_data)

    predicted_test = autoencoder.predict(test_data)

    metrics_logger = MetricsLogger(_run, "feature_classifier")

    feature_classifier, in_classifier, out_classifier = generate_classifiers(encoded_dim)

    feature_classifier.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.CategoricalCrossentropy(),
                               metrics=['accuracy'])
    history_feature_classifier = feature_classifier.fit(encoded_input, train_labels,
                                                        epochs=epochs,
                                                        validation_data=(encoded_test, test_labels),
                                                        verbose=1, callbacks=[metrics_logger,
                                                                              keras.callbacks.EarlyStopping(
                                                                                  monitor='val_accuracy', patience=10,
                                                                                  mode='auto',
                                                                                  restore_best_weights=True)])

    metrics_logger = MetricsLogger(_run, "in_classifier")

    in_classifier.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.CategoricalCrossentropy(),
                          metrics=['accuracy'])
    history_in_classifier = in_classifier.fit(train_data, train_labels,
                                              epochs=epochs,
                                              validation_data=(test_data, test_labels),
                                              verbose=1, callbacks=[metrics_logger,
                                                                    keras.callbacks.EarlyStopping(
                                                                        monitor='val_accuracy', patience=10,
                                                                        mode='auto',
                                                                        restore_best_weights=True)])
    metrics_logger = MetricsLogger(_run, "out_classifier")

    out_classifier.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
    history_out_classifier = out_classifier.fit(predicted_output, train_labels,
                                                epochs=epochs,
                                                validation_data=(predicted_test, test_labels),
                                                verbose=1, callbacks=[metrics_logger,
                                                                      keras.callbacks.EarlyStopping(
                                                                          monitor='val_accuracy', patience=10,
                                                                          mode='auto',
                                                                          restore_best_weights=True)])

    write_csv_as_text(history_feature_classifier, "history_feature_classifier", _run)
    write_csv_as_text(history_in_classifier, "history_in_classifier", _run)
    write_csv_as_text(history_out_classifier, "history_out_classifier", _run)

    index = 0
    for data in signal_data:
        test_network(autoencoder, encoder, in_classifier, feature_classifier, out_classifier, data, eval_data,
                     test_labels, index, _run)
        index = index + 10


if __name__ == "__main__":
    # @ex.automain
    # def launcher(batch_size, epochs, autoencoder_type, targets_type):
    dirName = 'tmp'

    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    # ex.run("experiment", config_updates={"batch_size": batch_size, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})

    epochs = 250

    # ##### Normal
    # autoencoder_type = 'Over_dim'
    #
    #
    # targets_type = 'Noisy'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})


    # targets_type = '10_Targets'
    # #ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    #
    #
    # targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    autoencoder_type = 'normal_dim'

    targets_type = 'Noisy'
    ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    #
    # targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    # targets_type = '10_Targets'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})

    #### 2 -Iterations
    autoencoder_type = 'Over_dim_iteration'

    targets_type = 'Noisy'
    ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})

    #targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                     "targets_type": targets_type})
    #ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                    "targets_type": targets_type})

    # targets_type = '10_Targets'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})

    autoencoder_type = 'normal_dim_iteration'

    targets_type = 'Noisy'
    ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    #
    # targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,"targets_type": targets_type})
    #
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    #targets_type = '10_Targets'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    ### tied_weights
    #
    autoencoder_type = 'Over_dim_tied'
    targets_type = 'Noisy'
    ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    #
    # targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    #targets_type = '10_Targets'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    autoencoder_type = 'nomal_dim_tied'

    targets_type = 'Noisy'
    ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    #
    # targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    # targets_type = '10_Targets'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                       "targets_type": targets_type})
    #
    autoencoder_type = 'nomal_dim_tied_iteration'

    targets_type = 'Noisy'
    ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    #
    # targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    # targets_type = '10_Targets'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})

    autoencoder_type = 'Over_dim_tied_iteration'

    targets_type = 'Noisy'
    ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
                                         "targets_type": targets_type})
    #
    # targets_type = 'Mnist'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    #
    # targets_type = '10_Targets'
    # ex.run("experiment", config_updates={"batch_size": 256, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 128, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 64, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
    # ex.run("experiment", config_updates={"batch_size": 32, "epochs": epochs, "autoencoder_type": autoencoder_type,
    #                                      "targets_type": targets_type})
