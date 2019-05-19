import os
import sys

import tensorflow as tf

import model
from datasets import data_generator

flags = tf.app.flags
FLAGS = flags.FLAGS

# dataset
flags.DEFINE_string('dataset', 'pp_detect', 'Name of the dataset.')
flags.DEFINE_string('train_split', 'train', 'Which split of the dataset to be used for training')
flags.DEFINE_string('test_split', 'val', 'Which split of the dataset to be used for testing')
flags.DEFINE_string('dataset_dir', './datasets/LHand/tfrecord',
                    'Where the dataset reside.')

# log and model
flags.DEFINE_string('model_dir', '/home/yjy/models_pretrain/fingerRoot/LHand',
                    'Where the model to be stored and loaded.')
flags.DEFINE_string('model_name', None, 'file name of model.')  # 'Finger_roots.h5'

# preprocess
flags.DEFINE_float('min_scale_factor', 0.7,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 1.1,
                   'Maximum scale factor for data augmentation.')
flags.DEFINE_multi_integer('train_crop_size', [480, 640],
                           'Image crop size [height, width] during training.')

# learning
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')
flags.DEFINE_float('base_learning_rate', 0.001,
                   'The base learning rate for model training.')
flags.DEFINE_integer('training_number_of_steps', 24000,
                     'The number of steps used for training')
flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                   'The rate to decay the base learning rate.')

# useless flags below:
flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')
flags.DEFINE_integer('learning_rate_decay_step', 400,
                     'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')


def build_model():
    m_max = 0
    m_model_name = FLAGS.model_name
    learning_rate = FLAGS.base_learning_rate

    m_model = model.get_model()
    '''
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    '''
    m_optimizer = tf.keras.optimizers.Adam()  # lr=learning_rate

    try:
        if FLAGS.model_name is None:
            for x in os.listdir(FLAGS.model_dir):
                if m_max < int(x[8:10]):
                    m_model_name = x
                    m_max = int(x[8:10])
        # m_model = tf.keras.models.load_model(filepath=os.path.join(FLAGS.model_dir, m_model_name))
        m_model.load_weights(filepath=os.path.join(FLAGS.model_dir, m_model_name))

    except tf.errors.NotFoundError as e:
        print(e)
    except TypeError as e:
        print(e)
    except OSError as e:
        print(e)
    except UnboundLocalError as e:
        print(e)

    m_model.compile(
        optimizer=m_optimizer,
        loss='mean_squared_error',  # tf.keras.losses.CategoricalCrossentropy()
        metrics=['mean_squared_error']  # tf.keras.metrics.Accuracy()
    )
    return m_model, m_max


def train(m_model, iterator, spe, epoch, initial_epoch):
    m_callbacks = []

    def lr_adjust_callback(epoch_index):
        m_lr = FLAGS.base_learning_rate / (1 + FLAGS.learning_rate_decay_factor * epoch_index * spe)
        tf.logging.info('epoch index=%d\n', epoch_index)
        tf.logging.info('learning rate=%d\n', m_lr)
        return m_lr

    # m_callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_adjust_callback, 1))
    m_callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_dir + '/weights.{epoch:02d}-{loss:.2f}.hdf5',
                                           save_weights_only=True, verbose=1))
    # m_callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(logs['loss'])))
    m_model.fit(x=iterator, epochs=epoch, callbacks=m_callbacks, steps_per_epoch=spe)
    # m_model.fit(x=iterator, epochs=epoch, callbacks=m_callbacks, initial_epoch=initial_epoch, steps_per_epoch=spe)
    # for i in range(epoch):
    #     m_model.fit(x=iterator, epochs=1, callbacks=m_callbacks, steps_per_epoch=spe)


def eval_and_predict(m_model, iterator):
    test_loss, test_acc = m_model.evaluate(iterator, steps=200)
    print('Test accuracy: ', test_acc)

    # predict
    '''
    test_images = None
    predict_result = m_model.predict(x=test_images, batch_size=batch_size)
    classes = tf.Session().run(tf.argmax(predict_result, 1))
    print(classes, len(classes))
    '''


def main(_):
    clone_batch_size = 8
    # steps_per_epoch = int(1800 / clone_batch_size)
    # m_epoch = int((FLAGS.training_number_of_steps+1800) / 1800)
    steps_per_epoch = int(3000 / clone_batch_size * 60 * 1)
    m_epoch = 1

    train_dataset = data_generator.Dataset(
        dataset_name=FLAGS.dataset,
        split_name=FLAGS.train_split,
        dataset_dir=FLAGS.dataset_dir,
        batch_size=clone_batch_size,
        crop_size=FLAGS.train_crop_size,
        min_scale_factor=FLAGS.min_scale_factor,
        max_scale_factor=FLAGS.max_scale_factor,
        num_readers=2,
        is_training=True,
        should_shuffle=True,
        should_repeat=True)
    train_iterator = train_dataset.get_one_shot_iterator()

    test_dataset = data_generator.Dataset(
        dataset_name=FLAGS.dataset,
        split_name=FLAGS.test_split,
        dataset_dir=FLAGS.dataset_dir,
        batch_size=clone_batch_size,
        crop_size=FLAGS.train_crop_size,
        min_scale_factor=FLAGS.min_scale_factor,
        max_scale_factor=FLAGS.max_scale_factor,
        num_readers=2,
        is_training=False,
        should_shuffle=True,
        should_repeat=True)
    test_iterator = test_dataset.get_one_shot_iterator()

    # build model
    my_model, init_e = build_model()

    # train
    train(my_model, train_iterator, steps_per_epoch, m_epoch, initial_epoch=init_e)

    # evaluate and predict
    eval_and_predict(my_model, test_iterator)

    # summary and save
    my_model.summary()
    # my_model.save(os.path.join(FLAGS.model_dir, 'Finger_roots.hdf5'))


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.app.run(main=main, argv=[sys.argv[0]])
