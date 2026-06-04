import time
import h5py
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping


def save_training_data(x, y, path):
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.create_dataset('x', data=x, dtype='f8')
        data.create_dataset('y', data=y, dtype='f8')


def load_training_data(path):
    with h5py.File(path, 'r') as f:
        return f['data']['x'][...], f['data']['y'][...]


def save_history(history, path):
    pd.DataFrame(history).to_csv(path, index=False)


def train_model(model, inputs, targets, loss, learning_rate, n_epochs,
                batch_size, validation_split, patience, return_metrics=False):
    start_time = time.time() if return_metrics else None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        jit_compile=True,
    )

    history = model.fit(
        inputs,
        targets,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience,
                                 restore_best_weights=True, verbose=1)],
    )

    if return_metrics:
        metrics = {
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'training_time': time.time() - start_time,
        }
        return history, metrics
    return history