import time
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping

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
        best_epoch_idx = min(
            range(len(history.history['val_loss'])),
            key=lambda i: history.history['val_loss'][i],
        )
        metrics = {
            'epoch': best_epoch_idx + 1,
            'train_loss': float(history.history['loss'][best_epoch_idx]),
            'val_loss': float(history.history['val_loss'][best_epoch_idx]),
            'training_time': time.time() - start_time,
        }
        return history, metrics
    return history
