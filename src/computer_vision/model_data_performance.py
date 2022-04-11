import tensorflow as tf


class DataPerformanceConfiguration:

    def __init__(self):
        self.tune = tf.data.AUTOTUNE

    def configure_data_performance(self, train_ds, val_ds):
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=self.tune)
        val_ds = val_ds.cache().prefetch(buffer_size=self.tune)
        return {"train": train_ds, "validate": val_ds}
