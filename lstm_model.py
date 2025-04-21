import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Layer
from sklearn.preprocessing import MinMaxScaler

class InverseTransform(Layer):
    def __init__(self, scaler, feature_index, n_features, **kwargs):
        super().__init__(**kwargs)
        self.scaler = scaler
        self.feature_index = feature_index
        self.n_features = n_features

    def call(self, inputs, **kwargs):
        # batch size
        batch_size = tf.shape(inputs)[0]
        # создаём нулевой тензор формы (batch_size, n_features)
        dummy = tf.zeros((batch_size, self.n_features), dtype=tf.float32)
        # извлекаем предсказания в виде вектора shape=(batch_size,)
        updates_flat = tf.squeeze(inputs, axis=1)
        # индексы (batch_index, feature_index) для scatter
        indices = tf.stack([
            tf.range(batch_size, dtype=tf.int32),
            tf.fill([batch_size], tf.constant(self.feature_index, dtype=tf.int32))
        ], axis=1)  # shape=(batch_size, 2)
        # разброс обновлений по dummy
        dummy = tf.tensor_scatter_nd_update(dummy, indices, updates_flat)
        # обратное преобразование через numpy scaler
        inverted = self.scaler.inverse_transform(dummy.numpy())
        # возвращаем только колонку target
        return inverted[:, self.feature_index]

def prepare_lstm_data(data, lookback, features):
    """
    Подготавливает скользящее окно X и целевой вектор y.
    Возвращает X shape=(n_samples, lookback, n_features),
    y shape=(n_samples,), и сам scaler.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[features].values)
    X, y = [], []
    target_idx = features.index('Close')
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i][target_idx])
    return np.array(X), np.array(y), scaler

def build_lstm_model(lookback, n_features):
    """
    Возвращает компилированную Sequential модель LSTM.
    """
    model = Sequential([
        tf.keras.layers.Input(shape=(lookback, n_features)),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(data, config, features):
    """
    Обучает LSTM на данных и возвращает (model, scaler).
    """
    lookback = config['lstm']['lookback']
    X, y, scaler = prepare_lstm_data(data, lookback, features)
    model = build_lstm_model(lookback, len(features))
    model.fit(X, y,
              epochs=config['lstm']['epochs'],
              batch_size=config['lstm']['batch_size'],
              verbose=1)
    return model, scaler

def predict_lstm(model, data, scaler, config, features):
    """
    Делает прогнозы, а затем корректно обратное масштабирование
    только для колонки 'Close'.
    """
    lookback = config['lstm']['lookback']
    X, _, _ = prepare_lstm_data(data, lookback, features)
    # предсказания в нормализованном виде shape=(n_samples, 1)
    preds_scaled = model.predict(X)
    # индекс таргет-колонки
    target_idx = features.index('Close')
    # число признаков
    n_features = len(features)
    # слой для обратного преобразования
    inverse_layer = InverseTransform(
        scaler=scaler,
        feature_index=target_idx,
        n_features=n_features
    )
    # получаем массив shape=(n_samples,)
    preds = inverse_layer(preds_scaled)
    return preds
