import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

def preprocess_data(data, numbers_per_play):
    X = []
    y = []
    for entry in data:
        for game, details in entry['juegos'].items():
            if len(details['numeros']) == numbers_per_play:
                numbers = [int(n) for n in details['numeros']]
                X.append(numbers)
                y.append(numbers)
    X = np.array(X)
    y = np.array(y)
    
    if X.size == 0:
        print(f"No data found for games with {numbers_per_play} numbers.")
        return X, y
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_shape[0], activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_model(numbers_per_play, epochs=25, batch_size=64):
    with open('lottery_data.json', 'r') as f:
        lottery_data = json.load(f)
    
    X, y = preprocess_data(lottery_data, numbers_per_play)
    
    if X.size == 0:
        print(f"Skipping training for {numbers_per_play} numbers due to lack of data.")
        return
    
    model = create_model((X.shape[1],))
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save(f'lottery_model_{numbers_per_play}.h5')
    
    evaluation = model.evaluate(X, y)
    print(f"Model evaluation for {numbers_per_play} numbers: {evaluation}")
    with open(f'model_history_{numbers_per_play}.json', 'w') as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    for numbers_per_play in [3, 5, 6, 20]:
        train_model(numbers_per_play, epochs=2500, batch_size=64)
