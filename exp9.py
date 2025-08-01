import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense 
# Step 1: Load Dataset 
vocab_size = 10000 
max_len = 200 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size) 
# Step 2: Pad Sequences 
x_train = pad_sequences(x_train, maxlen=max_len) 
x_test = pad_sequences(x_test, maxlen=max_len) 
# Step 3: Build RNN Model 
model = Sequential() 
model.add(Embedding(vocab_size, 32, input_length=max_len)) 
model.add(SimpleRNN(64)) 
model.add(Dense(1, activation='sigmoid')) 
# Step 4: Compile and Train 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1) 
# Step 5: Evaluate 
loss, accuracy = model.evaluate(x_test, y_test, verbose=0) 
print(f"Test Accuracy: {accuracy:.2f}") 