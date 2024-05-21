import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import keras


actions=np.array(["abre","fecha"])
n_sequences=30
sequence_length=30


label_map={label: num for num,label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(n_sequences):
       window=[]

       for frame_num in range(n_sequences):
           res=np.load(os.path.join("data_test",action,str(sequence),"{}.npy".format(frame_num)))
           window.append(res)
       sequences.append(window)
       labels.append(label_map[action])

x=np.array(sequences)
y=to_categorical(labels).astype(int)

x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.5)

log_dir=os.path.join("Logs")
tb_callback=TensorBoard(log_dir=log_dir)

model=keras.Sequential()
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,63)))
model.add(LSTM(128,return_sequences=True,activation='relu',input_shape=(30,63)))
model.add(LSTM(64,return_sequences=False,activation='relu',input_shape=(30,63)))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(actions.shape[0],activation='softmax'))

res=[.7,0.2,0.1]
actions[np.argmax(res)]
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.fit(x_train,y_train,epochs=2000,callbacks=[tb_callback])

model.save('action.h5')
#del model
model.load_weights('action.h5')