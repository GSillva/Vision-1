import cv2
import numpy as np
import mediapipe as mp
import os
from matplotlib import pyplot as plt
import time

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




class detectorh:

    colors=[(245,117,16),(117,245,16)]

    def prob_viz(res,actions,input,colors):
        out=input.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(out,(0,60+num*40),(int(prob*100),90+40*num),colors[num],-1)
            cv2.putText(out,actions[num],(0,85+num*40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

        return out
    
    actions=np.array(["abre","fecha"])
    n_sequences=30
    sequence_length=30


    

    model=keras.Sequential()
    model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,63)))
    model.add(LSTM(128,return_sequences=True,activation='relu',input_shape=(30,63)))
    model.add(LSTM(64,return_sequences=False,activation='relu',input_shape=(30,63)))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(actions.shape[0],activation='softmax'))

    res=[0.2,0.1]
    
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    #model.fit(x_train,y_train,epochs=2000,callbacks=[tb_callback])



    # Inicializar o módulo Holistic do MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    sequence=[]
    sentence=[]
    threshold=0.4

    # Inicializar a captura de vídeo
    cap = cv2.VideoCapture(0)

    def extracao_pontos(results):
        pose=np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        lh=np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        rh=np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

        return np.concatenate([lh])

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            
            # Flip frame horizontalmente para corresponder à visualização do espelho
            frame = cv2.flip(frame, 1)
            
            # Converta a cor de BGR para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Faça a detecção holistic
            results = holistic.process(rgb_frame)

            if results.pose_landmarks:
                # Extraia as coordenadas dos pontos-chave relevantes
                right_shoulder = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]), 
                                int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))
                left_shoulder = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]), 
                                int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
                waist = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * frame.shape[1]), 
                        int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * frame.shape[0]))
                
                # Calcular a posição média dos ombros e da cintura para estimar o peito
                chest_x = (right_shoulder[0] + left_shoulder[0] + waist[0]) // 3
                chest_y = (right_shoulder[1] + left_shoulder[1] + waist[1]) // 3

                if 0 <= chest_x < frame.shape[1] and 0 <= chest_y < frame.shape[0]:
                    # Obter a cor do pixel na posição do círculo
                    bgr_color = frame[chest_y, chest_x]
                    rgb_color=[0,0,0]
                    j=3
                    bgr2=[0,0,0]
                    
                    for i in range(3):
                        # Converter de BGR para RGB
                        j=j-1
                        rgb_color[i]= int(bgr_color[j])

                    for k in range(3):
                    
                        bgr2[k]= int(bgr_color[k])
                        
                    
                    cv2.circle(frame, (chest_x, chest_y), 50, (bgr2[0],bgr2[1],bgr2[2]), cv2.FILLED)
                
                else:
                  

                    cv2.circle(frame, (chest_x, chest_y), 50, (0,0,0), cv2.FILLED)
                

            # Desenhar a detecção holistic no frame
            #mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=None)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=None)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            key = extracao_pontos(results)
            sequence.insert(0,key)
            sequence=sequence[:30]

            if len(sequence)>30:
                sequence.pop(0)
            elif len(sequence)==30:
                res=model.predict(np.expand_dims(sequence,axis=0))[0]

    
                    
            if res[np.argmax(res)]>threshold:
                if len(sentence)>0:
                    if actions[np.argmax(res)] !=sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence)>5:
                sentence=sentence[-5:]

            frame=prob_viz(res,actions,frame,colors)
            cv2.rectangle(frame,(0,0),(640,40),(245,117,16),-1)
            cv2.putText(frame, ' '.join(sentence),(2,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            
            



            # Mostrar o frame resultante
            cv2.imshow('MediaPipe Holistic Detection', frame)
            cv2.waitKey(1) 
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

d=detectorh()
d.run()
