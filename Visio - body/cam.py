import cv2
import numpy as np
import mediapipe as mp
import os
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH= os.path.join("data_test")
actions=np.array(["abre","fecha"])
n_sequences=30
sequence_length=30
for action in actions:
                   for sequence in range(n_sequences):
                        try:
                            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
                        except:
                            pass

 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

    # Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

def extracao_pontos(results):
        pose=np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        lh=np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        rh=np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

        return np.concatenate([lh])

#def mediapipe_detection(image,model):
  #     image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   #    image.flags.writeable=False
    #   resul=model.process(image)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        for action in actions:
            for sequence in range(n_sequences):
                   for frame_n in range(sequence_length):
                        ret, frame = cap.read()

                        
                        
                        # Flip frame horizontalmente para corresponder à visualização do espelho
                        frame = cv2.flip(frame, 1)

                        image = holistic.process(frame)  
                        
                        # Converta a cor de BGR para RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Faça a detecção holistic
                        results = holistic.process(rgb_frame)  
                                  

                        # Desenhar a detecção holistic no frame
                        #mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None)
                        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=None)
                        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=None)
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        
                        if frame_n == 0:
                               cv2.putText(frame,"START",(120,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                               cv2.putText(frame,"Frame: {} Video: {}".format(action,sequence),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                               cv2.imshow("Imagem",frame)
                               cv2.waitKey(200)
                        else:
                               cv2.putText(frame,"Frame: {} Video: {}".format(action,sequence),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                               cv2.imshow("Imagem",frame)


                        keypoints=extracao_pontos(results)
                        npy_path=os.path.join(DATA_PATH,action,str(sequence),str(frame_n))
                        np.save(npy_path,keypoints)


                        # Mostrar o frame resultante
                        
                        if 0xFF==ord("q") & cv2.waitKey(1):
                               break
    # Liberar recursos
cap.release()
cv2.destroyAllWindows()
