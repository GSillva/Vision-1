import cv2
import mediapipe as mp

class detectorh:

    # Inicializar o módulo Holistic do MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Inicializar a captura de vídeo
    cap = cv2.VideoCapture(0)

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
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=None)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=None)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Mostrar o frame resultante
            cv2.imshow('MediaPipe Holistic Detection', frame)
            cv2.waitKey(1) 
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

d=detectorh()
d.run()
