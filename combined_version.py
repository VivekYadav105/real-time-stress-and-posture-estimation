import cv2
import tensorflow as tf
import numpy as np
import math as m
import mediapipe as mp
from ultralytics import YOLO
import tensorflow_hub as hub

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def calculate_shoulder_angle(a,b):
    a = np.array(a)
    b = np.array(b)
    
    radians = np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (224, 224))
    # Convert the image to a format that can be input to the model
    image = np.expand_dims(image, axis=0)
    # Normalize the pixel values
    image = image / 255.0
    return image

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def plot_pose(image,landmarks):
    for landmark in landmarks:
        cv2.circle(image,(int(landmark[0]),int(landmark[1])),3,(0, 0, 255),-1)
    cv2.line(image, (int(landmarks[8][0]),int(landmarks[8][1])), (int(landmarks[6][0]),int(landmarks[6][1])), (0, 255, 0), 2)   
    cv2.line(image, (int(landmarks[6][0]),int(landmarks[6][1])), (int(landmarks[5][0]),int(landmarks[5][1])), (0, 255, 0), 2)    
    cv2.line(image, (int(landmarks[5][0]),int(landmarks[5][1])), (int(landmarks[4][0]),int(landmarks[4][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[4][0]),int(landmarks[4][1])), (int(landmarks[0][0]),int(landmarks[0][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[0][0]),int(landmarks[0][1])), (int(landmarks[1][0]),int(landmarks[1][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[1][0]),int(landmarks[1][1])), (int(landmarks[2][0]),int(landmarks[2][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[2][0]),int(landmarks[2][1])), (int(landmarks[3][0]),int(landmarks[3][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[3][0]),int(landmarks[3][1])), (int(landmarks[7][0]),int(landmarks[7][1])), (0, 255, 0), 2)  


    #mouth line
    cv2.line(image, (int(landmarks[10][0]),int(landmarks[10][1])), (int(landmarks[9][0]),int(landmarks[9][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[9][0]),int(landmarks[9][1])), (int(landmarks[10][0]),int(landmarks[10][1])), (0, 255, 0), 2)  
  
    #main torse
    cv2.line(image, (int(landmarks[12][0]),int(landmarks[12][1])), (int(landmarks[11][0]),int(landmarks[11][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[12][0]),int(landmarks[12][1])), (int(landmarks[24][0]),int(landmarks[24][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[24][0]),int(landmarks[24][1])), (int(landmarks[23][0]),int(landmarks[23][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[11][0]),int(landmarks[11][1])), (int(landmarks[23][0]),int(landmarks[23][1])), (0, 255, 0), 2)  

    #legs
    cv2.line(image, (int(landmarks[24][0]),int(landmarks[24][1])), (int(landmarks[26][0]),int(landmarks[26][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[26][0]),int(landmarks[26][1])), (int(landmarks[28][0]),int(landmarks[28][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[23][0]),int(landmarks[23][1])), (int(landmarks[25][0]),int(landmarks[25][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[25][0]),int(landmarks[25][1])), (int(landmarks[27][0]),int(landmarks[27][1])), (0, 255, 0), 2)  
    
    #foot1
    cv2.line(image, (int(landmarks[28][0]),int(landmarks[28][1])), (int(landmarks[32][0]),int(landmarks[32][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[32][0]),int(landmarks[32][1])), (int(landmarks[30][0]),int(landmarks[30][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[28][0]),int(landmarks[28][1])), (int(landmarks[30][0]),int(landmarks[30][1])), (0, 255, 0), 2)  

    #foot2
    cv2.line(image, (int(landmarks[27][0]),int(landmarks[27][1])), (int(landmarks[29][0]),int(landmarks[29][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[29][0]),int(landmarks[29][1])), (int(landmarks[31][0]),int(landmarks[31][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[31][0]),int(landmarks[31][1])), (int(landmarks[27][0]),int(landmarks[27][1])), (0, 255, 0), 2)  

    #hand1
    cv2.line(image, (int(landmarks[12][0]),int(landmarks[12][1])), (int(landmarks[14][0]),int(landmarks[14][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[14][0]),int(landmarks[14][1])), (int(landmarks[16][0]),int(landmarks[16][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[16][0]),int(landmarks[16][1])), (int(landmarks[18][0]),int(landmarks[18][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[20][0]),int(landmarks[20][1])), (int(landmarks[18][0]),int(landmarks[18][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[20][0]),int(landmarks[20][1])), (int(landmarks[16][0]),int(landmarks[16][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[22][0]),int(landmarks[22][1])), (int(landmarks[16][0]),int(landmarks[16][1])), (0, 255, 0), 2)  

    #hand1
    cv2.line(image, (int(landmarks[11][0]),int(landmarks[11][1])), (int(landmarks[13][0]),int(landmarks[13][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[13][0]),int(landmarks[13][1])), (int(landmarks[15][0]),int(landmarks[15][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[15][0]),int(landmarks[15][1])), (int(landmarks[21][0]),int(landmarks[21][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[15][0]),int(landmarks[15][1])), (int(landmarks[19][0]),int(landmarks[19][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[19][0]),int(landmarks[19][1])), (int(landmarks[17][0]),int(landmarks[17][1])), (0, 255, 0), 2)  
    cv2.line(image, (int(landmarks[15][0]),int(landmarks[15][1])), (int(landmarks[17][0]),int(landmarks[17][1])), (0, 255, 0), 2)  


def pose_estimate(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

    angle = calculate_shoulder_angle(left_shoulder,right_shoulder)
    angle1 = calculate_angle(left_shoulder,right_shoulder,right_elbow)
    angle2 = calculate_angle(right_shoulder,left_shoulder,left_elbow)
            
    
            
    shoulder_posture = angle >= 0 and angle <= 10
    left_arm_posture = angle1 >= 90 and angle1 <= 124
    right_arm_posture = angle2 >= 90 and angle2 <= 124
    if shoulder_posture and left_arm_posture and right_arm_posture:
        stage = "Good"
        suggestion = None
    if shoulder_posture and not(left_arm_posture and right_arm_posture):
        stage="Bad"
        suggestion = "adjust your arm posture"
    if not(shoulder_posture) and (left_arm_posture and right_arm_posture):
        stage = "Bad"
        suggestion = "adjust your shoulder posture"
    if not(shoulder_posture) and not(left_arm_posture and right_arm_posture):
        stage = "Bad"
        suggestion = "adjust your posture"
        
    return [stage,angle,angle1,angle2]



if __name__ == "__main__":
    # Use OpenCV to capture video from the webcam
    model_path = "stress_detector.h5"
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    yolo_face = YOLO('face_detector.pt')
    person_model = YOLO(r"yolov8n.pt")
    frame = cv2.imread(r'D:\photos\Gokarna Trip\Day 2\Day 2 _murudeshwar\DSC_0008.JPG')
    cv2.resize(frame,(640,640),frame,cv2.INTER_LINEAR)
    with mp_pose.Pose(static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        new_frame = frame
        face_results = yolo_face(frame)
        for r in face_results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame,[x1,y1],[x2,y2],(255,0,0),2)
                cropped_frame = frame[y1:y2,x1:x2]
                processed_frame = preprocess_image(cropped_frame)
                prediction = model.predict(processed_frame)
                label = "not stressed" if prediction[0][0] > 0.5 else "stressed"
                cv2.putText(new_frame, label, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        person_results = person_model(frame)
        for r in person_results:
            boxes = r.boxes
            for box in boxes:
                if box.cls[0]==0:
                    x,y,x1,y1=box.xyxy[0]
                    x,y,x1,y1=int(x), int(y), int(x1), int(y1)
                    cropped_image = frame[y:y1,x:x1]
                    cv2.rectangle(new_frame, (x,y), (x1,y1), (255,0,255),3)
                    cropped_image = np.ascontiguousarray(cropped_image)
                    cropped_image.flags.writeable = True
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                    results = pose.process(cropped_image)
                    print(results)
                    try:
                        landmarks = results.pose_landmarks.landmark
                        adjusted_landmarks = [(lm.x * (x1-x) + x, lm.y * (y1-y) + y) for lm in results.pose_landmarks.landmark]
                        plot_pose(new_frame,adjusted_landmarks)
                        [stage,angle,angle1,angle2] = pose_estimate(landmarks)
                        color = (0,255,0)
                        if stage == "Bad":
                            color = (0,0,255)
                        cv2.putText(new_frame,str(round(angle1,3)),(int(adjusted_landmarks[12][0]),int(adjusted_landmarks[12][1])),cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                        cv2.putText(new_frame,str(round(angle2,3)),(int(adjusted_landmarks[11][0]),int(adjusted_landmarks[11][1])),cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                        cv2.putText(new_frame, stage, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    except Exception as e:
                        print(e)
                        pass

    cv2.imshow('frame', new_frame)
    cv2.waitKey(0)