import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import math as m


def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def is_normalized_landmark(landmark):
    print(landmark.x,landmark.y,landmark.z)
    x, y, z = landmark.x, landmark.y, landmark.z  # Assuming a landmark object with x, y, z attributes
    is_normalized = 0 <= x <= 1 and 0 <= y <= 1 and 0 <= z <= 1
    return is_normalized

def forward_facing(landmarks):
    head_landmark = landmarks[0]
    left_shoulder_landmark = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder_landmark = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    head_aligned_with_shoulders = (left_shoulder_landmark.y < head_landmark.y < right_shoulder_landmark.y)

    return "Forward" if head_aligned_with_shoulders else "Backward"

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
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        person_model = YOLO("yolov8n.pt")
        test_image = cv2.imread('./test-image-1.jpg')
        image = cv2.resize(test_image,(640,640),interpolation = cv2.INTER_LINEAR)

        with mp_pose.Pose(static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            predicted = person_model.predict(image)
            count = 0
            for r in predicted:
                boxes=r.boxes
                for box in boxes:
                    count+=1
                    if(box.cls[0]==0):
                        x,y,x1,y1=box.xyxy[0]
                        x,y,x1,y1=int(x), int(y), int(x1), int(y1)
                        cropped_image = image[y:y1,x:x1]
                        cv2.rectangle(image, (x,y), (x1,y1), (255,0,255),3)
                        cropped_image = np.ascontiguousarray(cropped_image)
                        cropped_image.flags.writeable = True
                        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                        results = pose.process(cropped_image)
                        try:
                            landmarks = results.pose_landmarks.landmark
                            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            print(is_normalized_landmark(landmark=landmarks[0]))
                            adjusted_landmarks = [(lm.x * (x1-x) + x, lm.y * (y1-y) + y) for lm in results.pose_landmarks.landmark]
                            plot_pose(image,adjusted_landmarks)
                            [stage,angle,angle1,angle2] = pose_estimate(landmarks)
                            color = (0,255,0)
                            print(count)
                            if stage == "Bad":
                                color = (0,0,255)
                            cv2.putText(image,str(round(angle1,3)),(int(adjusted_landmarks[12][0]),int(adjusted_landmarks[12][1])),cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                            cv2.putText(image,str(round(angle2,3)),(int(adjusted_landmarks[11][0]),int(adjusted_landmarks[11][1])),cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                            cv2.putText(image, stage, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


                        except Exception as e:
                            print(e)
                            pass
        cv2.imwrite("mutiple-person.jpg",image)
        cv2.imshow('Mediapipe Feed', image)
        cv2.waitKey(0)       
    