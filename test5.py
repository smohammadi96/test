import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load MiDaS model for depth estimation
def load_midas_model():
    model_type = "DPT_Large"  # You can choose "DPT_Large" or "DPT_Hybrid"
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    model.eval()
    return model

def predict_depth(model, image):
    transform = Compose([
        Resize(384, 384),  # Resize to the input size of MiDaS
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        depth_map = model(image_tensor)
    
    depth_map = depth_map.squeeze().numpy()  # Remove batch dimension
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))  # Resize to original image dimensions
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize to [0, 1]
    depth_map *= 255  # Scale to [0, 255]
    return depth_map.astype(np.uint8)

def classify_body_shape(landmarks, depth_map):
    # Extract 3D landmark coordinates
    shoulder_left = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                               depth_map[int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * depth_map.shape[0]),
                                          int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * depth_map.shape[1])]])
    shoulder_right = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                depth_map[int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * depth_map.shape[0]),
                                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * depth_map.shape[1])]])
    hip_left = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                         depth_map[int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * depth_map.shape[0]),
                                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * depth_map.shape[1])]])
    hip_right = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                          depth_map[int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * depth_map.shape[0]),
                                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * depth_map.shape[1])]])

    # Calculate 3D distances
    shoulder_width = np.linalg.norm(shoulder_left - shoulder_right)
    hip_width = np.linalg.norm(hip_left - hip_right)

    # Using waist approximation as before
    waist_width = (shoulder_width + hip_width) / 2  # Simplified for this example

    # Classify body shape
    if shoulder_width > hip_width * 1.1 and waist_width < hip_width:
        return "Inverted Triangle (Apple Shape)"
    elif shoulder_width < hip_width * 0.9 and waist_width < shoulder_width:
        return "Triangle (Pear Shape)"
    elif shoulder_width < hip_width * 1.1 and waist_width < shoulder_width:
        return "Funnel Shape"
    else:
        return "Rectangle"

def process_image(image_path):
    # Load the MiDaS model
    midas_model = load_midas_model()

    # Load and process the image
    image = cv2.imread(image_path)
    depth_map = predict_depth(midas_model, image)

    # Initialize MediaPipe Pose
    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results_pose = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results_pose.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Classify body shape using depth information
            body_shape = classify_body_shape(results_pose.pose_landmarks.landmark, depth_map)
            cv2.putText(image, body_shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the image with body shape classification
    cv2.imshow("Body Shape Estimation", image)
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Update this path to your image
    process_image(image_path)
