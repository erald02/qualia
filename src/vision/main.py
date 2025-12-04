import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class VisionDetector:
    def __init__(self, model_name='yolov8s.pt', conf_threshold=0.2):
        """
        Initialize the YOLO detector with tracking capabilities.
        
        Args:
            model_name (str): 'yolov8s.pt' recommended.
            conf_threshold (float): Lowered to 0.4 to catch 'hamster-like' animals.
        """
        print(f"Loading Model: {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
        # We accept Cat (15), Dog (16), Bear (21), Sheep (18), Cow (19), Horse (17)
        # Removed "Person" (0) to focus on animals, add back if needed.
        self.target_classes = [15, 16, 17, 18, 19, 21] 
        
        # Store previous center points to calculate speed: {track_id: (x, y)}
        self.track_history = defaultdict(lambda: None)

    def detect(self, frame):
        """
        Run tracking on a frame and extract behavioral metrics.
        """
        # "persist=True" tells YOLO to remember objects between frames for tracking
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, 
                                 classes=self.target_classes, verbose=False)
        
        parsed_results = []
        result = results[0]
        
        # Get frame dimensions for normalization
        height, width = frame.shape[:2]
        
        # Visual debug frame
        annotated_frame = result.plot()
        
        if result.boxes.id is not None:
            # result.boxes.id is a tensor of IDs (e.g., [1, 2, 5])
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                
                # 1. Calculate Center (for position)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 2. Calculate Size (Presence) - Normalized 0.0 to 1.0
                area = (x2 - x1) * (y2 - y1)
                frame_area = width * height
                presence = min(area / (frame_area * 0.5), 1.0) # Cap at 50% screen coverage
                
                # 3. Calculate Speed (Energy)
                speed = 0.0
                prev_center = self.track_history[track_id]
                if prev_center is not None:
                    # Euclidean distance
                    distance = math.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                    # Normalize: 50 pixels per frame is "High Energy"
                    speed = min(distance / 50.0, 1.0)
                
                # Update history
                self.track_history[track_id] = (center_x, center_y)

                parsed_results.append({
                    "id": track_id,
                    "label": self.model.names[cls],
                    "energy": float(speed),      # 0.0 (still) -> 1.0 (sprinting)
                    "presence": float(presence), # 0.0 (far) -> 1.0 (close)
                    "pan": float(center_x / width) # 0.0 (left) -> 1.0 (right)
                })
                
                # Draw Energy on screen for debug
                cv2.putText(annotated_frame, f"E:{speed:.2f}", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                print(parsed_results[-1])

        return parsed_results, annotated_frame

if __name__ == "__main__":
    print("Testing Behavior Detector...")
    INPUT_PATH = 'video/youtube_World’s Grumpiest Cat I Frozen Planet II I BBC.avi'
    cap = cv2.VideoCapture(INPUT_PATH)
    detector = VisionDetector() 

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        data, visual = detector.detect(frame)
        
        if data:
            # Print the energy of the first animal found
            print(f"Animal {data[0]['id']} ({data[0]['label']}) | Energy: {data[0]['energy']:.2f}")
            
        cv2.imshow("Qualia Behavior Test", visual)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()