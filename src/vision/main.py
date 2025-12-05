import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt

from audio import Synth

class Persister:
    def __init__(self, max_distance=150, max_skipped_frames=30):
        """
        A helper to keep IDs consistent even if detection flickers.
        
        Args:
            max_distance (int): Max pixels an object can move while 'invisible' to still be matched.
            max_skipped_frames (int): How long to remember a lost object before forgetting it.
        """
        self.max_distance = max_distance
        self.max_skipped_frames = max_skipped_frames
        self.id_map = {}
        self.active_tracks = {}
        self.next_custom_id = 1

    def update(self, yolo_detections):
        """
        Takes raw YOLO detections and returns detections with consistent IDs.
        """
        current_frame_yolo_ids = set()
        consistent_results = []
        
        for det in yolo_detections:
            yolo_id = det['id']
            center = det['center']
            current_frame_yolo_ids.add(yolo_id)

            if yolo_id in self.id_map:
                custom_id = self.id_map[yolo_id]
                self.active_tracks[custom_id]['last_center'] = center
                self.active_tracks[custom_id]['skipped_frames'] = 0
                
            else:
                found_match = False
                for cid, track_data in self.active_tracks.items():
                    if track_data['skipped_frames'] > 0:
                        last_c = track_data['last_center']
                        dist = math.hypot(center[0] - last_c[0], center[1] - last_c[1])
                        
                        if dist < self.max_distance:
                            self.id_map[yolo_id] = cid
                            track_data['last_center'] = center
                            track_data['skipped_frames'] = 0
                            custom_id = cid
                            found_match = True
                            break
                
                if not found_match:
                    custom_id = self.next_custom_id
                    self.next_custom_id += 1
                    self.id_map[yolo_id] = custom_id
                    self.active_tracks[custom_id] = {'last_center': center, 'skipped_frames': 0}

            det['consistent_id'] = custom_id
            consistent_results.append(det)

        active_custom_ids_this_frame = {self.id_map[y_id] for y_id in current_frame_yolo_ids if y_id in self.id_map}
        
        lost_ids_to_delete = []
        for cid in self.active_tracks:
            if cid not in active_custom_ids_this_frame:
                self.active_tracks[cid]['skipped_frames'] += 1
                if self.active_tracks[cid]['skipped_frames'] > self.max_skipped_frames:
                    lost_ids_to_delete.append(cid)

        for cid in lost_ids_to_delete:
            del self.active_tracks[cid]
            yolo_ids_to_remove = [yid for yid, map_cid in self.id_map.items() if map_cid == cid]
            for yid in yolo_ids_to_remove:
                del self.id_map[yid]

        return consistent_results


class VisionDetector:
    def __init__(self, model_name="yolov10s.pt", conf_threshold=0.2):
        """
        Initialize the YOLO detector with tracking capabilities.

        Args:
            model_name (str): 'yolov8s.pt' recommended.
            conf_threshold (float): Lowered to 0.4 to catch 'hamster-like' animals.
        """
        print(f"Loading Model: {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.stitcher = Persister(max_distance=100, max_skipped_frames=60)
        self.target_classes = [0]
        self.track_history = defaultdict(lambda: None)
        self.hist = []
        self.synth = Synth()

    def detect(self, frame):
        """
        Run tracking on a frame and extract behavioral metrics.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            classes=self.target_classes,
            verbose=False,
        )

        parsed_results = []
        result = results[0]
        height, width = frame.shape[:2]
        annotated_frame = result.plot()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                area = (x2 - x1) * (y2 - y1)
                frame_area = width * height
                presence = min(area / (frame_area * 0.5), 1.0)

                speed = 0.0
                prev_center = self.track_history[track_id]
                if prev_center is not None:
                    distance = math.sqrt(
                        (center_x - prev_center[0]) ** 2
                        + (center_y - prev_center[1]) ** 2
                    )
                    speed = min(distance / 50.0, 1.0)

                self.track_history[track_id] = (center_x, center_y)

                parsed_results.append(
                    {
                        "id": track_id,
                        "label": self.model.names[cls],
                        "energy": float(speed),
                        "presence": float(presence),
                        "pan": float(center_x / width),
                        "center": (center_x,center_y)
                    }
                )

                cv2.putText(
                    annotated_frame,
                    f"E:{speed:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        consistent_results = self.stitcher.update(parsed_results)

        if consistent_results:
            result = consistent_results[0]
            self.hist.append(
                {
                    "id": result["consistent_id"],
                    "label": result["label"],
                    "energy": result["energy"],
                    "presence": result["presence"],
                    "pan": result["pan"],
                    "center": result["center"]
                }
            )
        
            if result["energy"] > 0.2:
                self.synth.gen_sound(result["energy"], result["presence"])
    
        return consistent_results, annotated_frame


    def plot_history(self, output_path="detector_history.png"):
        """
        Plot the detection history and save as an image.

        Args:
            output_path (str): Path to save the plot image.
        """
        if not self.hist:
            print("No history data to plot.")
            return

        data_by_id = defaultdict(lambda: {"energy": [], "presence": [], "pan": [], "label": None, "frame": []})
        
        for idx, entry in enumerate(self.hist):
            track_id = entry["id"]
            data_by_id[track_id]["energy"].append(entry["energy"])
            data_by_id[track_id]["presence"].append(entry["presence"])
            data_by_id[track_id]["pan"].append(entry["pan"])
            data_by_id[track_id]["label"] = entry["label"]
            data_by_id[track_id]["frame"].append(idx)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Animal Behavior Metrics Over Time", fontsize=14, fontweight="bold")

        ax1 = axes[0]
        for track_id, data in data_by_id.items():
            ax1.plot(data["frame"], data["energy"], label=f"ID {track_id} ({data['label']})", alpha=0.7)
        ax1.set_ylabel("Energy", fontweight="bold")
        ax1.set_title("Energy (0.0 = still, 1.0 = sprinting)")
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

        ax2 = axes[1]
        for track_id, data in data_by_id.items():
            ax2.plot(data["frame"], data["presence"], label=f"ID {track_id} ({data['label']})", alpha=0.7)
        ax2.set_ylabel("Presence", fontweight="bold")
        ax2.set_title("Presence (0.0 = far, 1.0 = close)")
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=8)

        ax3 = axes[2]
        for track_id, data in data_by_id.items():
            ax3.plot(data["frame"], data["pan"], label=f"ID {track_id} ({data['label']})", alpha=0.7)
        ax3.set_xlabel("Frame Index", fontweight="bold")
        ax3.set_ylabel("Pan", fontweight="bold")
        ax3.set_title("Pan Position (0.0 = left, 1.0 = right)")
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    print("Testing Behavior Detector...")
    # INPUT_PATH = "video/youtube_World’s Grumpiest Cat I Frozen Planet II I BBC.avi"
    INPUT_PATH = "video/pplwalk.avi"
    cap = cv2.VideoCapture(INPUT_PATH)
    detector = VisionDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        data, visual = detector.detect(frame)

        if data:
            print(
                f"Animal {data[0]['id']} ({data[0]['label']}) | Energy: {data[0]['energy']:.2f}"
            )

        cv2.imshow("Qualia Behavior Test", visual)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.plot_history("detector_history.png")

    cap.release()
    cv2.destroyAllWindows()
