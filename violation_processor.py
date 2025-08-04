from boxmot import BotSort
import numpy as np
import torch
from pathlib import Path
from config import CONFIG

class ViolationProcessor:
    def __init__(self, config):
        """
        Initializes the ViolationProcessor.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Use .get() to provide a default value if the key is missing
        reid_weights_path = self.config.get('bot_sort_reid_weights', 'osnet_x0_25_msmt17.pt')
        weights = Path(reid_weights_path)

        self.tracker = BotSort(
            reid_weights=weights,
            device=device,
            half=False
        )

    def _calculate_iou(self, boxA, boxB):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_violations(self, discovered_objects, frame):
        """
        Processes discovered objects to detect safety violations using spatial association.
        """
        # Use a dummy confidence score and the actual class_id from discovery
        detections = np.array(
            [obj['box'] + [1.0, obj['class_id']] for obj in discovered_objects]
        )

        if detections.shape[0] == 0:
            return []

        # Update the tracker
        tracked_objects = self.tracker.update(detections, frame)

        persons = {}
        ppes = []

        # Separate persons and PPE from tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, class_id, _ = obj
            box = [int(x1), int(y1), int(x2), int(y2)]
            track_id = int(track_id)
            class_id = int(class_id)

            if class_id == self.config['discovery']['class_map'].get('person', -1):
                persons[track_id] = {'box': box, 'helmet': False, 'vest': False}
            elif class_id in [self.config['discovery']['class_map'].get('helmet', -2),
                              self.config['discovery']['class_map'].get('vest', -3)]:
                ppes.append({'box': box, 'class_id': class_id})

        # Associate PPE with persons based on IoU
        for person_id, person_data in persons.items():
            person_box = person_data['box']
            for ppe in ppes:
                ppe_box = ppe['box']
                iou = self._calculate_iou(person_box, ppe_box)

                # A simple threshold to associate PPE with a person
                if iou > 0:
                    if ppe['class_id'] == self.config['discovery']['class_map'].get('helmet', -2):
                        person_data['helmet'] = True
                    elif ppe['class_id'] == self.config['discovery']['class_map'].get('vest', -3):
                        person_data['vest'] = True

        # Check for violations
        violations = []
        for person_id, equipment in persons.items():
            if not equipment['helmet']:
                violations.append({'person_id': person_id, 'violation': 'Missing Helmet'})
            if not equipment['vest']:
                violations.append({'person_id': person_id, 'violation': 'Missing Vest'})

        if violations:
            print(f"Detected {len(violations)} violations.")
        return violations

if __name__ == '__main__':
    # Add the project root to the Python path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import CONFIG

    # Example usage:
    if CONFIG:
        violation_processor = ViolationProcessor(CONFIG)
        # Example discovered objects
        discovered_objects = [
            {'class_name': 'person', 'box': [100, 100, 200, 400]},
            {'class_name': 'vest', 'box': [120, 150, 180, 250]},
            # This person is missing a helmet
            {'class_name': 'person', 'box': [300, 100, 400, 400]},
        ]
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        violations = violation_processor.process_violations(discovered_objects, dummy_frame)
        print(violations)
