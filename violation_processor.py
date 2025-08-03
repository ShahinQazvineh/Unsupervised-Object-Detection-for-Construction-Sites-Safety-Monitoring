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
        weights = Path(self.config['bot_sort_reid_weights'])
        self.tracker = BotSort(
            reid_weights=weights,
        weights = Path('osnet_x0_25_msmt17.pt')
        self.tracker = BotSort(
            reid_weights=weights, # Placeholder, will be downloaded automatically
            device=device,
            half=False
        )

    def process_violations(self, discovered_objects, frame):
        """
        Processes the discovered objects to detect violations.

        Args:
            discovered_objects (list): A list of dictionaries, where each dictionary
                                       represents a detected object with its class name
                                       and bounding box in the format [x1, y1, x2, y2].
            frame (np.array): The current video frame.

        Returns:
            list: A list of detected violations.
        """
        # Convert the discovered objects to the format expected by BoTSORT
        # which is a numpy array of shape (n, 6) with columns [x1, y1, x2, y2, score, class_id]
        detections = []
        for obj in discovered_objects:
            box = obj['box']
            # We'll use a dummy score and class_id for now
            detections.append(box + [1.0, 0])
        detections = np.array(detections)

        # Update the tracker
        tracked_objects = self.tracker.update(detections, frame)

        violations = []
        # Example violation logic:
        # First, we need to map the tracked objects back to their class names.
        # This is a bit tricky since the tracker only deals with boxes and IDs.
        # For now, we'll assume the class_id from the input is preserved.
        # A more robust solution would be to associate the tracked boxes with the
        # initial detections.

        # Create a dictionary to hold the tracked persons and their associated equipment
        persons = {}
        for obj in tracked_objects:
            track_id = int(obj[4])
            class_id = int(obj[5])
            # Assuming class_id 0 is person, 1 is helmet, 2 is vest
            # This mapping should be consistent with the discovery process
            if class_id == self.config['person_class_id']: # Person
                if track_id not in persons:
                    persons[track_id] = {'helmet': False, 'vest': False}
            elif class_id in self.config['required_ppe_ids']: # PPE
            if class_id == 0: # Person
                if track_id not in persons:
                    persons[track_id] = {'helmet': False, 'vest': False}
            elif class_id == 1: # Helmet
                # This is a simplified association. A real implementation would need
                # to check for spatial proximity (e.g., IoU) between the person and the helmet.
                # For now, we'll just assume any helmet in the frame is worn by any person.
                 for person_id in persons:
                    if class_id == self.config['helmet_class_id']:
                        persons[person_id]['helmet'] = True
                    elif class_id == self.config['vest_class_id']:
                        persons[person_id]['vest'] = True

                    persons[person_id]['helmet'] = True
            elif class_id == 2: # Vest
                 for person_id in persons:
                    persons[person_id]['vest'] = True

        # Check for violations
        for person_id, equipment in persons.items():
            if not equipment['helmet']:
                violations.append({'person_id': person_id, 'violation': self.config['violation_map'][self.config['helmet_class_id']]})
            if not equipment['vest']:
                violations.append({'person_id': person_id, 'violation': self.config['violation_map'][self.config['vest_class_id']]})
                violations.append({'person_id': person_id, 'violation': 'no_helmet'})
            if not equipment['vest']:
                violations.append({'person_id': person_id, 'violation': 'no_vest'})


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

    from config import load_config

    # Example usage:
    config = load_config()
    if config:
        violation_processor = ViolationProcessor(config)
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
