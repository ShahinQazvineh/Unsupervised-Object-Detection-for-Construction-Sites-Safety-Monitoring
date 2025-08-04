import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.measure import find_contours
from config import CONFIG

class DiscoveryProcessor:
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(model_path)
        self.model.eval()

        # Hook to extract features from an intermediate layer
        self.features = None
        layer_index = self.config['model'].get('feature_layer', 8)
        self.model.blocks[layer_index].register_forward_hook(self._get_features_hook())

    def _build_model(self, model_path):
        model_name = self.config['model']['name']
        model = timm.create_model(model_name, pretrained=True)

        if model_path:
            print(f"Loading fine-tuned model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            # Adjust key loading based on actual checkpoint structure
            state_dict = checkpoint.get('student_state_dict', checkpoint.get('model_state_dict', checkpoint))
            model.load_state_dict(state_dict)

        return model.to(self.device)

    def _get_features_hook(self):
        def hook(model, input, output):
            self.features = output.detach()
        return hook

    def _preprocess_image(self, image):
        # Image is expected to be a numpy array (H, W, C)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def discover_objects(self, image, n_clusters=5, pca_dim=10):
        """
        Discovers objects in an image using feature clustering.
        """
        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            self.model(input_tensor)

        # Features are (1, num_patches + 1, dim), remove CLS token
        patch_features = self.features[:, 1:, :].squeeze(0).cpu().numpy()

        # PCA for dimensionality reduction
        if pca_dim > 0 and patch_features.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim)
            patch_features = pca.fit_transform(patch_features)

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(patch_features)

        # Reshape clusters to image patch grid
        patch_size = self.model.patch_embed.patch_size[0]
        grid_size = 224 // patch_size
        mask = clusters.reshape(grid_size, grid_size)

        # Upscale mask to original image size
        full_size_mask = cv2.resize(mask.astype(np.uint8),
                                    (image.shape[1], image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        discovered_objects = []
        for cluster_id in range(n_clusters):
            # Create a binary mask for the current cluster
            cluster_mask = (full_size_mask == cluster_id).astype(np.uint8)

            # Find contours
            contours = find_contours(cluster_mask, 0.5)

            for contour in contours:
                # Convert contour to bounding box
                y_min, x_min = contour.min(axis=0)
                y_max, x_max = contour.max(axis=0)

                # Filter out very small boxes
                if (x_max - x_min > 10) and (y_max - y_min > 10):
                    discovered_objects.append({
                        'box': [int(x_min), int(y_min), int(x_max), int(y_max)],
                        'cluster_id': cluster_id
                    })

        return discovered_objects, full_size_mask

if __name__ == '__main__':
    if CONFIG:
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy_image, (100, 100), (200, 300), (255, 0, 0), -1)
        cv2.rectangle(dummy_image, (300, 200), (450, 400), (0, 255, 0), -1)

        discovery_processor = DiscoveryProcessor(CONFIG)

        # Discover objects
        discovered_objects, mask = discovery_processor.discover_objects(dummy_image, n_clusters=3)

        print(f"Discovered {len(discovered_objects)} objects.")

        # Visualize the result
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(dummy_image)
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='viridis')
        plt.title('Discovered Segments')

        for obj in discovered_objects:
            x1, y1, x2, y2 = obj['box']
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            edgecolor='red', facecolor='none', lw=2))

        plt.show()
        print("DiscoveryProcessor executed successfully.")
