import torch
import timm
import cv2
import numpy as np
from config import load_config

class DiscoveryProcessor:
    def __init__(self, config, model_path=None):
        """
        Initializes the DiscoveryProcessor.

        Args:
            config (dict): The configuration dictionary.
            model_path (str, optional): Path to the fine-tuned model. If None, the pretrained model is loaded.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(model_path)
        self.class_map = self.config['discovery']['class_map']

    def _build_model(self, model_path):
        """
        Builds the DINOv2 model and loads the fine-tuned weights if provided.
        """
        model_name = self.config['model']['name']
        model = timm.create_model(model_name, pretrained=True)
        print(f"Model image size: {model.patch_embed.img_size}")


        if model_path:
            print(f"Loading fine-tuned model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

        return model.to(self.device)

    def get_attention_maps(self, image):
        """
        Gets the attention maps from the model for a given image.
        This is a simplified example and might need adjustments based on the model's architecture.
        """
        # The hook to get the attention maps
        def get_attention(module, input, output):
            self.attention = output

        # Register the hook on the last attention layer
        # This needs to be adapted to the specific model architecture
        # For ViT, it's usually the last block's attention
        handle = self.model.blocks[-1].attn.register_forward_hook(get_attention)

        # Preprocess the image and pass it through the model
        # This is a placeholder for actual image preprocessing
        img_size = self.model.patch_embed.img_size
        input_tensor = torch.randn(1, 3, img_size[0], img_size[1]).to(self.device) # Dummy tensor
        with torch.no_grad():
            _ = self.model(input_tensor)

        print(f"Attention shape: {self.attention.shape}")

        handle.remove() # Remove the hook
        return self.attention

    def generate_object_masks(self, image):
        """
        Generates object masks from the attention maps.
        """
        attentions = self.get_attention_maps(image).squeeze(0)
        # The attention tensor is 3D: (num_patches + 1, num_heads, dim_per_head)
        # We are interested in the attention from the class token to the patches
        # which is the first row of the attention matrix.
        # However, the output of the attention layer in timm's ViT is not the attention matrix.
        # It's the output of the attention layer.
        # To get the attention maps, we need to modify the forward pass of the attention layer.
        # For simplicity, we will just reshape the output and average over the heads.
        # This is a very rough approximation of object discovery.
        num_patches = attentions.shape[0] - 1
        w_featmap = h_featmap = int(np.sqrt(num_patches))
        attentions = attentions[1:, :] # remove cls token
        attentions = attentions.mean(axis=1) # average over heads
        mask = attentions.reshape(w_featmap, h_featmap)
        mask = cv2.resize(mask.cpu().numpy(), (image.shape[1], image.shape[0]))
        # Threshold the mask to get a binary mask
        _, mask = cv2.threshold(mask, mask.mean(), 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)

    def apply_class_map(self, discovered_objects):
        """
        Applies the manual class map to assign semantic labels.
        This is a placeholder as the object discovery is not implemented yet.
        """
        # In a real scenario, `discovered_objects` would be a list of objects,
        # each with a cluster ID. This function would map the cluster ID to a class name.
        labeled_objects = []
        for obj in discovered_objects:
            cluster_id = obj['cluster_id']
            if cluster_id in self.class_map:
                obj['class_name'] = self.class_map[cluster_id]
                labeled_objects.append(obj)
        return labeled_objects

if __name__ == '__main__':
    # Add the project root to the Python path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ppe_detection.config import load_config

    # Example usage:
    config = load_config()
    if config:
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Initialize the discovery processor
        # We are not providing a model_path, so it will use the pretrained model
        discovery_processor = DiscoveryProcessor(config)

        # Generate object masks
        masks = discovery_processor.generate_object_masks(dummy_image)
        print(f"Generated masks with shape: {masks.shape}")

        # In a real scenario, you would have a list of discovered objects
        # with cluster IDs to test the apply_class_map function.
        # For now, we'll just show that the processor can be initialized.
        print("DiscoveryProcessor initialized successfully.")
