import os
import torch
import timm
from config import load_config

class UnsupervisedTrainer:
    def __init__(self, config):
        """
        Initializes the UnsupervisedTrainer.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.start_epoch = 0

    def _build_model(self):
        """
        Builds the DINOv2 model.
        """
        model_name = self.config['model']['name']
        pretrained = self.config['model']['pretrained']
        print(f"Loading model: {model_name} (pretrained: {pretrained})")
        model = timm.create_model(model_name, pretrained=pretrained)
        return model.to(self.device)

    def _build_optimizer(self):
        """
        Builds the optimizer.
        """
        optimizer_name = self.config['training']['optimizer'].lower()
        learning_rate = self.config['training']['learning_rate']
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer

    def _load_checkpoint(self, checkpoint_path):
        """
        Loads a checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

    def _save_checkpoint(self, epoch):
        """
        Saves a checkpoint.
        """
        checkpoint_dir = self.config['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, f'latest_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def freeze_layers(self, num_layers_to_freeze):
        """
        Freezes the initial layers of the model.
        """
        # This is a simplified example. The actual implementation depends on the model architecture.
        # For a Vision Transformer, you might freeze the patch embedding and the first N transformer blocks.
        ct = 0
        for child in self.model.children():
            if ct < num_layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
                ct += 1
        print(f"Froze {num_layers_to_freeze} layers.")


    def train(self, data_loader):
        """
        The main training loop.
        """
        if self.config['training']['resume_training']:
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'latest_checkpoint.pt')
            self._load_checkpoint(checkpoint_path)

        num_layers_to_freeze = self.config['model'].get('frozen_layers', 0)
        if num_layers_to_freeze > 0:
            self.freeze_layers(num_layers_to_freeze)


        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            print(f"Epoch {epoch}/{self.config['training']['epochs']}")
            # --- Training logic goes here ---
            # This will be a self-supervised training loop.
            # For demonstration, we'll just simulate a training step.
            for i, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                # Forward pass, loss calculation, backward pass, optimizer step...
                if i % 10 == 0:
                    print(f"  Batch {i}/{len(data_loader)}")

            self._save_checkpoint(epoch)

if __name__ == '__main__':
    # Add the project root to the Python path
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ppe_detection.config import load_config

    # Example usage:
    config = load_config()
    if config:
        # This is a placeholder for a data loader.
        # In a real scenario, you would use your data_utils to create a real data loader.
        dummy_dataset = torch.utils.data.TensorDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 1, (100,)))
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=config['training']['batch_size'])

        trainer = UnsupervisedTrainer(config)
        trainer.train(dummy_loader)
