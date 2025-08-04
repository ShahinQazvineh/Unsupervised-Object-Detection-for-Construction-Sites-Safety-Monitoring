import os
import torch
import timm
import copy
from config import CONFIG
from dino_loss import DINOLoss

class UnsupervisedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create student and teacher models
        self.student = self._build_model()
        self.teacher = self._build_model()

        # Teacher model does not require gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Make sure teacher and student have same weights
        self.teacher.load_state_dict(self.student.state_dict())

        # The output dimension is now set in the _build_model method
        out_dim = self.config['model']['out_dim']

        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            n_crops=self.config['dino']['n_crops'],
            warmup_teacher_temp=self.config['dino']['warmup_teacher_temp'],
            teacher_temp=self.config['dino']['teacher_temp'],
            warmup_teacher_temp_epochs=self.config['dino']['warmup_teacher_temp_epochs'],
            n_epochs=self.config['training']['epochs']
        ).to(self.device)

        self.optimizer = self._build_optimizer()
        self.start_epoch = 0

    def _build_model(self):
        model_name = self.config['model']['name']
        pretrained = self.config['model']['pretrained']
        out_dim = self.config['model'].get('out_dim', 65536) # Default DINO head dim

        print(f"Loading model: {model_name} (pretrained: {pretrained})")
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) # num_classes=0 removes head

        # Get the embedding dimension from the model
        in_features = model.embed_dim

        # Create a DINO-compatible head and attach it
        model.head = torch.nn.Linear(in_features, out_dim)

        # Store the output dimension back in the config for the loss function
        self.config['model']['out_dim'] = out_dim

        return model.to(self.device)

    def _build_optimizer(self):
        optimizer_name = self.config['training']['optimizer'].lower()
        learning_rate = self.config['training']['learning_rate']
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            return
        checkpoint = torch.load(checkpoint_path)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch):
        checkpoint_dir = self.config['checkpoint_dir_abs']
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    @torch.no_grad()
    def _update_teacher(self, epoch):
        m = self.config['dino']['momentum_teacher']
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(m).add_((1 - m) * student_param.data)

    def train(self, data_loader):
        if self.config['training']['resume_training']:
            checkpoint_path = os.path.join(self.config['checkpoint_dir_abs'], 'latest_checkpoint.pt')
            self._load_checkpoint(checkpoint_path)

        from tqdm import tqdm

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.student.train()
            loop = tqdm(data_loader, leave=True)
            loop.set_description(f"Epoch {epoch}/{self.config['training']['epochs']}")

            total_loss = 0.0
            for i, (images, _) in enumerate(loop):
                images = [img.to(self.device) for img in images]

                teacher_output = self.teacher(images[:2]) # only global views pass through the teacher
                student_output = self.student(images)

                loss = self.dino_loss(student_output, teacher_output, epoch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_teacher(epoch)

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch} finished with average loss: {avg_loss:.4f}")

            self._save_checkpoint(epoch)

if __name__ == '__main__':
    # Add the project root to the Python path
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import CONFIG

    # Example usage:
    if CONFIG:
        # This is a placeholder for a data loader.
        # In a real scenario, you would use your data_utils to create a real data loader.
        dummy_dataset = torch.utils.data.TensorDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 1, (100,)))
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=CONFIG['training']['batch_size'])

        trainer = UnsupervisedTrainer(CONFIG)
        trainer.train(dummy_loader)
