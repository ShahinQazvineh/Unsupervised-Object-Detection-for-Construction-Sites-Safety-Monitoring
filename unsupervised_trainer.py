import os
import torch
import timm
import copy
from config import CONFIG
from dino_loss import DINOLoss

class DINOHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class UnsupervisedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create student and teacher models (headless)
        self.student, self.embed_dim = self._build_model()
        self.teacher, _ = self._build_model()

        # Create student and teacher DINO heads
        out_dim = self.config['model']['out_dim']
        self.student_head = DINOHead(self.embed_dim, out_dim).to(self.device)
        self.teacher_head = DINOHead(self.embed_dim, out_dim).to(self.device)

        # Teacher model and head do not require gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Make sure teacher and student have same weights
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

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
        print(f"Loading model: {model_name} (pretrained: {pretrained})")
        # Load model without a classification head
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = model.embed_dim
        return model.to(self.device), embed_dim

    def _build_optimizer(self):
        optimizer_name = self.config['training']['optimizer'].lower()
        learning_rate = self.config['training']['learning_rate']
        # Parameters now include the student model and the student head
        params = list(self.student.parameters()) + list(self.student_head.parameters())
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.student_head.load_state_dict(checkpoint['student_head_state_dict'])
        self.teacher_head.load_state_dict(checkpoint['teacher_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch):
        checkpoint_dir = self.config['checkpoint_dir_abs']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'student_head_state_dict': self.student_head.state_dict(),
            'teacher_head_state_dict': self.teacher_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    @torch.no_grad()
    def _update_teacher(self, epoch):
        m = self.config['dino']['momentum_teacher']
        # Update teacher model
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(m).add_((1 - m) * student_param.data)
        # Update teacher head
        for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_param.data.mul_(m).add_((1 - m) * student_param.data)

    def train(self, data_loader):
        if self.config['training']['resume_training']:
            checkpoint_path = os.path.join(self.config['checkpoint_dir_abs'], 'latest_checkpoint.pt')
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)

        from tqdm import tqdm

        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.student.train()
            self.student_head.train()
            loop = tqdm(data_loader, leave=True)
            loop.set_description(f"Epoch {epoch}/{self.config['training']['epochs']}")

            total_loss = 0.0
            for i, (images, _) in enumerate(loop):
                images = [img.to(self.device) for img in images]

                # Get model embeddings
                student_embeddings = self.student(images)
                teacher_embeddings = self.teacher(images[:2])

                # Pass embeddings through DINO heads
                student_output = self.student_head(student_embeddings)
                teacher_output = self.teacher_head(teacher_embeddings)

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
