"""Model definitions and utilities."""
import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class SimpleBaseline(nn.Module):
    """Simple CNN baseline model."""
    
    def __init__(self, num_classes=100):
        super(SimpleBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_baseline_model(num_classes=100, device="cpu"):
    """Create baseline model."""
    logger.info("Creating baseline CNN model...")
    model = SimpleBaseline(num_classes=num_classes)
    model.to(device)
    return model

def create_advanced_model(num_classes=100, device="cpu"):
    """Create advanced model using transfer learning (EfficientNet-B5) with ImageNet 1000 classes."""
    logger.info("Creating advanced model with transfer learning (EfficientNet-B5)...")
    
    # Load pre-trained EfficientNet-B5 with ImageNet 1000 weights
    # Keep the original 1000-class output for better real-world object recognition
    model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
    
    model.to(device)
    return model

class ModelTrainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, device="cpu", learning_rate=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"Batch [{batch_idx + 1}], Loss: {loss.item():.4f}"
                )
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs):
        """Train model for multiple epochs."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )
            logger.info(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
