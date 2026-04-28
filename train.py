"""Training script for model training."""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    BASELINE_MODEL_PATH, ADVANCED_MODEL_PATH, NUM_CLASSES
)
from src.data import DataManager
from src.model import (
    create_baseline_model, create_advanced_model,
    ModelTrainer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_baseline_model():
    """Train baseline model."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING BASELINE MODEL")
    logger.info("="*60)
    
    # Initialize data manager and load data
    config = __import__('src.config', fromlist=[''])
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader = data_manager.get_dataloaders()
    
    # Create model and trainer
    model = create_baseline_model(num_classes=NUM_CLASSES, device=DEVICE)
    trainer = ModelTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE)
    
    # Train model
    trainer.train(train_loader, val_loader, NUM_EPOCHS)
    
    # Save model
    trainer.save_model(BASELINE_MODEL_PATH)
    
    logger.info("\nBaseline model training completed!")
    return trainer, test_loader

def train_advanced_model():
    """Train advanced model with transfer learning."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING ADVANCED MODEL (Transfer Learning - ResNet50)")
    logger.info("="*60)
    
    # Initialize data manager and load data
    config = __import__('src.config', fromlist=[''])
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader = data_manager.get_dataloaders()
    
    # Create model and trainer
    model = create_advanced_model(num_classes=NUM_CLASSES, device=DEVICE)
    trainer = ModelTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE)
    
    # Train model
    trainer.train(train_loader, val_loader, NUM_EPOCHS)
    
    # Save model
    trainer.save_model(ADVANCED_MODEL_PATH)
    
    logger.info("\nAdvanced model training completed!")
    return trainer, test_loader

def evaluate_model(trainer, test_loader, model_name="Model"):
    """Evaluate model on test set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING {model_name} ON TEST SET")
    logger.info(f"{'='*60}")
    
    test_loss, test_acc = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    return test_acc

if __name__ == "__main__":
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    
    # Train baseline model
    baseline_trainer, test_loader = train_baseline_model()
    baseline_acc = evaluate_model(baseline_trainer, test_loader, "BASELINE MODEL")
    
    # Train advanced model
    advanced_trainer, test_loader = train_advanced_model()
    advanced_acc = evaluate_model(advanced_trainer, test_loader, "ADVANCED MODEL")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Baseline Model Test Accuracy: {baseline_acc:.2f}%")
    logger.info(f"Advanced Model Test Accuracy: {advanced_acc:.2f}%")
    logger.info(f"Improvement: {advanced_acc - baseline_acc:.2f}%")
    logger.info("\nModels saved:")
    logger.info(f"  - Baseline: {BASELINE_MODEL_PATH}")
    logger.info(f"  - Advanced: {ADVANCED_MODEL_PATH}")
