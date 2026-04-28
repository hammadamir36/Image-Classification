"""Inference module for making predictions."""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class Inference:
    """Inference engine for model predictions."""
    
    def __init__(self, model_path, model, device="cpu", image_size=224):
        self.device = device
        self.image_size = image_size
        self.model = model
        self.model.to(device)
        self.model.eval()
        
        # Load ImageNet class labels from JSON file
        self.class_labels = self._load_class_labels()
        
        # Load model weights if available and compatible
        if Path(model_path).exists():
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=device)
                )
                logger.info(f"Loaded model from {model_path}")
            except (RuntimeError, KeyError) as e:
                logger.info(f"Using fresh ImageNet pretrained weights (saved model incompatible with new EfficientNet-B5 architecture)")
    
    def _load_class_labels(self):
        """
        Load ImageNet class labels from JSON file.
        
        Returns:
            dict: Mapping of class indices to class names
        """
        try:
            json_path = Path(__file__).parent / 'imagenet_class_index.json'
            with open(json_path, 'r') as f:
                class_index = json.load(f)
            
            # Convert JSON format {"0": ["n01440764", "tench"], ...} 
            # to simple dict {0: "tench", ...}
            class_labels = {int(idx): names[1] for idx, names in class_index.items()}
            logger.info(f"Loaded {len(class_labels)} ImageNet class labels from {json_path}")
            return class_labels
        except FileNotFoundError:
            logger.warning(f"ImageNet class index JSON file not found at {json_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading class labels: {e}")
            return {}
    
    def _get_class_name(self, class_idx):
        """
        Get the human-readable class name for a given index.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            str: Class name or fallback text if not found
        """
        if class_idx in self.class_labels:
            return self.class_labels[class_idx]
        else:
            return f"Class {int(class_idx)}"
    
    def preprocess_image(self, image_path_or_pil):
        """Preprocess image for inference."""
        from torchvision import transforms
        
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
        
        # Use ImageNet standard normalization values
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)
    
    def predict(self, image_path_or_pil, top_k=5):
        """
        Make prediction on image.
        
        Args:
            image_path_or_pil: Path to image or PIL Image
            top_k: Return top k predictions
        
        Returns:
            dict: Predictions with class names and probabilities
        """
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_path_or_pil)
            image_tensor = image_tensor.to(self.device)
            
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, top_k)
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            predictions = []
            for idx, prob in zip(top_indices, top_probs):
                class_name = self._get_class_name(int(idx))
                predictions.append({
                    'class_id': int(idx),
                    'class_name': class_name,
                    'probability': float(prob),
                    'confidence': f"{float(prob) * 100:.2f}%"
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'top_prediction': predictions[0]['class_name'] if predictions else 'Unknown'
            }
    
    def predict_with_tta(self, image_path_or_pil, top_k=5, num_augmentations=5):
        """
        Make prediction with Test-Time Augmentation (TTA) for improved accuracy.
        
        Args:
            image_path_or_pil: Path to image or PIL Image
            top_k: Return top k predictions
            num_augmentations: Number of augmented versions to average
        
        Returns:
            dict: Predictions with class names and probabilities
        """
        from torchvision import transforms
        
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
        
        # Define augmentation pipeline
        augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        
        base_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        with torch.no_grad():
            all_probabilities = None
            
            for _ in range(num_augmentations):
                # Apply augmentations
                augmented_image = augmentation_transforms(image)
                image_tensor = base_transform(augmented_image)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Get prediction
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                if all_probabilities is None:
                    all_probabilities = probabilities.cpu()
                else:
                    all_probabilities += probabilities.cpu()
            
            # Average probabilities across augmentations
            all_probabilities /= num_augmentations
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(all_probabilities, top_k)
            top_probs = top_probs.numpy()[0]
            top_indices = top_indices.numpy()[0]
            
            predictions = []
            for idx, prob in zip(top_indices, top_probs):
                class_name = self._get_class_name(int(idx))
                predictions.append({
                    'class_id': int(idx),
                    'class_name': class_name,
                    'probability': float(prob),
                    'confidence': f"{float(prob) * 100:.2f}%"
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'top_prediction': predictions[0]['class_name'] if predictions else 'Unknown'
            }
    
    def batch_predict(self, image_paths, top_k=5):
        """Batch prediction on multiple images."""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, top_k)
            results.append(result)
        return results
