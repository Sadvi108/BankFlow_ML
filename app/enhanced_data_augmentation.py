"""
Enhanced Data Augmentation for Bank Receipt Training

This module provides comprehensive data augmentation techniques to expand
the limited training dataset for reference ID extraction.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReceiptAugmentor:
    """Augments receipt images to create diverse training examples."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def augment_image(self, image: np.ndarray, augmentation_level: str = 'medium') -> np.ndarray:
        """
        Apply random augmentations to receipt image.
        
        Args:
            image: Input image as numpy array
            augmentation_level: 'light', 'medium', or 'heavy'
        
        Returns:
            Augmented image
        """
        img = image.copy()
        
        # Define augmentation probabilities based on level
        prob_map = {
            'light': 0.3,
            'medium': 0.5,
            'heavy': 0.7
        }
        prob = prob_map.get(augmentation_level, 0.5)
        
        # Apply augmentations with probability
        if random.random() < prob:
            img = self._rotate(img)
        
        if random.random() < prob:
            img = self._add_noise(img)
        
        if random.random() < prob:
            img = self._adjust_brightness_contrast(img)
        
        if random.random() < prob * 0.5:  # Less frequent
            img = self._blur(img)
        
        if random.random() < prob * 0.3:  # Even less frequent
            img = self._add_shadow(img)
        
        return img
    
    def _rotate(self, image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
        """Rotate image by small random angle."""
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_REPLICATE,
                                 flags=cv2.INTER_LINEAR)
        return rotated
    
    def _add_noise(self, image: np.ndarray, noise_level: float = 10.0) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Randomly adjust brightness and contrast."""
        # Convert to PIL for easier manipulation
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Brightness
        brightness_factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)
        
        # Contrast
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast_factor)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _blur(self, image: np.ndarray) -> np.ndarray:
        """Apply slight Gaussian blur."""
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _add_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add random shadow effect."""
        h, w = image.shape[:2]
        
        # Create random shadow mask
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        # Random shadow position
        if random.random() < 0.5:
            # Vertical shadow
            shadow_start = random.randint(0, w // 2)
            shadow_end = random.randint(w // 2, w)
            for i in range(shadow_start, shadow_end):
                alpha = (i - shadow_start) / (shadow_end - shadow_start)
                shadow_mask[:, i] = 0.6 + 0.4 * alpha
        else:
            # Horizontal shadow
            shadow_start = random.randint(0, h // 2)
            shadow_end = random.randint(h // 2, h)
            for i in range(shadow_start, shadow_end):
                alpha = (i - shadow_start) / (shadow_end - shadow_start)
                shadow_mask[i, :] = 0.6 + 0.4 * alpha
        
        # Apply shadow
        if len(image.shape) == 3:
            shadow_mask = np.stack([shadow_mask] * 3, axis=2)
        
        shadowed = (image.astype(np.float32) * shadow_mask).astype(np.uint8)
        return shadowed
    
    def create_augmented_dataset(self, 
                                 images: List[np.ndarray],
                                 annotations: List[Dict],
                                 num_augmentations: int = 5) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Create augmented dataset from original images and annotations.
        
        Args:
            images: List of original images
            annotations: List of corresponding annotations
            num_augmentations: Number of augmented versions per image
        
        Returns:
            Tuple of (augmented_images, augmented_annotations)
        """
        aug_images = []
        aug_annotations = []
        
        for img, ann in zip(images, annotations):
            # Keep original
            aug_images.append(img)
            aug_annotations.append(ann)
            
            # Create augmented versions
            for i in range(num_augmentations):
                aug_level = random.choice(['light', 'medium', 'heavy'])
                aug_img = self.augment_image(img, aug_level)
                
                # Copy annotation with augmentation metadata
                aug_ann = ann.copy()
                aug_ann['augmented'] = True
                aug_ann['augmentation_level'] = aug_level
                aug_ann['original_id'] = ann.get('id', '')
                aug_ann['id'] = f"{ann.get('id', '')}_{i+1}"
                
                aug_images.append(aug_img)
                aug_annotations.append(aug_ann)
        
        logger.info(f"Created {len(aug_images)} images from {len(images)} originals "
                   f"({num_augmentations} augmentations per image)")
        
        return aug_images, aug_annotations


class OCRErrorSimulator:
    """Simulates common OCR errors for data augmentation."""
    
    def __init__(self):
        # Common OCR character confusions
        self.confusions = {
            '0': ['O', 'D', 'Q'],
            'O': ['0', 'Q', 'D'],
            '1': ['I', 'l', '|'],
            'I': ['1', 'l', '|'],
            'l': ['1', 'I', '|'],
            '5': ['S', '8'],
            'S': ['5', '8'],
            '8': ['B', '3', '5'],
            'B': ['8', '3'],
            '2': ['Z'],
            'Z': ['2'],
            '6': ['G', 'b'],
            'G': ['6', 'C'],
            'C': ['G', 'O'],
        }
    
    def simulate_ocr_errors(self, text: str, error_rate: float = 0.05) -> str:
        """
        Simulate OCR errors in text.
        
        Args:
            text: Original text
            error_rate: Probability of error per character (0.0 to 1.0)
        
        Returns:
            Text with simulated OCR errors
        """
        result = []
        for char in text:
            if random.random() < error_rate and char in self.confusions:
                # Replace with confused character
                result.append(random.choice(self.confusions[char]))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def generate_ocr_variations(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate multiple OCR error variations of text."""
        variations = [text]  # Include original
        
        for _ in range(num_variations):
            error_rate = random.uniform(0.02, 0.1)
            variation = self.simulate_ocr_errors(text, error_rate)
            if variation not in variations:
                variations.append(variation)
        
        return variations


class SyntheticReceiptGenerator:
    """Generates synthetic receipt variations."""
    
    def __init__(self):
        self.reference_id_patterns = [
            lambda: ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=random.randint(8, 15))),
            lambda: f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}{''.join(random.choices('0123456789', k=random.randint(8, 12)))}",
            lambda: f"PBB{''.join(random.choices('0123456789', k=12))}",
            lambda: f"MB{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))}",
            lambda: f"CIMB{''.join(random.choices('0123456789', k=10))}",
        ]
    
    def generate_reference_id(self) -> str:
        """Generate a synthetic reference ID."""
        pattern = random.choice(self.reference_id_patterns)
        return pattern()
    
    def create_synthetic_variation(self, annotation: Dict) -> Dict:
        """
        Create synthetic variation of an annotation with new reference ID.
        
        Args:
            annotation: Original annotation
        
        Returns:
            New annotation with synthetic reference ID
        """
        new_ann = annotation.copy()
        
        # Generate new reference ID
        new_ref_id = self.generate_reference_id()
        
        # Update annotation
        if 'ground_truth' in new_ann:
            new_ann['ground_truth']['transaction_id'] = new_ref_id
        
        if 'fields' in new_ann:
            new_ann['fields']['transaction_number'] = new_ref_id
            new_ann['fields']['reference_number'] = new_ref_id
        
        # Update OCR text (simple replacement of old ID with new ID)
        if 'ocr_text' in new_ann and 'ground_truth' in annotation:
            old_id = annotation['ground_truth'].get('transaction_id', '')
            if old_id:
                new_ann['ocr_text'] = new_ann['ocr_text'].replace(old_id, new_ref_id)
        
        # Mark as synthetic
        new_ann['synthetic'] = True
        new_ann['original_id'] = annotation.get('id', '')
        new_ann['id'] = f"{annotation.get('id', '')}_synthetic"
        
        return new_ann


def augment_training_data(annotations: List[Dict],
                          image_dir: Path,
                          num_image_augmentations: int = 3,
                          num_synthetic_variations: int = 2) -> List[Dict]:
    """
    Main function to augment training data.
    
    Args:
        annotations: List of original annotations
        image_dir: Directory containing images
        num_image_augmentations: Number of image augmentations per sample
        num_synthetic_variations: Number of synthetic ID variations per sample
    
    Returns:
        Augmented annotations list
    """
    augmentor = ReceiptAugmentor()
    ocr_simulator = OCRErrorSimulator()
    synthetic_gen = SyntheticReceiptGenerator()
    
    augmented_annotations = []
    
    for ann in annotations:
        # Add original
        augmented_annotations.append(ann)
        
        # Image augmentations
        # Note: Actual image augmentation happens during training
        # Here we just create annotation entries
        for i in range(num_image_augmentations):
            aug_ann = ann.copy()
            aug_ann['augmented'] = True
            aug_ann['augmentation_type'] = 'image'
            aug_ann['augmentation_level'] = random.choice(['light', 'medium', 'heavy'])
            aug_ann['id'] = f"{ann.get('id', '')}_aug_{i}"
            augmented_annotations.append(aug_ann)
        
        # Synthetic variations
        for i in range(num_synthetic_variations):
            syn_ann = synthetic_gen.create_synthetic_variation(ann)
            syn_ann['id'] = f"{ann.get('id', '')}_syn_{i}"
            augmented_annotations.append(syn_ann)
    
    logger.info(f"Augmented {len(annotations)} annotations to {len(augmented_annotations)} total")
    
    return augmented_annotations
