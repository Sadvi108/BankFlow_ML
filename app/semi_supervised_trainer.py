"""
Semi-Supervised Learning for Bank Receipt Training

This module implements semi-supervised learning to leverage unlabeled receipts
by generating pseudo-labels from high-confidence predictions.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class SemiSupervisedTrainer:
    """Implements semi-supervised learning for receipt processing."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.85,
                 min_agreement: int = 2):
        """
        Initialize semi-supervised trainer.
        
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels
            min_agreement: Minimum number of methods that must agree
        """
        self.confidence_threshold = confidence_threshold
        self.min_agreement = min_agreement
    
    def generate_pseudo_labels(self,
                               unlabeled_receipts: List[Path],
                               ocr_pipeline,
                               pattern_extractor) -> List[Dict]:
        """
        Generate pseudo-labels for unlabeled receipts.
        
        Args:
            unlabeled_receipts: List of paths to unlabeled receipt files
            ocr_pipeline: OCR pipeline for text extraction
            pattern_extractor: Pattern-based extractor (ultimate_patterns_v3)
        
        Returns:
            List of pseudo-labeled annotations
        """
        pseudo_labels = []
        
        logger.info(f"Generating pseudo-labels for {len(unlabeled_receipts)} receipts...")
        
        for receipt_path in tqdm(unlabeled_receipts, desc="Pseudo-labeling"):
            try:
                # Extract text using OCR
                ocr_result = ocr_pipeline.process_file(str(receipt_path))
                text = ocr_result.get('text', '')
                
                if not text or len(text) < 50:
                    logger.debug(f"Skipping {receipt_path.name}: insufficient text")
                    continue
                
                # Extract fields using pattern matcher
                extraction = pattern_extractor(text)
                
                # Check confidence and quality
                if self._is_high_quality_extraction(extraction):
                    # Create pseudo-label annotation
                    annotation = self._create_annotation(
                        receipt_path, 
                        ocr_result, 
                        extraction
                    )
                    pseudo_labels.append(annotation)
                    logger.debug(f"✓ Pseudo-labeled {receipt_path.name}")
                else:
                    logger.debug(f"✗ Low confidence for {receipt_path.name}")
                    
            except Exception as e:
                logger.warning(f"Error processing {receipt_path.name}: {e}")
                continue
        
        logger.info(f"Generated {len(pseudo_labels)} high-quality pseudo-labels "
                   f"from {len(unlabeled_receipts)} receipts "
                   f"({len(pseudo_labels)/len(unlabeled_receipts)*100:.1f}% success rate)")
        
        return pseudo_labels
    
    def _is_high_quality_extraction(self, extraction: Dict) -> bool:
        """
        Check if extraction meets quality criteria for pseudo-labeling.
        
        Args:
            extraction: Extraction result from pattern matcher
        
        Returns:
            True if high quality, False otherwise
        """
        # Must have bank identified
        if extraction.get('bank_name') == 'Unknown':
            return False
        
        # Must have at least one transaction ID
        transaction_ids = extraction.get('all_ids', [])
        if not transaction_ids:
            return False
        
        # Check confidence score
        confidence = extraction.get('confidence', 0)
        if confidence < self.confidence_threshold:
            return False
        
        # Validate transaction ID format
        primary_id = transaction_ids[0]
        if not self._is_valid_transaction_id(primary_id):
            return False
        
        # Additional quality checks
        # - Must have reasonable length
        if len(primary_id) < 6 or len(primary_id) > 25:
            return False
        
        # - Must contain at least one digit
        if not any(c.isdigit() for c in primary_id):
            return False
        
        return True
    
    def _is_valid_transaction_id(self, tid: str) -> bool:
        """Validate transaction ID format."""
        if not tid:
            return False
        
        # Remove spaces and quotes for validation
        tid_clean = tid.replace(' ', '').replace("'", "")
        
        # Basic validation
        if len(tid_clean) < 6 or len(tid_clean) > 30:
            return False
        
        # Must be alphanumeric (with allowed special chars)
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/')
        if not all(c in allowed_chars for c in tid_clean.upper()):
            return False
        
        # Must not be a common false positive
        false_positives = {
            'REFERENCE', 'TRANSACTION', 'PAYMENT', 'TRANSFER',
            'MAYBANK', 'CIMB', 'PUBLIC', 'RHB', 'DUITNOW'
        }
        if tid_clean.upper() in false_positives:
            return False
        
        return True
    
    def _create_annotation(self,
                          receipt_path: Path,
                          ocr_result: Dict,
                          extraction: Dict) -> Dict:
        """Create annotation from extraction results."""
        transaction_ids = extraction.get('all_ids', [])
        primary_id = transaction_ids[0] if transaction_ids else None
        
        annotation = {
            'id': receipt_path.stem,
            'file_path': str(receipt_path),
            'ocr_text': ocr_result.get('text', ''),
            'ocr_tokens': ocr_result.get('tokens', []),
            'ocr_confidence': ocr_result.get('confidence', 0),
            'ground_truth': {
                'bank_name': extraction.get('bank_name'),
                'transaction_id': primary_id,
                'date': extraction.get('date'),
                'amount': extraction.get('amount'),
            },
            'fields': {
                'transaction_number': primary_id,
                'reference_number': primary_id,
                'bank': extraction.get('bank_name'),
                'date': extraction.get('date'),
                'amount': extraction.get('amount'),
            },
            'meta': {
                'pseudo_labeled': True,
                'confidence': extraction.get('confidence', 0),
                'all_candidate_ids': transaction_ids,
                'extraction_method': extraction.get('method', 'pattern'),
            }
        }
        
        return annotation
    
    def filter_and_rank_pseudo_labels(self,
                                      pseudo_labels: List[Dict],
                                      top_k: Optional[int] = None) -> List[Dict]:
        """
        Filter and rank pseudo-labels by quality.
        
        Args:
            pseudo_labels: List of pseudo-labeled annotations
            top_k: If specified, return only top k highest quality labels
        
        Returns:
            Filtered and ranked pseudo-labels
        """
        # Score each pseudo-label
        scored_labels = []
        for label in pseudo_labels:
            score = self._calculate_quality_score(label)
            scored_labels.append((label, score))
        
        # Sort by score (descending)
        scored_labels.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum score threshold
        min_score = 0.7
        filtered_labels = [label for label, score in scored_labels if score >= min_score]
        
        # Take top k if specified
        if top_k is not None:
            filtered_labels = filtered_labels[:top_k]
        
        logger.info(f"Filtered {len(pseudo_labels)} pseudo-labels to {len(filtered_labels)} "
                   f"high-quality samples")
        
        return filtered_labels
    
    def _calculate_quality_score(self, annotation: Dict) -> float:
        """Calculate quality score for pseudo-label."""
        score = 0.0
        
        # Base score from extraction confidence
        confidence = annotation.get('meta', {}).get('confidence', 0)
        score += confidence * 0.5
        
        # OCR quality
        ocr_conf = annotation.get('ocr_confidence', 0)
        score += (ocr_conf / 100) * 0.2
        
        # Transaction ID quality
        tid = annotation.get('ground_truth', {}).get('transaction_id', '')
        if tid:
            # Longer IDs are generally more reliable
            length_score = min(len(tid) / 15, 1.0) * 0.1
            score += length_score
            
            # Alphanumeric IDs are more reliable than pure numeric
            if not tid.replace(' ', '').isdigit():
                score += 0.1
        
        # Bank identification
        if annotation.get('ground_truth', {}).get('bank_name') != 'Unknown':
            score += 0.1
        
        return min(score, 1.0)
    
    def combine_labeled_and_pseudo(self,
                                   labeled_data: List[Dict],
                                   pseudo_labels: List[Dict],
                                   pseudo_weight: float = 0.5) -> Tuple[List[Dict], List[float]]:
        """
        Combine labeled and pseudo-labeled data with sample weights.
        
        Args:
            labeled_data: Original labeled annotations
            pseudo_labels: Pseudo-labeled annotations
            pseudo_weight: Weight for pseudo-labeled samples (0.0 to 1.0)
        
        Returns:
            Tuple of (combined_data, sample_weights)
        """
        combined_data = []
        sample_weights = []
        
        # Add labeled data with weight 1.0
        for label in labeled_data:
            combined_data.append(label)
            sample_weights.append(1.0)
        
        # Add pseudo-labeled data with reduced weight
        for pseudo in pseudo_labels:
            combined_data.append(pseudo)
            # Weight based on confidence
            confidence = pseudo.get('meta', {}).get('confidence', 0.5)
            weight = pseudo_weight * confidence
            sample_weights.append(weight)
        
        logger.info(f"Combined {len(labeled_data)} labeled + {len(pseudo_labels)} pseudo-labeled "
                   f"= {len(combined_data)} total samples")
        
        return combined_data, sample_weights
    
    def save_pseudo_labels(self, pseudo_labels: List[Dict], output_path: Path):
        """Save pseudo-labels to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for label in pseudo_labels:
                f.write(json.dumps(label, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(pseudo_labels)} pseudo-labels to {output_path}")


def create_semi_supervised_dataset(labeled_annotations: List[Dict],
                                   unlabeled_receipts_dir: Path,
                                   ocr_pipeline,
                                   pattern_extractor,
                                   output_dir: Path,
                                   max_pseudo_labels: Optional[int] = None) -> Tuple[List[Dict], List[float]]:
    """
    Main function to create semi-supervised dataset.
    
    Args:
        labeled_annotations: Original labeled annotations
        unlabeled_receipts_dir: Directory containing unlabeled receipts
        ocr_pipeline: OCR pipeline instance
        pattern_extractor: Pattern extraction function
        output_dir: Directory to save results
        max_pseudo_labels: Maximum number of pseudo-labels to generate
    
    Returns:
        Tuple of (combined_annotations, sample_weights)
    """
    trainer = SemiSupervisedTrainer(
        confidence_threshold=0.85,
        min_agreement=2
    )
    
    # Get unlabeled receipts
    unlabeled_receipts = []
    for ext in ['*.pdf', '*.png', '*.jpg', '*.jpeg']:
        unlabeled_receipts.extend(unlabeled_receipts_dir.glob(ext))
    
    logger.info(f"Found {len(unlabeled_receipts)} unlabeled receipts")
    
    # Generate pseudo-labels
    pseudo_labels = trainer.generate_pseudo_labels(
        unlabeled_receipts,
        ocr_pipeline,
        pattern_extractor
    )
    
    # Filter and rank
    pseudo_labels = trainer.filter_and_rank_pseudo_labels(
        pseudo_labels,
        top_k=max_pseudo_labels
    )
    
    # Save pseudo-labels
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pseudo_labels(
        pseudo_labels,
        output_dir / 'pseudo_labels.jsonl'
    )
    
    # Combine with labeled data
    combined_data, sample_weights = trainer.combine_labeled_and_pseudo(
        labeled_annotations,
        pseudo_labels,
        pseudo_weight=0.5
    )
    
    # Save combined dataset
    with open(output_dir / 'combined_annotations.jsonl', 'w', encoding='utf-8') as f:
        for ann in combined_data:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    
    # Save sample weights
    np.save(output_dir / 'sample_weights.npy', np.array(sample_weights))
    
    logger.info(f"Semi-supervised dataset created: {len(combined_data)} total samples")
    logger.info(f"  - Labeled: {len(labeled_annotations)}")
    logger.info(f"  - Pseudo-labeled: {len(pseudo_labels)}")
    
    return combined_data, sample_weights
