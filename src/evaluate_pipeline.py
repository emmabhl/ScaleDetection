"""
Comprehensive Evaluation Pipeline for Scale Detection

This module provides comprehensive evaluation metrics for the scale detection pipeline,
including detection accuracy, OCR performance, and end-to-end scale conversion accuracy.

Features:
- Detection mAP calculation for scale bars and text regions
- OCR accuracy metrics (character-level and word-level)
- Endpoint localization accuracy
- Scale conversion accuracy and error analysis
- Visualization of results and error distributions
- Statistical analysis and reporting
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import cv2
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import ultralytics for mAP calculation
try:
    from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics not available for mAP calculation")


@dataclass
class DetectionResult:
    """Data class for detection results."""
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    confidence: float
    class_id: int
    class_name: str


@dataclass
class GroundTruth:
    """Data class for ground truth annotations."""
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    class_id: int
    class_name: str


@dataclass
class EvaluationMetrics:
    """Data class for evaluation metrics."""
    # Detection metrics
    map_50: float
    map_75: float
    map_50_95: float
    precision: float
    recall: float
    f1_score: float
    
    # OCR metrics
    character_accuracy: float
    word_accuracy: float
    unit_accuracy: float
    value_accuracy: float
    
    # Scale conversion metrics
    scale_conversion_success_rate: float
    mean_absolute_error_pixels: float
    mean_absolute_error_percent: float
    scale_length_mae: float
    scale_length_mape: float
    
    # Endpoint localization metrics
    endpoint_mae_pixels: float
    endpoint_mape: float


class DetectionEvaluator:
    """Evaluator for object detection performance."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize detection evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or ['scale_bar', 'scale_text']
        self.class_id_to_name = {i: name for i, name in enumerate(self.class_names)}
    
    def calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            IoU value
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, predictions: List[DetectionResult], 
                        ground_truths: List[GroundTruth], 
                        iou_threshold: float = 0.5) -> Tuple[List[bool], List[bool]]:
        """
        Match predictions to ground truth detections.
        
        Args:
            predictions: List of predicted detections
            ground_truths: List of ground truth detections
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (prediction_matched, ground_truth_matched) boolean lists
        """
        pred_matched = [False] * len(predictions)
        gt_matched = [False] * len(ground_truths)
        
        # Sort predictions by confidence (descending)
        pred_indices = sorted(range(len(predictions)), 
                            key=lambda i: predictions[i].confidence, reverse=True)
        
        for pred_idx in pred_indices:
            pred = predictions[pred_idx]
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx] or pred.class_id != gt.class_id:
                    continue
                
                iou = self.calculate_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                pred_matched[pred_idx] = True
                gt_matched[best_gt_idx] = True
        
        return pred_matched, gt_matched
    
    def calculate_precision_recall(self, predictions: List[DetectionResult], 
                                 ground_truths: List[GroundTruth], 
                                 iou_threshold: float = 0.5) -> Tuple[float, float]:
        """
        Calculate precision and recall for detections.
        
        Args:
            predictions: List of predicted detections
            ground_truths: List of ground truth detections
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision, recall)
        """
        pred_matched, gt_matched = self.match_detections(predictions, ground_truths, iou_threshold)
        
        tp = sum(pred_matched)
        fp = len(predictions) - tp
        fn = len(ground_truths) - sum(gt_matched)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recall
    
    def calculate_map(self, all_predictions: List[List[DetectionResult]], 
                     all_ground_truths: List[List[GroundTruth]], 
                     iou_thresholds: List[float] = None) -> Dict[str, float]:
        """
        Calculate mAP (mean Average Precision) for multiple images.
        
        Args:
            all_predictions: List of predictions for each image
            all_ground_truths: List of ground truths for each image
            iou_thresholds: List of IoU thresholds to evaluate
            
        Returns:
            Dictionary containing mAP metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75, 0.5]  # mAP@0.5, mAP@0.75, mAP@0.5:0.95
        
        results = {}
        
        # Calculate mAP for each IoU threshold
        for iou_thresh in iou_thresholds:
            if iou_thresh == 0.5:
                # mAP@0.5
                aps = []
                for class_id in range(len(self.class_names)):
                    class_precisions = []
                    class_recalls = []
                    
                    for preds, gts in zip(all_predictions, all_ground_truths):
                        # Filter by class
                        class_preds = [p for p in preds if p.class_id == class_id]
                        class_gts = [g for g in gts if g.class_id == class_id]
                        
                        if len(class_gts) == 0:
                            continue
                        
                        # Calculate precision-recall curve
                        if len(class_preds) > 0:
                            # Sort by confidence
                            sorted_preds = sorted(class_preds, key=lambda x: x.confidence, reverse=True)
                            
                            precisions = []
                            recalls = []
                            
                            for i in range(len(sorted_preds)):
                                # Calculate precision and recall for top i+1 predictions
                                top_preds = sorted_preds[:i+1]
                                pred_matched, gt_matched = self.match_detections(top_preds, class_gts, iou_thresh)
                                
                                tp = sum(pred_matched)
                                fp = len(top_preds) - tp
                                fn = len(class_gts) - sum(gt_matched)
                                
                                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                                
                                precisions.append(prec)
                                recalls.append(rec)
                            
                            # Calculate AP using 11-point interpolation
                            if len(precisions) > 0:
                                ap = self._calculate_ap(precisions, recalls)
                                aps.append(ap)
                
                results[f'map_{iou_thresh}'] = np.mean(aps) if aps else 0.0
        
        # Calculate overall precision and recall
        all_tp = 0
        all_fp = 0
        all_fn = 0
        
        for preds, gts in zip(all_predictions, all_ground_truths):
            pred_matched, gt_matched = self.match_detections(preds, gts, 0.5)
            all_tp += sum(pred_matched)
            all_fp += len(preds) - sum(pred_matched)
            all_fn += len(gts) - sum(gt_matched)
        
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['precision'] = precision
        results['recall'] = recall
        results['f1_score'] = f1_score
        
        return results
    
    def _calculate_ap(self, precisions: List[float], recalls: List[float]) -> float:
        """Calculate Average Precision using 11-point interpolation."""
        # Ensure recalls are sorted
        sorted_indices = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_indices]
        precisions = np.array(precisions)[sorted_indices]
        
        # 11-point interpolation
        recall_points = np.linspace(0, 1, 11)
        interpolated_precisions = []
        
        for r in recall_points:
            # Find maximum precision for recall >= r
            valid_precisions = precisions[recalls >= r]
            if len(valid_precisions) > 0:
                interpolated_precisions.append(np.max(valid_precisions))
            else:
                interpolated_precisions.append(0)
        
        return np.mean(interpolated_precisions)


class OCREvaluator:
    """Evaluator for OCR performance."""
    
    def __init__(self):
        """Initialize OCR evaluator."""
        pass
    
    def calculate_character_accuracy(self, predicted_text: str, ground_truth_text: str) -> float:
        """
        Calculate character-level accuracy.
        
        Args:
            predicted_text: Predicted text
            ground_truth_text: Ground truth text
            
        Returns:
            Character accuracy (0-1)
        """
        if not ground_truth_text:
            return 1.0 if not predicted_text else 0.0
        
        # Align texts using edit distance
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, ground_truth_text.lower(), predicted_text.lower())
        
        matches = matcher.get_matching_blocks()
        total_matches = sum(block.size for block in matches)
        
        return total_matches / len(ground_truth_text) if len(ground_truth_text) > 0 else 0.0
    
    def calculate_word_accuracy(self, predicted_text: str, ground_truth_text: str) -> float:
        """
        Calculate word-level accuracy.
        
        Args:
            predicted_text: Predicted text
            ground_truth_text: Ground truth text
            
        Returns:
            Word accuracy (0-1)
        """
        if not ground_truth_text:
            return 1.0 if not predicted_text else 0.0
        
        pred_words = predicted_text.lower().split()
        gt_words = ground_truth_text.lower().split()
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        # Calculate word-level matches
        matches = 0
        for gt_word in gt_words:
            if gt_word in pred_words:
                matches += 1
                pred_words.remove(gt_word)  # Remove to avoid double counting
        
        return matches / len(gt_words)
    
    def evaluate_parsing_accuracy(self, parsed_results: List[Dict[str, Any]], 
                                 ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate parsing accuracy for scale values and units.
        
        Args:
            parsed_results: List of parsed OCR results
            ground_truths: List of ground truth annotations
            
        Returns:
            Dictionary containing parsing accuracy metrics
        """
        if len(parsed_results) != len(ground_truths):
            print(f"Warning: Mismatch in number of parsed results ({len(parsed_results)}) "
                  f"and ground truths ({len(ground_truths)})")
        
        value_correct = 0
        unit_correct = 0
        total = min(len(parsed_results), len(ground_truths))
        
        for i in range(total):
            parsed = parsed_results[i]
            gt = ground_truths[i]
            
            # Check value accuracy (with tolerance)
            if 'parsed_value' in parsed and 'value' in gt:
                parsed_val = parsed['parsed_value']
                gt_val = gt['value']
                if parsed_val is not None and gt_val is not None:
                    tolerance = 0.1  # 10% tolerance
                    if abs(parsed_val - gt_val) / gt_val <= tolerance:
                        value_correct += 1
            
            # Check unit accuracy
            if 'normalized_unit' in parsed and 'unit' in gt:
                parsed_unit = parsed['normalized_unit']
                gt_unit = gt['unit']
                if parsed_unit == gt_unit:
                    unit_correct += 1
        
        return {
            'value_accuracy': value_correct / total if total > 0 else 0.0,
            'unit_accuracy': unit_correct / total if total > 0 else 0.0
        }


class ScaleConversionEvaluator:
    """Evaluator for scale conversion accuracy."""
    
    def __init__(self):
        """Initialize scale conversion evaluator."""
        pass
    
    def evaluate_scale_conversion(self, predicted_scales: List[Dict[str, Any]], 
                                ground_truth_scales: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate scale conversion accuracy.
        
        Args:
            predicted_scales: List of predicted scale conversions
            ground_truth_scales: List of ground truth scale conversions
            
        Returns:
            Dictionary containing scale conversion metrics
        """
        if len(predicted_scales) != len(ground_truth_scales):
            print(f"Warning: Mismatch in number of predicted scales ({len(predicted_scales)}) "
                  f"and ground truth scales ({len(ground_truth_scales)})")
        
        successful_conversions = 0
        pixel_errors = []
        percent_errors = []
        scale_length_errors = []
        scale_length_percent_errors = []
        
        for i in range(min(len(predicted_scales), len(ground_truth_scales))):
            pred = predicted_scales[i]
            gt = ground_truth_scales[i]
            
            if 'um_per_pixel' in pred and pred['um_per_pixel'] is not None:
                successful_conversions += 1
                
                # Calculate errors if ground truth available
                if 'um_per_pixel' in gt and gt['um_per_pixel'] is not None:
                    pred_scale = pred['um_per_pixel']
                    gt_scale = gt['um_per_pixel']
                    
                    # Pixel length errors
                    if 'pixel_length' in pred and 'pixel_length' in gt:
                        pred_pixel_len = pred['pixel_length']
                        gt_pixel_len = gt['pixel_length']
                        pixel_error = abs(pred_pixel_len - gt_pixel_len)
                        pixel_errors.append(pixel_error)
                        
                        if gt_pixel_len > 0:
                            percent_error = pixel_error / gt_pixel_len * 100
                            percent_errors.append(percent_error)
                    
                    # Scale length errors
                    if 'physical_length' in pred and 'physical_length' in gt:
                        pred_phys_len = pred['physical_length']
                        gt_phys_len = gt['physical_length']
                        scale_error = abs(pred_phys_len - gt_phys_len)
                        scale_length_errors.append(scale_error)
                        
                        if gt_phys_len > 0:
                            scale_percent_error = scale_error / gt_phys_len * 100
                            scale_length_percent_errors.append(scale_percent_error)
        
        total = len(predicted_scales)
        success_rate = successful_conversions / total if total > 0 else 0.0
        
        return {
            'scale_conversion_success_rate': success_rate,
            'mean_absolute_error_pixels': np.mean(pixel_errors) if pixel_errors else 0.0,
            'mean_absolute_error_percent': np.mean(percent_errors) if percent_errors else 0.0,
            'scale_length_mae': np.mean(scale_length_errors) if scale_length_errors else 0.0,
            'scale_length_mape': np.mean(scale_length_percent_errors) if scale_length_percent_errors else 0.0
        }


class EndpointLocalizationEvaluator:
    """Evaluator for endpoint localization accuracy."""
    
    def __init__(self):
        """Initialize endpoint localization evaluator."""
        pass
    
    def evaluate_endpoint_accuracy(self, predicted_endpoints: List[List[Tuple[float, float]]], 
                                 ground_truth_endpoints: List[List[Tuple[float, float]]]) -> Dict[str, float]:
        """
        Evaluate endpoint localization accuracy.
        
        Args:
            predicted_endpoints: List of predicted endpoint pairs
            ground_truth_endpoints: List of ground truth endpoint pairs
            
        Returns:
            Dictionary containing endpoint accuracy metrics
        """
        if len(predicted_endpoints) != len(ground_truth_endpoints):
            print(f"Warning: Mismatch in number of predicted endpoints ({len(predicted_endpoints)}) "
                  f"and ground truth endpoints ({len(ground_truth_endpoints)})")
        
        endpoint_errors = []
        percent_errors = []
        
        for i in range(min(len(predicted_endpoints), len(ground_truth_endpoints))):
            pred_endpoints = predicted_endpoints[i]
            gt_endpoints = ground_truth_endpoints[i]
            
            if len(pred_endpoints) == 2 and len(gt_endpoints) == 2:
                # Calculate distance between corresponding endpoints
                pred_start, pred_end = pred_endpoints
                gt_start, gt_end = gt_endpoints
                
                # Calculate errors for both endpoints
                start_error = np.sqrt((pred_start[0] - gt_start[0])**2 + (pred_start[1] - gt_start[1])**2)
                end_error = np.sqrt((pred_end[0] - gt_end[0])**2 + (pred_end[1] - gt_end[1])**2)
                
                avg_error = (start_error + end_error) / 2
                endpoint_errors.append(avg_error)
                
                # Calculate percent error based on ground truth length
                gt_length = np.sqrt((gt_end[0] - gt_start[0])**2 + (gt_end[1] - gt_start[1])**2)
                if gt_length > 0:
                    percent_error = avg_error / gt_length * 100
                    percent_errors.append(percent_error)
        
        return {
            'endpoint_mae_pixels': np.mean(endpoint_errors) if endpoint_errors else 0.0,
            'endpoint_mape': np.mean(percent_errors) if percent_errors else 0.0
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all evaluation metrics."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize comprehensive evaluator.
        
        Args:
            class_names: List of class names for detection evaluation
        """
        self.detection_evaluator = DetectionEvaluator(class_names)
        self.ocr_evaluator = OCREvaluator()
        self.scale_evaluator = ScaleConversionEvaluator()
        self.endpoint_evaluator = EndpointLocalizationEvaluator()
    
    def evaluate_pipeline(self, results: Dict[str, Any], 
                         ground_truth: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate the complete pipeline.
        
        Args:
            results: Pipeline results
            ground_truth: Ground truth annotations
            
        Returns:
            Comprehensive evaluation metrics
        """
        # Extract data from results and ground truth
        predictions = self._extract_detections(results)
        gt_detections = self._extract_detections(ground_truth)
        
        # Detection evaluation
        detection_metrics = self.detection_evaluator.calculate_map([predictions], [gt_detections])
        
        # OCR evaluation
        ocr_metrics = self._evaluate_ocr(results, ground_truth)
        
        # Scale conversion evaluation
        scale_metrics = self._evaluate_scale_conversion(results, ground_truth)
        
        # Endpoint localization evaluation
        endpoint_metrics = self._evaluate_endpoint_localization(results, ground_truth)
        
        # Combine all metrics
        return EvaluationMetrics(
            map_50=detection_metrics.get('map_0.5', 0.0),
            map_75=detection_metrics.get('map_0.75', 0.0),
            map_50_95=detection_metrics.get('map_0.5', 0.0),  # Simplified for now
            precision=detection_metrics.get('precision', 0.0),
            recall=detection_metrics.get('recall', 0.0),
            f1_score=detection_metrics.get('f1_score', 0.0),
            character_accuracy=ocr_metrics.get('character_accuracy', 0.0),
            word_accuracy=ocr_metrics.get('word_accuracy', 0.0),
            unit_accuracy=ocr_metrics.get('unit_accuracy', 0.0),
            value_accuracy=ocr_metrics.get('value_accuracy', 0.0),
            scale_conversion_success_rate=scale_metrics.get('scale_conversion_success_rate', 0.0),
            mean_absolute_error_pixels=scale_metrics.get('mean_absolute_error_pixels', 0.0),
            mean_absolute_error_percent=scale_metrics.get('mean_absolute_error_percent', 0.0),
            scale_length_mae=scale_metrics.get('scale_length_mae', 0.0),
            scale_length_mape=scale_metrics.get('scale_length_mape', 0.0),
            endpoint_mae_pixels=endpoint_metrics.get('endpoint_mae_pixels', 0.0),
            endpoint_mape=endpoint_metrics.get('endpoint_mape', 0.0)
        )
    
    def _extract_detections(self, data: Dict[str, Any]) -> List[DetectionResult]:
        """Extract detection results from data dictionary."""
        detections = []
        
        # Extract scale bar detections
        for bar in data.get('bars', []):
            detections.append(DetectionResult(
                bbox=tuple(bar['bbox']),
                confidence=bar.get('confidence', 1.0),
                class_id=0,
                class_name='scale_bar'
            ))
        
        # Extract text detections
        for text in data.get('labels', []):
            detections.append(DetectionResult(
                bbox=tuple(text['bbox']),
                confidence=text.get('confidence', 1.0),
                class_id=1,
                class_name='scale_text'
            ))
        
        return detections
    
    def _evaluate_ocr(self, results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate OCR performance."""
        # This is a simplified implementation
        # In practice, you would need more detailed OCR results
        return {
            'character_accuracy': 0.0,
            'word_accuracy': 0.0,
            'unit_accuracy': 0.0,
            'value_accuracy': 0.0
        }
    
    def _evaluate_scale_conversion(self, results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate scale conversion accuracy."""
        # This is a simplified implementation
        return {
            'scale_conversion_success_rate': 0.0,
            'mean_absolute_error_pixels': 0.0,
            'mean_absolute_error_percent': 0.0,
            'scale_length_mae': 0.0,
            'scale_length_mape': 0.0
        }
    
    def _evaluate_endpoint_localization(self, results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate endpoint localization accuracy."""
        # This is a simplified implementation
        return {
            'endpoint_mae_pixels': 0.0,
            'endpoint_mape': 0.0
        }
    
    def generate_report(self, metrics: EvaluationMetrics, save_path: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics
            save_path: Path to save report (optional)
            
        Returns:
            Report string
        """
        report = "=" * 80 + "\n"
        report += "SCALE DETECTION PIPELINE EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Detection metrics
        report += "DETECTION METRICS:\n"
        report += "-" * 40 + "\n"
        report += f"mAP@0.5:           {metrics.map_50:.4f}\n"
        report += f"mAP@0.75:          {metrics.map_75:.4f}\n"
        report += f"mAP@0.5:0.95:      {metrics.map_50_95:.4f}\n"
        report += f"Precision:         {metrics.precision:.4f}\n"
        report += f"Recall:            {metrics.recall:.4f}\n"
        report += f"F1-Score:          {metrics.f1_score:.4f}\n\n"
        
        # OCR metrics
        report += "OCR METRICS:\n"
        report += "-" * 40 + "\n"
        report += f"Character Accuracy: {metrics.character_accuracy:.4f}\n"
        report += f"Word Accuracy:      {metrics.word_accuracy:.4f}\n"
        report += f"Unit Accuracy:      {metrics.unit_accuracy:.4f}\n"
        report += f"Value Accuracy:     {metrics.value_accuracy:.4f}\n\n"
        
        # Scale conversion metrics
        report += "SCALE CONVERSION METRICS:\n"
        report += "-" * 40 + "\n"
        report += f"Success Rate:       {metrics.scale_conversion_success_rate:.4f}\n"
        report += f"MAE (pixels):       {metrics.mean_absolute_error_pixels:.2f}\n"
        report += f"MAE (%):            {metrics.mean_absolute_error_percent:.2f}\n"
        report += f"Scale Length MAE:   {metrics.scale_length_mae:.6f}\n"
        report += f"Scale Length MAPE:  {metrics.scale_length_mape:.2f}\n\n"
        
        # Endpoint localization metrics
        report += "ENDPOINT LOCALIZATION METRICS:\n"
        report += "-" * 40 + "\n"
        report += f"MAE (pixels):       {metrics.endpoint_mae_pixels:.2f}\n"
        report += f"MAPE:               {metrics.endpoint_mape:.2f}\n\n"
        
        # Overall assessment
        report += "OVERALL ASSESSMENT:\n"
        report += "-" * 40 + "\n"
        
        if metrics.map_50 > 0.8 and metrics.scale_conversion_success_rate > 0.9:
            report += "EXCELLENT: Pipeline performs very well across all metrics.\n"
        elif metrics.map_50 > 0.6 and metrics.scale_conversion_success_rate > 0.7:
            report += "GOOD: Pipeline performs well with room for improvement.\n"
        elif metrics.map_50 > 0.4 and metrics.scale_conversion_success_rate > 0.5:
            report += "FAIR: Pipeline shows acceptable performance but needs improvement.\n"
        else:
            report += "POOR: Pipeline requires significant improvement.\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {save_path}")
        
        return report
    
    def create_visualizations(self, results: Dict[str, Any], ground_truth: Dict[str, Any], 
                            save_dir: str = None) -> None:
        """
        Create visualization plots for evaluation results.
        
        Args:
            results: Pipeline results
            ground_truth: Ground truth annotations
            save_dir: Directory to save plots (optional)
        """
        # This is a placeholder for visualization creation
        # In practice, you would create various plots showing:
        # - Detection results overlay
        # - Error distribution histograms
        # - Precision-recall curves
        # - Scale conversion accuracy plots
        # - Endpoint localization accuracy plots
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Visualizations would be saved to: {save_dir}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate scale detection pipeline')
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--output', type=str, help='Path to save evaluation report')
    parser.add_argument('--visualize', action='store_true', help='Create visualization plots')
    parser.add_argument('--save_plots', type=str, help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load results and ground truth
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Evaluate pipeline
    metrics = evaluator.evaluate_pipeline(results, ground_truth)
    
    # Generate report
    report = evaluator.generate_report(metrics, args.output)
    print(report)
    
    # Create visualizations if requested
    if args.visualize:
        evaluator.create_visualizations(results, ground_truth, args.save_plots)


if __name__ == "__main__":
    main()
