# improved_scalebar_classifier.py
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImprovedScaleBarClassifier:
    def __init__(self, 
                 model_name: str = "facebook/dinov2-base", 
                 device: str = None,
                 use_mid_layers: bool = True,
                 mid_layer_indices: List[int] = [4, 6, 8]):
        """Initialize the ImprovedScaleBarClassifier.

        Args:
            model_name (str): The name of the model to use.
            device (str): The device to run the model on.
            use_mid_layers (bool): Whether to use mid-layer features instead of final layer.
            mid_layer_indices (List[int]): Which transformer layers to extract features from.
        """
        self.device = device or (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu")
        
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.use_mid_layers = use_mid_layers
        self.mid_layer_indices = mid_layer_indices
        
        self.known_embs = {}
        self.embedding_dir = Path("embeddings")
        self.embedding_dir.mkdir(exist_ok=True)
        
        # Patch sliding parameters
        self.patch_size = (224, 224)  # Standard size for vision transformers
        self.stride = 112  # 50% overlap
        
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract features from the model, using mid-layers if specified.
        
        Args:
            image (Image.Image): PIL Image to process.
            
        Returns:
            torch.Tensor: Extracted features.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
            
            if self.use_mid_layers:
                # Extract and combine features from multiple mid-layers
                hidden_states = outputs.hidden_states
                mid_features = []
                
                for layer_idx in self.mid_layer_indices:
                    if layer_idx < len(hidden_states):
                        # For ViT models, skip CLS token and average patch tokens
                        layer_features = hidden_states[layer_idx][:, 1:].mean(dim=1)
                        mid_features.append(layer_features)
                
                if mid_features:
                    # Concatenate features from different layers
                    combined_features = torch.cat(mid_features, dim=-1)
                else:
                    # Fallback to final layer
                    combined_features = outputs.last_hidden_state.mean(dim=1)
            else:
                # Use final layer (original approach)
                combined_features = outputs.last_hidden_state.mean(dim=1)
                
        return combined_features / combined_features.norm(dim=-1, keepdim=True)

    def extract_patches(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int]]]:
        """Extract patches from an image using sliding window.
        
        Args:
            image (Image.Image): Input image.
            
        Returns:
            List[Tuple[Image.Image, Tuple[int, int]]]: List of (patch, (x, y)) tuples.
        """
        width, height = image.size
        patches = []
        
        for y in range(0, height - self.patch_size[1] + 1, self.stride):
            for x in range(0, width - self.patch_size[0] + 1, self.stride):
                # Extract patch
                patch = image.crop((x, y, x + self.patch_size[0], y + self.patch_size[1]))
                patches.append((patch, (x, y)))
                
        return patches

    def compute_patch_embeddings(self, image_path: str) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Compute embeddings for all patches in an image.
        
        Args:
            image_path (str): Path to the image.
            
        Returns:
            Tuple[torch.Tensor, List[Tuple[int, int]]]: Embeddings and patch positions.
        """
        image = Image.open(image_path).convert("RGB")
        patches = self.extract_patches(image)
        
        embeddings = []
        positions = []
        
        for patch, pos in patches:
            emb = self.extract_features(patch)
            embeddings.append(emb)
            positions.append(pos)
            
        return torch.cat(embeddings, dim=0), positions

    def compute_reference_embedding(self, image_path: str) -> torch.Tensor:
        """Compute embedding for a reference image (assumed to be already cropped scale bar).
        
        Args:
            image_path (str): Path to the reference image.
            
        Returns:
            torch.Tensor: Normalized embedding.
        """
        image = Image.open(image_path).convert("RGB")
        return self.extract_features(image)

    def load_or_compute_known_embeddings(self, ref_dir: str = "atypical_examples"):
        """Load or compute embeddings for known reference images.
        
        Args:
            ref_dir (str): Directory containing subdirectories of reference images.
        """
        ref_dir = Path(ref_dir)
        for category_dir in ref_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            # Use improved embedding file naming to distinguish methods
            method_suffix = "mid_layers" if self.use_mid_layers else "final_layer"
            emb_path = self.embedding_dir / f"{category_dir.name}_{method_suffix}.pt"
            
            if emb_path.exists():
                print(f"Loaded precomputed embeddings for {category_dir.name} ({method_suffix})")
                self.known_embs[category_dir.name] = torch.load(emb_path, map_location=self.device)
            else:
                print(f"Computing embeddings for {category_dir.name} using {method_suffix}...")
                embs = []
                for img_path in category_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        emb = self.compute_reference_embedding(img_path)
                        embs.append(emb)
                
                if embs:
                    embs = torch.cat(embs, dim=0)
                    torch.save(embs, emb_path)
                    self.known_embs[category_dir.name] = embs

    def classify_image_patch_based(self, 
                                   image_path: str, 
                                   threshold: float = 0.35,
                                   top_k_patches: int = 5,
                                   visualize: bool = False) -> Dict:
        """Classify image using patch-based approach.
        
        Args:
            image_path (str): Path to the image to classify.
            threshold (float): Similarity threshold.
            top_k_patches (int): Number of top patches to consider for final decision.
            visualize (bool): Whether to create visualization.
            
        Returns:
            Dict: Classification results with patch-level details.
        """
        # Compute embeddings for all patches
        patch_embeddings, patch_positions = self.compute_patch_embeddings(image_path)
        
        # Compare each patch to known categories
        patch_scores = []
        all_category_scores = {cat: [] for cat in self.known_embs.keys()}
        
        for i, patch_emb in enumerate(patch_embeddings):
            patch_result = {"position": patch_positions[i], "scores": {}}
            
            for cat, refs in self.known_embs.items():
                sims = torch.cosine_similarity(patch_emb.unsqueeze(0), refs).cpu().numpy()
                max_sim = float(np.max(sims))
                patch_result["scores"][cat] = max_sim
                all_category_scores[cat].append(max_sim)
            
            patch_scores.append(patch_result)
        
        # Aggregate results: use top-k patches for each category
        final_scores = {}
        best_patches = {}
        
        for cat in self.known_embs.keys():
            cat_scores = all_category_scores[cat]
            # Get top-k scores for this category
            top_scores = sorted(cat_scores, reverse=True)[:top_k_patches]
            final_scores[cat] = np.mean(top_scores) if top_scores else 0.0
            
            # Find best patch position for this category
            best_patch_idx = np.argmax(cat_scores)
            best_patches[cat] = {
                "position": patch_positions[best_patch_idx],
                "score": cat_scores[best_patch_idx]
            }
        
        # Make final prediction
        best_category = max(final_scores, key=final_scores.get)
        predicted_category = best_category if final_scores[best_category] > threshold else "normal"
        
        result = {
            "predicted_category": predicted_category,
            "aggregated_scores": final_scores,
            "best_patches": best_patches,
            "patch_details": patch_scores,
            "confidence": final_scores[best_category] if best_category in final_scores else 0.0
        }
        
        if visualize:
            self._visualize_results(image_path, result)
        
        return result

    def _visualize_results(self, image_path: str, result: Dict):
        """Visualize classification results with patch locations."""
        image = Image.open(image_path).convert("RGB")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Image with best patch locations
        ax2.imshow(image)
        ax2.set_title(f"Predicted: {result['predicted_category']} "
                      f"(conf: {result['confidence']:.3f})")
        
        # Draw rectangles for best patches of each category
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (cat, patch_info) in enumerate(result['best_patches'].items()):
            if patch_info['score'] > 0.2:  # Only show significant patches
                x, y = patch_info['position']
                rect = patches.Rectangle((x, y), self.patch_size[0], self.patch_size[1],
                                       linewidth=2, edgecolor=colors[i % len(colors)], 
                                       facecolor='none', alpha=0.8)
                ax2.add_patch(rect)
                ax2.text(x, y-10, f"{cat}: {patch_info['score']:.3f}", 
                        color=colors[i % len(colors)], fontweight='bold')
        
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    def classify_image_region_based(self, 
                                    image_path: str, 
                                    regions: List[Tuple[int, int, int, int]] = None,
                                    threshold: float = 0.35) -> Dict:
        """Classify image by focusing on specific regions (e.g., bottom area for scale bars).
        
        Args:
            image_path (str): Path to the image.
            regions (List[Tuple[int, int, int, int]]): List of (x1, y1, x2, y2) regions to check.
                If None, will use bottom 20% of image.
            threshold (float): Similarity threshold.
            
        Returns:
            Dict: Classification results.
        """
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        if regions is None:
            # Default: check bottom 20% of image
            regions = [(0, int(0.8 * height), width, height)]
        
        region_results = []
        
        for region in regions:
            x1, y1, x2, y2 = region
            cropped = image.crop((x1, y1, x2, y2))
            
            # Resize to standard input size
            cropped = cropped.resize(self.patch_size)
            region_emb = self.extract_features(cropped)
            
            # Compare to known categories
            scores = {}
            for cat, refs in self.known_embs.items():
                sims = torch.cosine_similarity(region_emb, refs).cpu().numpy()
                scores[cat] = float(np.max(sims))
            
            region_results.append({
                "region": region,
                "scores": scores
            })
        
        # Aggregate results from all regions
        final_scores = {}
        for cat in self.known_embs.keys():
            cat_scores = [r["scores"][cat] for r in region_results]
            final_scores[cat] = max(cat_scores) if cat_scores else 0.0
        
        best_category = max(final_scores, key=final_scores.get)
        predicted_category = best_category if final_scores[best_category] > threshold else "normal"
        
        return {
            "predicted_category": predicted_category,
            "aggregated_scores": final_scores,
            "region_details": region_results,
            "confidence": final_scores[best_category]
        }

    def compare_methods(self, image_path: str, visualize: bool = True) -> Dict:
        """Compare different classification methods on the same image.
        
        Args:
            image_path (str): Path to the image.
            visualize (bool): Whether to show comparison visualization.
            
        Returns:
            Dict: Comparison results from different methods.
        """
        results = {}
        
        # Method 1: Original global embedding (for comparison)
        image = Image.open(image_path).convert("RGB")
        
        # Temporarily disable mid-layer extraction for comparison
        original_use_mid = self.use_mid_layers
        self.use_mid_layers = False
        global_emb = self.extract_features(image)
        
        scores = {}
        for cat, refs in self.known_embs.items():
            sims = torch.cosine_similarity(global_emb, refs).cpu().numpy()
            scores[cat] = float(np.max(sims))
        
        best_cat = max(scores, key=scores.get)
        results["global_final_layer"] = {
            "predicted_category": best_cat if scores[best_cat] > 0.27 else "normal",
            "scores": scores
        }

        
        # Restore mid-layer setting and method 2: Global embedding with mid-layers  
        self.use_mid_layers = original_use_mid
        
        if self.use_mid_layers:
            mid_layer_emb = self.extract_features(image)
            scores = {}
            for cat, refs in self.known_embs.items():
                sims = torch.cosine_similarity(mid_layer_emb, refs).cpu().numpy()
                scores[cat] = float(np.max(sims))
            
            best_cat = max(scores, key=scores.get)
            results["global_mid_layers"] = {
                "predicted_category": best_cat if scores[best_cat] > 0.35 else "normal",
                "scores": scores
            }
        
        # Method 3: Patch-based
        results["patch_based"] = self.classify_image_patch_based(image_path, visualize=False)
        
        # Method 4: Region-based
        results["region_based"] = self.classify_image_region_based(image_path)
        
        if visualize:
            self._visualize_method_comparison(image_path, results)
        
        return results

    def _visualize_method_comparison(self, image_path: str, results: Dict):
        """Visualize comparison between different methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        image = Image.open(image_path)
        
        methods = ["global_final_layer", "global_mid_layers", "patch_based", "region_based"]
        titles = ["Global (Final Layer)", "Global (Mid Layers)", "Patch-Based", "Region-Based"]
        
        for i, (method, title) in enumerate(zip(methods, titles)):
            if method in results:
                axes[i].imshow(image)
                result = results[method]
                pred = result["predicted_category"]
                conf = result.get("confidence", max(result["scores"].values()))
                axes[i].set_title(f"{title}\nPred: {pred} (conf: {conf:.3f})")
            else:
                axes[i].text(0.5, 0.5, f"{method}\nnot available", 
                           ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()