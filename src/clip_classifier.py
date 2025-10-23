# scale_bar_classifier.py
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from pathlib import Path
import json

class ScaleBarClassifier:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        """Initialize the ScaleBarClassifier.

        Args:
            model_name (str, optional): The name of the model to use. 
                Defaults to "facebook/dinov2-base".
            device (str, optional): The device to run the model on. Defaults to None.
        """
        self.device = device or (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.known_embs = {}
        self.embedding_dir = Path("embeddings")
        self.embedding_dir.mkdir(exist_ok=True)

    # --- Embedding Computation ---
    def compute_image_embedding(self, image_path: str) -> torch.Tensor:
        """Compute the embedding for a given image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            torch.Tensor: The normalized image embedding.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).last_hidden_state.mean(dim=1)
        return emb / emb.norm(dim=-1, keepdim=True)

    # --- Precompute or load stored embeddings ---
    def load_or_compute_known_embeddings(self, ref_dir: str = "ref_images"):
        """Load or compute embeddings for known reference images.
        
        Args:
            ref_dir (str, optional): Directory containing subdirectories of reference images. 
                Each subdirectory is treated as a category. Defaults to "ref_images".
        """
        ref_dir = Path(ref_dir)
        for category_dir in ref_dir.iterdir():
            if not category_dir.is_dir():
                continue
            emb_path = self.embedding_dir / f"{category_dir.name}.pt"
            if emb_path.exists():
                print(f"Loaded precomputed embeddings for {category_dir.name}")
                self.known_embs[category_dir.name] = torch.load(emb_path, map_location=self.device)
            else:
                print(f"Computing embeddings for {category_dir.name}...")
                embs = []
                for img_path in category_dir.glob("*"):
                    embs.append(self.compute_image_embedding(img_path))
                embs = torch.cat(embs, dim=0)
                torch.save(embs, emb_path)
                self.known_embs[category_dir.name] = embs

    # --- Add new references dynamically ---
    def add_new_references(self, image_paths: list[str], category: str) -> None:
        """Add new reference images to a given category.
        
        Args:
            image_paths (list[str]): List of image file paths to add.
            category (str): The category to which the images belong.
        """
        embs = [self.compute_image_embedding(p) for p in image_paths]
        new_embs = torch.cat(embs, dim=0)
        if category in self.known_embs:
            self.known_embs[category] = torch.cat([self.known_embs[category], new_embs], dim=0)
        else:
            self.known_embs[category] = new_embs
        torch.save(self.known_embs[category], self.embedding_dir / f"{category}.pt")

    # --- Classification ---
    def classify_image(self, image_path: str, threshold: float = 0.27) -> dict:
        """Classify an image based on similarity to known categories.
        
        Args:
            image_path (str): The path to the image file to classify.
            threshold (float, optional): Similarity threshold to consider a match. Defaults to 0.27.
            
        Returns:
            dict: A dictionary with predicted category and similarity scores.
        """
        img_emb = self.compute_image_embedding(image_path)
        scores = {}
        for cat, refs in self.known_embs.items():
            sims = torch.cosine_similarity(img_emb, refs).cpu().numpy()
            scores[cat] = float(np.max(sims))
        best_cat = max(scores, key=scores.get)
        return {
            "predicted_category": best_cat if scores[best_cat] > threshold else "normal",
            "scores": scores
        }
