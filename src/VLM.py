from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os
from PIL import Image
from prompt import PROMPT_TEMPLATE
import argparse
from tqdm import tqdm
import json


def VLM_scale_detection(
    filepath: str, 
    output_folder: str,
    model_id: str, 
    max_side: int
) -> None:
    """
    Perform scale detection using a Vision-Language Model (VLM).
    
    Args:
        filepath (str): The path to the directory containing images for scale detection.
        model_id (str): The identifier of the pre-trained VLM model.
        max_side (int): The maximum side length for image resizing to manage memory usage.
    """
    
    # Load model and processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, 
        cache_dir="/scratch/eboehly/hf_cache",
        device_map="auto", 
        dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir="/scratch/eboehly/hf_cache"
    )

    # Function to resize image while preserving aspect ratio
    def resize_max_side(pil_img: Image.Image, max_side: int):
        w, h = pil_img.size
        if max(w, h) <= max_side:
            return pil_img
        # preserve aspect ratio
        if w >= h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        return pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

    # Process each image in the specified directory
    for filename in tqdm(os.listdir(filepath)):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue

        image_path = os.path.join(filepath, filename)
        image = Image.open(image_path).convert("RGB")

        # Resize to limit memory footprint (important!)
        image = resize_max_side(image, max_side)

        # Build messages same as before (image inserted by processor)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, # Placeholder for image, inserted by processor separately
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }
        ]

        # Use apply_chat_template on the messages only to get the text prompt
        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Create inputs including both image and text
        inputs = processor(images=[image], text=text, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0, # deterministic
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Trim prompt portion correctly using sequence lengths (remove prompt, keep only generated)
        input_ids = inputs["input_ids"]
        seq_lens = input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, seq_lens:]
        # Decode output to text
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # Keep only content before <END_JSON> token
        output_text = output_text.split("<END_JSON>")[0].strip()
        # Transform output text to dictionary
        output_text = json.loads(output_text)

        # Save the output in a json file in the specified output folder
        os.makedirs(output_folder, exist_ok=True)
        filename = filename.removesuffix(".jpg").removesuffix(".png")
        output_path = os.path.join(output_folder, filename + ".json")
        with open(output_path, "w") as f:
            json.dump(output_text, f, indent=2)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Scale Detection")
    parser.add_argument("--filepath", type=str, default="data/annot", help="Path to the directory containing images")
    parser.add_argument("--output_folder", type=str, default="outputs_vlm", help="Folder to save output JSON files")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Pre-trained VLM model ID")
    parser.add_argument("--max_side", type=int, default=2048, help="Maximum side length for image resizing")
    
    args = parser.parse_args()
    
    VLM_scale_detection(
        filepath=args.filepath,
        output_folder=args.output_folder,
        model_id=args.model_id,
        max_side=args.max_side
    )