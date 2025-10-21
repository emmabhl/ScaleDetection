from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os
from PIL import Image
from src.prompt import PROMPT_TEMPLATE
import argparse


def VLM_scale_detection(
    filepath: str, 
    output_folder: str,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct", 
    max_side: int = 1024
) -> None:
    """
    Perform scale detection using a Vision-Language Model (VLM).
    
    Args:
        filepath (str): The path to the directory containing images for scale detection.
        model_id (str): The identifier of the pre-trained VLM model.
        max_side (int): The maximum side length for image resizing to manage memory usage.
    """
    
    # Load model (with lower-precision to save memory)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, device_map="auto", dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)

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

    for filename in os.listdir(filepath):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue

        image_path = os.path.join(filepath, filename)
        image = Image.open(image_path).convert("RGB")

        # Resize to limit memory footprint (important!)
        image = resize_max_side(image, max_side)
        print(f"{filename} resized -> {image.size}")

        # Build messages same as before (image inserted by processor)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }
        ]

        # Option A: use apply_chat_template -> then processor(images=..., text=...)
        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Create inputs (pixel_values will be resized/normalized by the processor)
        # Note: processor returns a BatchEncoding; move tensors explicitly to model.device
        inputs = processor(images=[image], text=text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        # Trim prompt portion correctly using sequence lengths
        input_ids = inputs["input_ids"]
        # input_ids is shape (batch, seq_len)
        seq_lens = input_ids.shape[1]
        # For batch size 1:
        generated_ids_trimmed = generated_ids[:, seq_lens:]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Save the output in a json file in a new folder "outputs_vlm"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename + ".json")
        with open(output_path, "w") as f:
            import json
            json.dump(output_text, f, indent=2)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Scale Detection")
    parser.add_argument("--filepath", type=str, default="data/annot", help="Path to the directory containing images")
    parser.add_argument("--output_folder", type=str, default="outputs_vlm", help="Folder to save output JSON files")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Pre-trained VLM model ID")
    parser.add_argument("--max_side", type=int, default=1024, help="Maximum side length for image resizing")
    
    args = parser.parse_args()
    
    VLM_scale_detection(
        filepath=args.filepath,
        output_folder=args.output_folder,
        model_id=args.model_id,
        max_side=args.max_side
    )