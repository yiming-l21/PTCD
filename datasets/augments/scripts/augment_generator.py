import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class SentimentDescriptionAugmenter:
    """Generate varied descriptions for sentiment analysis images"""
    
    PROMPT_TEMPLATES = {
        "detailed_objective": {
            "negative": "<image>\nThis image has a negative sentiment. Provide a detailed, objective description of what is visible in the image, including objects, people, actions, colors, setting, and atmosphere.",
            "neutral": "<image>\nThis image has a neutral sentiment. Provide a detailed, objective description of what is visible in the image, including objects, people, actions, colors, setting, and atmosphere.",
            "positive": "<image>\nThis image has a positive sentiment. Provide a detailed, objective description of what is visible in the image, including objects, people, actions, colors, setting, and atmosphere."
        },
        
        "concise_factual": {
            "negative": "<image>\nFor sentiment analysis, this is negative. In 2-3 sentences, describe the key elements and scene in this image.",
            "neutral": "<image>\nFor sentiment analysis, this is neutral. In 2-3 sentences, describe the key elements and scene in this image.",
            "positive": "<image>\nFor sentiment analysis, this is positive. In 2-3 sentences, describe the key elements and scene in this image."
        },
        
        "emotion_focused": {
            "negative": "<image>\nThis image conveys negative sentiment. Describe the image focusing on the emotional cues, atmosphere, and mood that contribute to this sentiment.",
            "neutral": "<image>\nThis image conveys neutral sentiment. Describe the image focusing on the emotional cues, atmosphere, and mood that contribute to this sentiment.",
            "positive": "<image>\nThis image conveys positive sentiment. Describe the image focusing on the emotional cues, atmosphere, and mood that contribute to this sentiment."
        },
        
        "compositional": {
            "negative": "<image>\nGiven the negative sentiment, describe this image by analyzing: 1) Main subjects and their arrangement, 2) Visual elements like lighting and colors, 3) Overall scene context.",
            "neutral": "<image>\nGiven the neutral sentiment, describe this image by analyzing: 1) Main subjects and their arrangement, 2) Visual elements like lighting and colors, 3) Overall scene context.",
            "positive": "<image>\nGiven the positive sentiment, describe this image by analyzing: 1) Main subjects and their arrangement, 2) Visual elements like lighting and colors, 3) Overall scene context."
        },
        
        "action_context": {
            "negative": "<image>\nThis is a negative sentiment image. Describe what is happening in the image, including any actions, interactions, or events visible in the scene.",
            "neutral": "<image>\nThis is a neutral sentiment image. Describe what is happening in the image, including any actions, interactions, or events visible in the scene.",
            "positive": "<image>\nThis is a positive sentiment image. Describe what is happening in the image, including any actions, interactions, or events visible in the scene."
        },
        
        "visual_elements": {
            "negative": "<image>\nFor a sentiment analysis task (negative), describe the visual characteristics: colors, lighting, composition, spatial relationships, and notable details.",
            "neutral": "<image>\nFor a sentiment analysis task (neutral), describe the visual characteristics: colors, lighting, composition, spatial relationships, and notable details.",
            "positive": "<image>\nFor a sentiment analysis task (positive), describe the visual characteristics: colors, lighting, composition, spatial relationships, and notable details."
        },
        
        "scene_narrative": {
            "negative": "<image>\nLabel: Negative sentiment. Describe this image as if explaining the scene to someone who cannot see it, including context and relevant details.",
            "neutral": "<image>\nLabel: Neutral sentiment. Describe this image as if explaining the scene to someone who cannot see it, including context and relevant details.",
            "positive": "<image>\nLabel: Positive sentiment. Describe this image as if explaining the scene to someone who cannot see it, including context and relevant details."
        },
        
        "alternative_perspective": {
            "negative": "<image>\nThis image is classified as negative. Provide an alternative description focusing on different aspects than a typical description might emphasize.",
            "neutral": "<image>\nThis image is classified as neutral. Provide an alternative description focusing on different aspects than a typical description might emphasize.",
            "positive": "<image>\nThis image is classified as positive. Provide an alternative description focusing on different aspects than a typical description might emphasize."
        }
    }
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
    
    def generate_descriptions(self, pixel_values, sentiment):
        """Generate all description types for a sentiment analysis image"""
        sentiment = sentiment.lower()
        results = {}
        
        for strategy in self.PROMPT_TEMPLATES.keys():
            question = self.PROMPT_TEMPLATES[strategy][sentiment]
            
            try:
                response = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    question, 
                    self.generation_config
                )
                results[strategy] = response
            except Exception as e:
                print(f"Error with {strategy}: {e}")
                results[strategy] = f"Error: {str(e)}"
        
        return results


def parse_tsv_line(line):
    """Parse a TSV line handling variable number of columns"""
    parts = line.strip().split('\t')
    
    if len(parts) == 4:
        # Format: index, label, image_id, text
        return {
            'index': parts[0],
            'label': parts[1],
            'image_id': parts[2],
            'text': parts[3],
            'entity': None
        }
    elif len(parts) == 5:
        # Format: index, label, image_id, text, entity
        return {
            'index': parts[0],
            'label': parts[1],
            'image_id': parts[2],
            'text': parts[3],
            'entity': parts[4]
        }
    else:
        raise ValueError(f"Unexpected number of columns: {len(parts)}")


def label_to_sentiment(label_str):
    """Convert label to sentiment string"""
    # Handle both numeric and string labels
    if label_str == '0' or label_str.lower() == 'negative':
        return 'negative'
    elif label_str == '1' or label_str.lower() == 'neutral':
        return 'neutral'
    elif label_str == '2' or label_str.lower() == 'positive':
        return 'positive'
    else:
        raise ValueError(f"Unknown label: {label_str}")


def process_dataset(dataset_name, dataset_root, split, model, tokenizer, augmenter):
    """Process a single dataset split (train or dev)"""
    
    dataset_path = Path(dataset_root) / dataset_name
    tsv_file = dataset_path / f"{split}.tsv"
    imgs_dir = dataset_path / "imgs"
    augments_dir = dataset_path / "augments"
    
    # Create augments directory
    augments_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name}/{split}.tsv")
    print(f"{'='*80}")
    
    # Read TSV file
    with open(tsv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0]
    data_lines = lines[1:]
    
    # Storage for different description types
    description_types = list(SentimentDescriptionAugmenter.PROMPT_TEMPLATES.keys())
    type_outputs = {desc_type: [] for desc_type in description_types}
    augment_output = []  # For combined augment file
    
    # Process each line
    for line in tqdm(data_lines, desc=f"{dataset_name}/{split}"):
        if not line.strip():
            continue
        
        try:
            parsed = parse_tsv_line(line)
            image_path = imgs_dir / parsed['image_id']
            
            # Check if image exists
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                # Use original text for all types if image missing
                for desc_type in description_types:
                    if parsed['entity']:
                        type_outputs[desc_type].append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{parsed['text']}\t{parsed['entity']}\n")
                    else:
                        type_outputs[desc_type].append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{parsed['text']}\n")
                
                if parsed['entity']:
                    augment_output.append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{parsed['text']}\t{parsed['entity']}\n")
                else:
                    augment_output.append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{parsed['text']}\n")
                continue
            
            # Load image
            pixel_values = load_image(str(image_path), max_num=12).to(torch.bfloat16).cuda()
            
            # Get sentiment
            sentiment = label_to_sentiment(parsed['label'])
            
            # Generate all descriptions
            descriptions = augmenter.generate_descriptions(pixel_values, sentiment)
            
            # Save to separate type files
            for desc_type, description in descriptions.items():
                if parsed['entity']:
                    type_outputs[desc_type].append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{description}\t{parsed['entity']}\n")
                else:
                    type_outputs[desc_type].append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{description}\n")
            
            # Concatenate all descriptions for augment file (one per line)
            combined_description = '\n'.join([descriptions[dt] for dt in description_types])
            
            if parsed['entity']:
                augment_output.append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{combined_description}\t{parsed['entity']}\n")
            else:
                augment_output.append(f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{combined_description}\n")
            
        except Exception as e:
            print(f"Error processing line: {line.strip()}")
            print(f"Error: {e}")
            continue
    
    # Write separate files for each description type
    for desc_type in description_types:
        output_file = augments_dir / f"{split}_{desc_type}.tsv"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.writelines(type_outputs[desc_type])
        print(f"✓ Saved: {output_file}")
    
    # Write combined augment file
    augment_file = augments_dir / f"{split}_augment.tsv"
    with open(augment_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(augment_output)
    print(f"✓ Saved: {augment_file}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate augmented descriptions for multiple datasets')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory containing all datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, 
                        help='List of dataset names to process (e.g., masad mvsa-m mvsa-s)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'dev'],
                        help='Splits to process (default: train dev)')
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL3_5-14B',
                        help='Path to the VLM model')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-DATASET AUGMENTATION GENERATOR")
    print("="*80)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Model: {args.model_path}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    with init_empty_weights():
        model_empty = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    
    max_memory = get_balanced_memory(
        model_empty,
        max_memory=None,
        no_split_module_classes=model_empty._no_split_modules if hasattr(model_empty, '_no_split_modules') else None,
        dtype=torch.bfloat16,
    )
    
    device_map = infer_auto_device_map(
        model_empty,
        max_memory=max_memory,
        no_split_module_classes=model_empty._no_split_modules if hasattr(model_empty, '_no_split_modules') else None,
        dtype=torch.bfloat16,
    )
    
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    
    print("✓ Model loaded successfully\n")
    
    # Create augmenter
    augmenter = SentimentDescriptionAugmenter(model, tokenizer)
    
    # Process each dataset and split
    for dataset_name in args.datasets:
        for split in args.splits:
            process_dataset(dataset_name, args.dataset_root, split, model, tokenizer, augmenter)
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(args.datasets)} dataset(s) × {len(args.splits)} split(s)")
    print("Each dataset now has:")
    print("  - {split}_augment.tsv (all descriptions concatenated)")
    print("  - {split}_{description_type}.tsv (8 separate files per description type)")
    print("="*80)


if __name__ == "__main__":
    main()