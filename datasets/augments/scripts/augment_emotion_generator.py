# augment_emotion_dataset.py

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
import re

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Text cleaner: remove Markdown, links, headers, lists, quotes, code ticks, newlines, extra spaces.
def clean_text(text: str) -> str:
    if text is None:
        return ""
    # Remove markdown emphasis
    text = re.sub(r'\*\*|__|\*|_', '', text)
    # Remove code backticks
    text = re.sub(r'`{1,3}', '', text)
    # Remove headers, list markers, blockquotes
    text = re.sub(r'^#{1,6}\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
    # Convert links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove strike
    text = re.sub(r'~~', '', text)
    # Remove lingering link brackets
    text = re.sub(r'\[|\]|\(|\)', '', text)
    # Remove all newlines
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_transform(input_size: int):
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
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
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

class EmotionDescriptionAugmenter:
    """
    Generate varied descriptions tailored to the new emotion labels:
    Angry, Bored, Calm, Fear, Happy, Love, Sad
    All prompts explicitly prohibit Markdown and newlines in outputs.
    """

    EMOTIONS = ["angry", "bored", "calm", "fear", "happy", "love", "sad"]

    # Redesigned prompts per emotion for 8 strategies
    PROMPT_TEMPLATES = {
        "detailed_objective": {
            "angry": "<image>\nThis image expresses anger. Provide a detailed, objective description of visible subjects, actions, colors, setting, and atmosphere that convey anger without interpreting beyond what is shown. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nThis image expresses boredom. Provide a detailed, objective description of visible subjects, actions (or lack of action), colors, setting, and atmosphere that convey boredom. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nThis image expresses calm. Provide a detailed, objective description of visible subjects, environment, colors, lighting, and atmosphere that convey calmness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nThis image expresses fear. Provide a detailed, objective description of visible subjects, environment, cues of tension, colors, lighting, and atmosphere that convey fear. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nThis image expresses happiness. Provide a detailed, objective description of visible subjects, actions, colors, setting, and atmosphere that convey happiness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nThis image expresses love. Provide a detailed, objective description of visible subjects, interactions, colors, setting, and atmosphere that convey affection or closeness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nThis image expresses sadness. Provide a detailed, objective description of visible subjects, actions, colors, setting, and atmosphere that convey sadness. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
        "concise_factual": {
            "angry": "<image>\nEmotion: anger. In 2-3 sentences, describe the key elements and scene, focusing on observable details that convey anger. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nEmotion: boredom. In 2-3 sentences, describe the key elements and scene, focusing on observable details that convey boredom. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nEmotion: calm. In 2-3 sentences, describe the key elements and scene, focusing on observable details that convey calmness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nEmotion: fear. In 2-3 sentences, describe the key elements and scene, focusing on observable details that convey fear. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nEmotion: happiness. In 2-3 sentences, describe the key elements and scene, focusing on observable details that convey happiness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nEmotion: love. In 2-3 sentences, describe the key elements and scene, focusing on observable details that convey affection or closeness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nEmotion: sadness. In 2-3 sentences, describe the key elements and scene, focusing on observable details that convey sadness. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
        "emotion_focused": {
            "angry": "<image>\nFocus on how anger is conveyed: facial expressions, body language, motion, color, lighting, and composition. Describe only what is visually evident. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nFocus on how boredom is conveyed: postures, inactivity, repetition, minimal engagement, color palette, and composition. Describe only what is visually evident. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nFocus on how calmness is conveyed: relaxed poses, open space, soft light, gentle colors, symmetry, and balance. Describe only what is visually evident. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nFocus on how fear is conveyed: tense poses, defensive gestures, shadows, obscured areas, harsh contrasts, and confined spaces. Describe only what is visually evident. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nFocus on how happiness is conveyed: smiles, playful actions, bright colors, festive elements, and dynamic composition. Describe only what is visually evident. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nFocus on how love is conveyed: proximity, touch, eye contact, tender gestures, warm tones, and harmonious framing. Describe only what is visually evident. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nFocus on how sadness is conveyed: downcast gazes, slumped posture, tears or wetness, cool or muted colors, isolation, and negative space. Describe only what is visually evident. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
        "compositional": {
            "angry": "<image>\nAnalyze composition for anger: 1) Main subjects and arrangement, 2) Lighting, color, and motion cues, 3) Scene context and tension. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nAnalyze composition for boredom: 1) Subjects and spacing, 2) Lighting, repetition, and muted palette, 3) Scene context suggesting inactivity. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nAnalyze composition for calm: 1) Subjects and balance, 2) Soft light and gentle colors, 3) Context supporting tranquility. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nAnalyze composition for fear: 1) Subjects and proximity, 2) Shadows, contrasts, and color, 3) Context suggesting threat or unease. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nAnalyze composition for happiness: 1) Subjects and interactions, 2) Brightness, color, and rhythm, 3) Context indicating celebration or joy. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nAnalyze composition for love: 1) Subjects and closeness, 2) Warm tones and soft light, 3) Context suggesting intimacy or care. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nAnalyze composition for sadness: 1) Subjects and isolation, 2) Dim light, cool tones, and negative space, 3) Context indicating loss or sorrow. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
        "action_context": {
            "angry": "<image>\nDescribe what is happening emphasizing cues of anger such as forceful motions, tense interactions, or confrontational scenes. Stick to observable details. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nDescribe what is happening emphasizing cues of boredom such as idling, waiting, routine activities, or lack of engagement. Stick to observable details. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nDescribe what is happening emphasizing cues of calm such as rest, meditation, leisure, or stillness. Stick to observable details. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nDescribe what is happening emphasizing cues of fear such as fleeing, hiding, defensive postures, or looming threats. Stick to observable details. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nDescribe what is happening emphasizing cues of happiness such as celebrating, smiling, playing, or sharing moments. Stick to observable details. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nDescribe what is happening emphasizing cues of love such as hugging, holding hands, caring gestures, or affectionate exchanges. Stick to observable details. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nDescribe what is happening emphasizing cues of sadness such as crying, parting, memorials, or solitary reflection. Stick to observable details. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
        "visual_elements": {
            "angry": "<image>\nFor anger, describe visual characteristics: color palette (e.g., reds), lighting contrast, motion blur, facial expressions, and spatial relationships. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nFor boredom, describe visual characteristics: subdued palette, repetitive forms, minimal action, empty space, and slow rhythm. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nFor calm, describe visual characteristics: soft light, low contrast, gentle colors, balanced framing, and open space. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nFor fear, describe visual characteristics: deep shadows, stark contrasts, desaturated tones, tight framing, and obscured details. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nFor happiness, describe visual characteristics: bright colors, high key lighting, lively patterns, energetic poses, and dynamic framing. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nFor love, describe visual characteristics: warm tones, soft highlights, gentle focus, close framing, and mirrored or harmonious shapes. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nFor sadness, describe visual characteristics: cool or muted colors, low light, soft focus, negative space, and stillness. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
        "scene_narrative": {
            "angry": "<image>\nEmotion label: angry. Describe this scene as if explaining it to someone who cannot see it, including concrete details that convey anger. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nEmotion label: bored. Describe this scene as if explaining it to someone who cannot see it, including concrete details that convey boredom. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nEmotion label: calm. Describe this scene as if explaining it to someone who cannot see it, including concrete details that convey calmness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nEmotion label: fear. Describe this scene as if explaining it to someone who cannot see it, including concrete details that convey fear. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nEmotion label: happy. Describe this scene as if explaining it to someone who cannot see it, including concrete details that convey happiness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nEmotion label: love. Describe this scene as if explaining it to someone who cannot see it, including concrete details that convey affection or closeness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nEmotion label: sad. Describe this scene as if explaining it to someone who cannot see it, including concrete details that convey sadness. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
        "alternative_perspective": {
            "angry": "<image>\nProvide an alternative description of this angry image by emphasizing lesser-noticed elements or background details that still support the emotion of anger. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "bored": "<image>\nProvide an alternative description of this bored image by emphasizing lesser-noticed elements or background details that still support the emotion of boredom. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "calm": "<image>\nProvide an alternative description of this calm image by emphasizing lesser-noticed elements or background details that still support the emotion of calmness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "fear": "<image>\nProvide an alternative description of this fearful image by emphasizing lesser-noticed elements or background details that still support the emotion of fear. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "happy": "<image>\nProvide an alternative description of this happy image by emphasizing lesser-noticed elements or background details that still support the emotion of happiness. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "love": "<image>\nProvide an alternative description of this loving image by emphasizing lesser-noticed elements or background details that still support the emotion of love. Do not use any Markdown formatting and do not use line breaks or newlines.",
            "sad": "<image>\nProvide an alternative description of this sad image by emphasizing lesser-noticed elements or background details that still support the emotion of sadness. Do not use any Markdown formatting and do not use line breaks or newlines."
        },
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        self.strategy_list = list(self.PROMPT_TEMPLATES.keys())

    def generate_descriptions_batch(self, pixel_values_list, num_patches_list, emotions):
        """
        Args:
            pixel_values_list: List[Tensor], each tensor is patches for one image
            num_patches_list: List[int], number of patches per image in the same order
            emotions: List[str], each in self.EMOTIONS (lowercase)
        Returns:
            List[dict] mapping strategy -> cleaned response for each image
        """
        batch_size = len(emotions)
        results = [{} for _ in range(batch_size)]

        for strategy in self.strategy_list:
            questions = [self.PROMPT_TEMPLATES[strategy][emo] for emo in emotions]
            pixel_values_batch = torch.cat(pixel_values_list, dim=0)
            try:
                responses = self.model.batch_chat(
                    self.tokenizer,
                    pixel_values_batch,
                    num_patches_list=num_patches_list,
                    questions=questions,
                    generation_config=self.generation_config
                )
                cleaned = [clean_text(r) for r in responses]
                for i, resp in enumerate(cleaned):
                    results[i][strategy] = resp
            except Exception as e:
                print(f"Error with batch {strategy}: {e}")
                for i in range(batch_size):
                    results[i][strategy] = f"Error: {str(e)}"
        return results

def parse_tsv_line_new(line: str):
    """
    Robust TSV parser for this dataset:
    - Expected columns: index, label, image_id, text (but there can be more than 4)
    - We take: index=parts[0], label=parts[1], image_id=parts[2], text='\t'.join(parts[3:])
    """
    parts = line.rstrip('\n').split('\t')
    if len(parts) < 3:
        raise ValueError(f"Unexpected number of columns: {len(parts)} in line: {line!r}")
    index = parts[0]
    label = parts[1]
    image_id = parts[2]
    text = '\t'.join(parts[3:]) if len(parts) > 3 else ''
    return {
        'index': index,
        'label': label,
        'image_id': image_id,
        'text': text
    }

def label_to_emotion(label_str: str) -> str:
    """
    Normalize labels to the supported set.
    Supports case-insensitive match for: Angry, Bored, Calm, Fear, Happy, Love, Sad
    """
    s = (label_str or "").strip().lower()
    mapping = {
        "angry": "angry",
        "anger": "angry",
        "bored": "bored",
        "boredom": "bored",
        "calm": "calm",
        "fear": "fear",
        "fearful": "fear",
        "scared": "fear",
        "happy": "happy",
        "happiness": "happy",
        "joy": "happy",
        "love": "love",
        "loving": "love",
        "sad": "sad",
        "sadness": "sad",
        "depressed": "sad",
    }
    if s in mapping:
        return mapping[s]
    # Try to recover frequent variants
    if "angr" in s:
        return "angry"
    if "bored" in s:
        return "bored"
    if "calm" in s or "peace" in s or "tranquil" in s:
        return "calm"
    if "fear" in s or "afraid" in s or "scare" in s:
        return "fear"
    if "happy" in s or "joy" in s or "delight" in s:
        return "happy"
    if "love" in s or "affection" in s or "romance" in s:
        return "love"
    if "sad" in s or "sorrow" in s or "melanch" in s:
        return "sad"
    raise ValueError(f"Unknown label: {label_str}")

def process_tsv_file(tsv_path: Path, imgs_dir: Path, model, tokenizer, augmenter, batch_size=8):
    """
    Process only this dataset TSV.
    Outputs:
      - {split}_{description_type}.tsv for each strategy
      - {split}_augment.tsv with all descriptions (each strategy expands to a record)
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    if not imgs_dir.exists():
        print(f"Warning: Images directory not found: {imgs_dir}")

    augments_dir = tsv_path.parent / "augments"
    augments_dir.mkdir(exist_ok=True)

    split = tsv_path.stem
    description_types = list(EmotionDescriptionAugmenter.PROMPT_TEMPLATES.keys())

    print(f"\n{'='*80}")
    print(f"Processing TSV: {tsv_path}")
    print(f"Images dir: {imgs_dir}")
    print(f"{'='*80}")

    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print("Empty TSV; nothing to do.")
        return

    header = lines[0]
    data_lines = lines[1:]

    header_written = {desc_type: False for desc_type in description_types}
    header_written['augment'] = False

    output_files = {
        desc_type: augments_dir / f"{split}_{desc_type}.tsv"
        for desc_type in description_types
    }
    output_files['augment'] = augments_dir / f"{split}_augment.tsv"

    num_batches = (len(data_lines) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"{split}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data_lines))
        batch_lines = data_lines[start_idx:end_idx]

        batch_type_data = {desc_type: [] for desc_type in description_types}
        batch_augment_data = []

        batch_parsed = []
        batch_pixel_values = []
        batch_num_patches = []
        batch_emotions = []
        batch_valid_indices = []

        for line in batch_lines:
            if not line.strip():
                batch_parsed.append(None)
                batch_valid_indices.append(None)
                continue
            try:
                parsed = parse_tsv_line_new(line)
                image_path = imgs_dir / parsed['image_id']
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}")
                    batch_parsed.append(parsed)
                    batch_valid_indices.append(None)
                    continue

                pixel_values = load_image(str(image_path), max_num=12).to(torch.bfloat16).cuda()
                emotion = label_to_emotion(parsed['label'])

                batch_parsed.append(parsed)
                batch_pixel_values.append(pixel_values)
                batch_num_patches.append(pixel_values.size(0))
                batch_emotions.append(emotion)
                batch_valid_indices.append(len(batch_pixel_values) - 1)

            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error: {e}")
                batch_parsed.append(None)
                batch_valid_indices.append(None)
                continue

        if batch_pixel_values:
            batch_descriptions = augmenter.generate_descriptions_batch(
                batch_pixel_values,
                batch_num_patches,
                batch_emotions
            )
        else:
            batch_descriptions = []

        for parsed, valid_idx in zip(batch_parsed, batch_valid_indices):
            if parsed is None:
                continue

            if valid_idx is None:
                cleaned_original_text = clean_text(parsed['text'])
                for desc_type in description_types:
                    line_out = f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{cleaned_original_text}\n"
                    batch_type_data[desc_type].append(line_out)
                batch_augment_data.append(
                    f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{cleaned_original_text}\n"
                )
            else:
                descriptions = batch_descriptions[valid_idx]
                for desc_type, description in descriptions.items():
                    line_out = f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{description}\n"
                    batch_type_data[desc_type].append(line_out)
                for dt in description_types:
                    description = descriptions[dt]
                    line_out = f"{parsed['index']}\t{parsed['label']}\t{parsed['image_id']}\t{description}\n"
                    batch_augment_data.append(line_out)

        for desc_type in description_types:
            with open(output_files[desc_type], 'a', encoding='utf-8') as f:
                if not header_written[desc_type]:
                    f.write(header)
                    header_written[desc_type] = True
                f.writelines(batch_type_data[desc_type])

        with open(output_files['augment'], 'a', encoding='utf-8') as f:
            if not header_written['augment']:
                f.write(header)
                header_written['augment'] = True
            f.writelines(batch_augment_data)

        print(f"✓ Batch {batch_idx+1}/{num_batches} saved to files")

    print(f"\n✓ All batches processed for {split}")
    for desc_type in description_types:
        print(f"  - {output_files[desc_type].name}")
    print(f"  - {output_files['augment'].name}")
    print(f"{'='*80}\n")

def _normalize_splits(splits):
    # Support space- or comma-separated entries: ["train,dev", "test"] -> ["train","dev","test"]
    out = []
    for s in splits or []:
        if not s:
            continue
        parts = [p.strip() for p in s.split(',')]
        out.extend([p for p in parts if p])
    # preserve order while removing duplicates
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def main():
    parser = argparse.ArgumentParser(
        description='Generate augmented descriptions for the new emotion dataset (Angry, Bored, Calm, Fear, Happy, Love, Sad).'
    )
    # Option A: direct TSV(s)
    parser.add_argument('--tsv_file', type=str, default=None, help='Path to a single TSV file to process (deprecated; use --tsv_files)')
    parser.add_argument('--tsv_files', type=str, nargs='+', default=None, help='Path(s) to TSV file(s) to process')
    # Option B: root/name/split(s)
    parser.add_argument('--dataset_root', type=str, default=None, help='Root directory containing the dataset')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset directory name under root')
    parser.add_argument('--split', type=str, default=None, help='Single split name (e.g., train or dev); deprecated in favor of --splits')
    parser.add_argument('--splits', type=str, nargs='+', default=None, help='Split names (e.g., train dev test) or comma-separated')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL3_5-8B', help='VLM model path')

    args = parser.parse_args()

    # Normalize CLI into a list of TSV paths or a list of split names
    tsv_paths = None
    if args.tsv_files:
        tsv_paths = [Path(p).resolve() for p in args.tsv_files]
    elif args.tsv_file:
        # backward compatibility
        tsv_paths = [Path(args.tsv_file).resolve()]
        print("Note: --tsv_file is deprecated; prefer --tsv_files for multi-file processing.")

    split_names = _normalize_splits(args.splits) if args.splits else None
    if not split_names and args.split:
        split_names = _normalize_splits([args.split])
        print("Note: --split is deprecated; prefer --splits for multi-split processing.")

    # Load model/tokenizer once
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
        no_split_module_classes=getattr(model_empty, '_no_split_modules', None),
        dtype=torch.bfloat16,
    )

    device_map = infer_auto_device_map(
        model_empty,
        max_memory=max_memory,
        no_split_module_classes=getattr(model_empty, '_no_split_modules', None),
        dtype=torch.bfloat16,
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    print("✓ Model loaded successfully\n")

    augmenter = EmotionDescriptionAugmenter(model, tokenizer)

    processed_files = []

    if tsv_paths:
        print("="*80)
        print("Multi-TSV mode")
        for tsv_path in tsv_paths:
            dataset_dir = tsv_path.parent
            imgs_dir = dataset_dir / "imgs"
            print("-"*80)
            print(f"TSV: {tsv_path}")
            print(f"Images dir: {imgs_dir}")
            try:
                process_tsv_file(tsv_path, imgs_dir, model, tokenizer, augmenter, batch_size=args.batch_size)
                processed_files.append(tsv_path)
            except Exception as e:
                print(f"Error processing {tsv_path}: {e}")
        print("="*80)
    else:
        # Root/Name/Split(s) mode
        if not (args.dataset_root and args.dataset_name and split_names):
            raise ValueError("Provide either --tsv_files (or --tsv_file) OR all of --dataset_root, --dataset_name, and --splits/--split")
        dataset_dir = Path(args.dataset_root).resolve() / args.dataset_name
        imgs_dir = dataset_dir / "imgs"

        print("="*80)
        print("Root/Name/Split(s) mode")
        print(f"Dataset dir: {dataset_dir}")
        print(f"Images dir: {imgs_dir}")
        print(f"Splits: {', '.join(split_names)}")
        print("="*80)

        for split_name in split_names:
            tsv_path = dataset_dir / f"{split_name}.tsv"
            try:
                process_tsv_file(tsv_path, imgs_dir, model, tokenizer, augmenter, batch_size=args.batch_size)
                processed_files.append(tsv_path)
            except Exception as e:
                print(f"Error processing split '{split_name}' ({tsv_path}): {e}")

    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE")
    print("="*80)
    if processed_files:
        print("\nProcessed file(s):")
        for p in processed_files:
            print(f"  - {p.name}")
    else:
        print("\nNo files were processed.")
    print("Generated files are saved per TSV under an 'augments' directory next to each TSV.")
    print("Notes:")
    print("  1) All texts are cleaned of Markdown, line breaks, and extra spaces.")
    print("  2) Results are saved incrementally after each batch.")
    print("="*80)

if __name__ == "__main__":
    main()