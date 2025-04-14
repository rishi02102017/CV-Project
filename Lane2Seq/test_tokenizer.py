# lane2seq_project/test_tokenizer.py

import random
import json
import os
from datasets.lane_dataset import TuSimpleDataset
from utils.tokenizer import LaneTokenizer
from utils.visualizer import draw_lanes
from PIL import Image
import cv2

# Special tokens mapping for readability
SPECIAL_TOKENS = {
    0: "<START>",
    1: "<END>",
    2: "<FORMAT_SEGMENTATION>",
    3: "<FORMAT_ANCHOR>",
    4: "<FORMAT_PARAMETER>",
    5: "<LANE_SEPARATOR>"
}

def interpret_tokens(sequence):
    readable = []
    for token in sequence:
        if token in SPECIAL_TOKENS:
            readable.append(SPECIAL_TOKENS[token])
        else:
            readable.append(str(token))
    return readable

def test_tokenizer(dataset_path, format_type='anchor', nbins=1000, num_samples=5):
    # Init tokenizer
    tokenizer = LaneTokenizer(nbins=nbins)

    # Load dataset
    dataset = TuSimpleDataset(
        root_dir=dataset_path,
        split='train',
        nbins=nbins,
        format_type=format_type
    )

    # Output folder
    output_dir = f"test_outputs_{format_type}"
    os.makedirs(output_dir, exist_ok=True)

    # Pick random samples
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        sample = dataset.samples[idx]
        raw_image_path = f"{dataset.root_dir}/{sample['raw_file']}"
        image = Image.open(raw_image_path).convert('RGB')

        # ✅ Resize image for visualization to match annotation scaling
        image = image.resize((dataset.image_size[1], dataset.image_size[0]), Image.BILINEAR)


        # ✅ Use resized image dimensions for encoding/decoding
        height, width = dataset.image_size

        # ✅ Pass original size properly to annotation conversion
        original_image = Image.open(raw_image_path).convert('RGB')
        original_size = original_image.size
        original_width, original_height = original_image.size
        target_height, target_width = dataset.image_size  # dataset.image_size is (H, W)
        annotation = dataset._convert_annotation(sample, original_size=original_size, target_size=(width, height))

        print(f"[DEBUG] Annotation lanes after conversion: {len(annotation['lanes'])}")

        # Encode
        input_seq, target_seq = tokenizer.encode(annotation, (width, height), format_type=format_type)

        # Decode
        decoded_annotation = tokenizer.decode(input_seq, (width, height), format_type=format_type)

        for i, (orig_lane, decoded_lane) in enumerate(zip(annotation['lanes'], decoded_annotation)):
            print(f"\n[COMPARE] Lane {i}")
            print(f"Original params: {orig_lane.get('params', 'points')}")
            print(f"Decoded params: {decoded_lane.get('params', 'points')}")

        # Prepare sequence JSON for inspection
        json_output = {
            "raw_file": sample['raw_file'],
            "input_sequence_tokens": input_seq,
            "input_sequence_readable": interpret_tokens(input_seq),
            "target_sequence_tokens": target_seq,
            "target_sequence_readable": interpret_tokens(target_seq),
            "decoded_annotation": decoded_annotation,
            "original_annotation": annotation
        }

        # Save sequence JSON
        sequence_json_path = os.path.join(output_dir, f"sample_{idx}_{format_type}_sequence.json")
        with open(sequence_json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved sequence JSON to {sequence_json_path}")

        # Draw original lanes (Green)
        original_image_vis = draw_lanes(image.copy(), annotation['lanes'], color=(0, 255, 0))

        # Draw decoded lanes (Red)
        decoded_image_vis = draw_lanes(image.copy(), decoded_annotation, color=(0, 0, 255))

        # Side-by-side comparison
        concatenated = Image.fromarray(cv2.hconcat([original_image_vis, decoded_image_vis]))
        side_by_side_output = os.path.join(output_dir, f"sample_{idx}_{format_type}_sidebyside.jpg")
        concatenated.save(side_by_side_output)
        print(f"Saved side-by-side comparison to {side_by_side_output}")

        # Overlay both original and decoded
        overlay_image = draw_lanes(image.copy(), annotation['lanes'], color=(0, 255, 0))
        overlay_image = draw_lanes(overlay_image, decoded_annotation, color=(0, 0, 255))
        overlay_output = os.path.join(output_dir, f"sample_{idx}_{format_type}_overlay.jpg")
        Image.fromarray(overlay_image).save(overlay_output)
        print(f"Saved overlay comparison to {overlay_output}")

    print(f"\n✅ All samples processed! Output folder: {output_dir}")

if __name__ == "__main__":
    dataset_path = "../archive/TUSimple/train_set"
    format_type = "segmentation"  # Change to "segmentation", "anchor", or "parameter"
    test_tokenizer(dataset_path, format_type=format_type)
