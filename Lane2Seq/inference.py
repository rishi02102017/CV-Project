import os
import torch
import yaml
import json
import cv2
from tqdm import tqdm

from models.lane2seq import Lane2Seq
from datasets.lane_dataset import TuSimpleDataset
from utils.tokenizer import LaneTokenizer
from utils.visualizer import draw_lanes
from utils.utils import load_checkpoint_inference
from utils.utils import denormalize_image

# === Load config ===
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# === Initialize tokenizer ===
tokenizer = LaneTokenizer(nbins=config['vocab_size'] - 7)  # ✅ consistent with training

# === Initialize model ===
model = Lane2Seq(
    encoder_checkpoint=config['encoder_checkpoint'],
    vocab_size=config['vocab_size'],
    hidden_size=config['hidden_size'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    ff_size=config['ff_size'],
    max_seq_length=config['max_seq_length'],
    end_token=tokenizer.END_TOKEN
).to(device)

# === Load checkpoint ===
checkpoint_path = config['inference']['checkpoint']
load_checkpoint_inference(model, checkpoint_path)

# === Prepare dataset ===
test_data_path = config['inference']['data_path']
output_dir = config['inference']['output_dir']
os.makedirs(output_dir, exist_ok=True)

dataset = TuSimpleDataset(
    root_dir=test_data_path,
    split='test',  # ✅ make sure split is correct
    nbins=config['vocab_size'] - 7,
    format_type=config['format_type'],
    image_size=config['image_size']
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# === Inference loop ===
model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Running Inference"):
        image = batch['image'].to(device)
        raw_file = batch['raw_file'][0]  # e.g., 'clips/0530/1492626780768443246_0/20.jpg'
    
        # Clean filename: remove extension, replace slashes with underscores
        clean_name = os.path.splitext(raw_file)[0].replace('/', '_')
    
        # Generate outputs paths
        json_path = os.path.join(output_dir, f"{clean_name}.json")
        vis_img_path = os.path.join(output_dir, f"{clean_name}.png")
    
        # Prepare prompt token
        format_token_map = {
            'segmentation': tokenizer.FORMAT_TOKENS['segmentation'],
            'anchor': tokenizer.FORMAT_TOKENS['anchor'],
            'parameter': tokenizer.FORMAT_TOKENS['parameter']
        }
        prompt_token = torch.tensor([[tokenizer.START_TOKEN, format_token_map[config['format_type']]]], device=device)
    
        # Generate prediction
        generated_seq = model.generate(image, prompt_token)
        width, height = config['image_size'][1], config['image_size'][0]
        # Explicitly pass (width, height)
        prediction = tokenizer.decode(
            generated_seq[0].tolist(),
            (config['image_size'][1], config['image_size'][0]),  # (width, height)
            config['format_type']
        )

    
        # Save JSON
        with open(json_path, 'w') as jf:
            json.dump(prediction, jf, indent=4)
    
        # Save Visualization if valid
        # Convert to numpy and transpose to HWC
        #image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        original_image = denormalize_image(image.squeeze(0))
        vis_img = draw_lanes(original_image, prediction)

        #vis_img = draw_lanes(image_np, prediction)
        if vis_img is not None and vis_img.shape[0] > 0 and vis_img.shape[1] > 0:
            #vis_img = (vis_img * 255).astype('uint8')  # scale back to 0-255
            cv2.imwrite(vis_img_path, vis_img)
        else:
            print(f"[Warning] Skipped invalid visualization for {clean_name}")


print(f"Inference completed. Results saved to {output_dir}")
