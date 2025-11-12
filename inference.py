import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append("/content/CV_Project/Code-Files")
from model.model_for_video import GSANet


MODEL_PATH = "/content/drive/MyDrive/CVProject_1_log/model/best_model.pth"
DAVIS_TEST_ROOT = "/content/CV_Project/DAVIS-data/dataset/DAVIS_test"
OUT_DIR = "/content/CV_Project/Code-Files/DAVIS-evaluation/DAVIS/Results/480p/GSANet_FT"

SLOT_OUT = "/content/CV_Project/outputs/slot_maps"
os.makedirs(f"{SLOT_OUT}/rgb_fg", exist_ok=True)
os.makedirs(f"{SLOT_OUT}/rgb_bg", exist_ok=True)
os.makedirs(f"{SLOT_OUT}/flow_fg", exist_ok=True)
os.makedirs(f"{SLOT_OUT}/flow_bg", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(OUT_DIR, exist_ok=True)

model = GSANet(num_slots=2).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)

if "model_state_dict" in ckpt:
    print("Loading model_state_dict from checkpoint...")
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        if 'stream_fuser' in key:
            if key in model_dict and value.shape != model_dict[key].shape:
                print(f"Skipping {key}: shape mismatch {value.shape} vs {model_dict[key].shape}")
                continue
        filtered_state_dict[key] = value
    
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded weights! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
else:
    print("Loading full checkpoint directly...")
    model.load_state_dict(ckpt, strict=False)

model.eval()
print("Model ready for inference!")

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

flow_transform = transforms.Compose([transforms.ToTensor()])


def pad_to_multiple(tensor, multiple=32):
    h, w = tensor.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor, (h, w)


def sample_reference_frames(frame_list, current_idx, num_refs=4):
    total_frames = len(frame_list)
    if current_idx > 0:
        step = max(1, current_idx // num_refs)
        ref_indices = [max(0, current_idx - step * (i + 1)) for i in range(num_refs)]
        ref_indices = ref_indices[::-1]
    else:
        ref_indices = [0] * num_refs
    while len(ref_indices) < num_refs:
        ref_indices.append(ref_indices[-1])
    return ref_indices[:num_refs]

def save_map(tensor, path):
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    cv2.imwrite(path, arr)

with torch.no_grad():
    seqs = sorted(os.listdir(DAVIS_TEST_ROOT))
    for seq_name in seqs:
        seq_path = os.path.join(DAVIS_TEST_ROOT, seq_name)
        rgb_dir = os.path.join(seq_path, "RGB")
        flow_dir = os.path.join(seq_path, "FLOW")

        if not os.path.isdir(rgb_dir):
            continue

        save_folder = os.path.join(OUT_DIR, seq_name)
        os.makedirs(save_folder, exist_ok=True)
        print(f"\n Processing sequence: {seq_name}")

        frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
        for i in tqdm(range(len(frames)), desc=f"{seq_name:20s}", leave=False):
            img_curr = Image.open(os.path.join(rgb_dir, frames[i])).convert("RGB")
            w, h = img_curr.size

            flow_path = os.path.join(flow_dir, frames[i])
            if os.path.exists(flow_path):
                flow_img = cv2.imread(flow_path)
                flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
                flow_img = cv2.resize(flow_img, (w, h))
            else:
                flow_img = np.zeros((h, w, 3), dtype=np.uint8)

            ref_indices = sample_reference_frames(frames, i, num_refs=4)
            ref_imgs_list, ref_flows_list = [], []

            for ref_idx in ref_indices:
                ref_img = Image.open(os.path.join(rgb_dir, frames[ref_idx])).convert("RGB").resize((w, h))
                ref_imgs_list.append(img_transform(ref_img))
                ref_flow_path = os.path.join(flow_dir, frames[ref_idx])
                if os.path.exists(ref_flow_path):
                    ref_flow = cv2.imread(ref_flow_path)
                    ref_flow = cv2.cvtColor(ref_flow, cv2.COLOR_BGR2RGB)
                    ref_flow = cv2.resize(ref_flow, (w, h))
                else:
                    ref_flow = np.zeros((h, w, 3), dtype=np.uint8)
                ref_flows_list.append(flow_transform(Image.fromarray(ref_flow)))

            rgb_input = img_transform(img_curr).unsqueeze(0)
            flow_input = flow_transform(Image.fromarray(flow_img)).unsqueeze(0)
            ref_imgs = torch.cat(ref_imgs_list, dim=0).unsqueeze(0)
            ref_flows = torch.cat(ref_flows_list, dim=0).unsqueeze(0)

            rgb_input, orig_size = pad_to_multiple(rgb_input.to(device))
            flow_input, _ = pad_to_multiple(flow_input.to(device))
            ref_imgs, _ = pad_to_multiple(ref_imgs.to(device))
            ref_flows, _ = pad_to_multiple(ref_flows.to(device))

            try:
                outputs, fine_slots = model(rgb_input, flow_input, ref_imgs, ref_flows)
                pred = outputs[0]
                fine_slot_rgb_fg, fine_slot_rgb_bg, fine_slot_flow_fg, fine_slot_flow_bg = fine_slots

                pred = pred[:, :, :orig_size[0], :orig_size[1]]
                mask = pred.squeeze().cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                out_path = os.path.join(save_folder, frames[i].replace(".jpg", ".png"))
                cv2.imwrite(out_path, mask)

                base_name = frames[i].replace(".jpg", ".png")
                save_map(fine_slot_rgb_fg,  f"{SLOT_OUT}/rgb_fg/{seq_name}_{base_name}")
                save_map(fine_slot_rgb_bg,  f"{SLOT_OUT}/rgb_bg/{seq_name}_{base_name}")
                save_map(fine_slot_flow_fg, f"{SLOT_OUT}/flow_fg/{seq_name}_{base_name}")
                save_map(fine_slot_flow_bg, f"{SLOT_OUT}/flow_bg/{seq_name}_{base_name}")

            except Exception as e:
                print(f"\nError processing {seq_name}/{frames[i]}: {e}")
                import traceback
                traceback.print_exc()
                blank_mask = np.zeros((h, w), dtype=np.uint8)
                out_path = os.path.join(save_folder, frames[i].replace(".jpg", ".png"))
                cv2.imwrite(out_path, blank_mask)
                continue

        print(f"Finished sequence: {seq_name}")

print("\nAll sequences processed! Masks & slot maps saved to:")
print(f"  - Predictions: {OUT_DIR}")
print(f"  - Slot maps:   {SLOT_OUT}")