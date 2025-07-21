import os
import torch,torchvision
import torchvision.transforms as T
from clip_util import CLIPWrapper
from models.clip import clip_vit_l14
from models import VQVAE, build_vae_var
from condition_model import InjExtraCondAlignModel
from tokenizer import tokenize
from PIL import Image as PImage
from pathlib import Path

# 定义normalize函数，将[0,1]转换为[-1,1]
def normalize_01_into_pm1(tensor):
    return tensor.mul(2.0).sub(1.0)

# ========== Setup ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DEPTH = 16  # or 20, 24, 30
vae_ckpt = '/content/drive/MyDrive/huawei/var/vae_ch160v4096z32.pth'
var_ckpt = '/content/drive/MyDrive/huawei/VAR-TACO/inj_1/ar-ckpt-last-0715.pth'
condition_ckpt = '/content/drive/MyDrive/huawei/VAR-TACO/inj_1/condalign2-ckpt-best.pth'

    
# ========== Load models ==========
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device=device, patch_nums=(1,2,3,4,5,6,8,10,13,16),
    n_cond_embed=768, depth=MODEL_DEPTH, shared_aln=False,
)

# 加载VAE
if os.path.exists(vae_ckpt):
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    print(f"✅ VAE loaded from {vae_ckpt}")
else:
    print(f"❌ VAE checkpoint not found: {vae_ckpt}")

# 加载VAR
if os.path.exists(var_ckpt):
    var_checkpoint = torch.load(var_ckpt, map_location='cpu')
    if 'trainer' in var_checkpoint and 'var_wo_ddp' in var_checkpoint['trainer']:
        var.load_state_dict(var_checkpoint['trainer']['var_wo_ddp'], strict=False)
    else:
        var.load_state_dict(var_checkpoint, strict=False)
    print(f"✅ VAR loaded from {var_ckpt}")
else:
    print(f"❌ VAR checkpoint not found: {var_ckpt}")

vae.eval().to(device)
var.eval().to(device)
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)

# 加载CLIP
clip = clip_vit_l14(pretrained=True).eval().to(device)
clip = CLIPWrapper(clip, normalize=True)

# 加载Condition Model
condition_model = InjExtraCondAlignModel(image_dim=32, text_dim=768, out_dim=768).to(device)

if os.path.exists(condition_ckpt):
    print(f"Loading condition model from {condition_ckpt}")
    condition_checkpoint = torch.load(condition_ckpt, map_location=device)
    if 'model_state_dict' in condition_checkpoint:
            condition_checkpoint = condition_checkpoint['model_state_dict']
    
    # 处理DDP保存的模型（参数名有module.前缀）
    if isinstance(condition_checkpoint, dict) and any(key.startswith('module.') for key in condition_checkpoint.keys()):
        new_checkpoint = {}
        for key, value in condition_checkpoint.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 移除'module.'前缀
                new_checkpoint[new_key] = value
            else:
                new_checkpoint[key] = value
        condition_checkpoint = new_checkpoint
    
    condition_model.load_state_dict(condition_checkpoint)
    print(f"✅ Condition model loaded successfully!")
else:
    print(f"❌ Condition model checkpoint not found: {condition_ckpt}")

condition_model.eval()
for p in condition_model.parameters(): p.requires_grad_(False)

# ========== Image-Text Pairs ==========
kodak_dir = "/content/drive/MyDrive/huawei/kodak"  # 更新路径
output_dir = "/content/drive/MyDrive/huawei/VAR-TACO/recon_kodak"
os.makedirs(output_dir, exist_ok=True)

kodak_data = [
    ("kodim01.png", "a brick building with red doors and windows"),
    ("kodim02.png", "a red door with a metal door knocker on it"),
    ("kodim03.png", "a group of hats on the side of a wall"),
    ("kodim04.png", "a woman wearing a red hat and a red dress"),
    ("kodim05.png", "a group of people on dirt bikes in a race"),
    ("kodim06.png", "a boat floating in the water in the ocean"),
    ("kodim07.png", "a pink flower in front of a window"),
    ("kodim08.png", "a group of older buildings in a city"),
    ("kodim09.png", "a group of small sailboats in the water"),
    ("kodim10.png", "a group of sailboats in the water"),
    ("kodim11.png", "a boat in the water next to a pier"),
    ("kodim12.png", "a man and a woman walking on the beach"),
    ("kodim13.png", "a stream of water with trees and mountains in the background"),
    ("kodim14.png", "a group of people in a raft on a river"),
    ("kodim15.png", "a young girl with paint on her face"),
    ("kodim16.png", "a large body of water with palm trees on an island"),
    ("kodim17.png", "a statue of a woman holding a coconut"),
    ("kodim18.png", "a woman in a dress holding an umbrella"),
    ("kodim19.png", "a lighthouse next to a white picket fence"),
    ("kodim20.png", "a small plane sitting on the grass in a field"),
    ("kodim21.png", "a lighthouse on a rocky island in the ocean"),
    ("kodim22.png", "a red barn sitting next to a body of water"),
    ("kodim23.png", "two colorful parrots standing next to each other"),
    ("kodim24.png", "a house with a painting on the side of it"),
]

# ========== Preprocess ==========
# VAE输入预处理（与训练时保持一致）
image_preprocess = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
])

# ========== Sampling Loop ==========
for fname, caption in kodak_data:
    try:
        print(f"Processing {fname} ...")
        img_path = os.path.join(kodak_dir, fname)
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}, skipping...")
            continue
            
        image = PImage.open(img_path).convert("RGB")
        
        # 图像预处理（用于VAE编码）
        image_tensor = image_preprocess(image).unsqueeze(0).to(device)
        
        # 
        batch_size = 16
        image_tensor = image_tensor.expand(batch_size, -1, -1, -1)

        # 获取fhat_k和text embedding
        fhat_k = vae.img_to_fhat_k(image_tensor, k=5)  # (B, 32, H, W)
        text_token = tokenize([caption] * batch_size).to(device)
        text_embeddings = clip.encode_text(text_token)  # (B, 768)

        # 使用condition model融合特征
        with torch.no_grad():
            embed_fused = condition_model(text_embeddings, fhat_k)  # (B, 768)

        # Sampling - 优化参数
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device=='cuda')):
                recon = var.autoregressive_infer_from_fhat(
                    B=batch_size, embed_fused=embed_fused,
                    top_k=500, top_p=0.9, temperature=0.8, more_smooth=True  # 优化的参数
                )

        # Save image - 使用更好的布局
        grid = torchvision.utils.make_grid(recon, nrow=4, padding=2, pad_value=1.0, normalize=True)
        img = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        out_image = PImage.fromarray(img)
        out_image.save(os.path.join(output_dir, fname))
        print(f"✅ Saved: {os.path.join(output_dir, fname)}")
        
        # 可选：释放内存
        del fhat_k, text_embeddings, embed_fused, recon
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Failed on {fname}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n🎯 Processing completed! Results saved in {output_dir}/")
