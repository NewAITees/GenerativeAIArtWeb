# NOTE: Must have folder `models` with the following files:
# - `clip_g.safetensors` (openclip bigG, same as SDXL)
# - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL)
# - `t5xxl.safetensors` (google T5-v1.1-XXL)
# - `sd3_medium.safetensors` (or whichever main MMDiT model file)
# Also can have
# - `sd3_vae.safetensors` (holds the VAE separately if needed)

import datetime
import math
import os
import pickle
import re
from typing import List, Optional

import fire
import numpy as np
import sd3_impls
import torch
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from PIL import Image
from safetensors import safe_open
from sd3_impls import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)
from tqdm import tqdm
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

#################################################################################################
### Wrappers for model parts
#################################################################################################


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self, model_folder: str):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        # t5xxl_fp16.safetensors を優先して使用し、存在しない場合は t5xxl.safetensors にフォールバック
        t5xxl_fp16_path = f"{model_folder}/t5xxl_fp16.safetensors"
        t5xxl_path = f"{model_folder}/t5xxl.safetensors"
        
        model_path = t5xxl_fp16_path if os.path.exists(t5xxl_fp16_path) else t5xxl_path
        
        with safe_open(
            model_path, framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)


CONTROLNET_MAP = {
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
}


class SD3:
    def __init__(
        self, model, shift, control_model_file=None, verbose=False, device="cpu"
    ):

        # NOTE 8B ControlNets were trained with a slightly different forward pass and conditioning,
        # so this is a flag to enable that logic.
        self.using_8b_controlnet = False

        with safe_open(model, framework="pt", device="cpu") as f:
            control_model_ckpt = None
            if control_model_file is not None:
                control_model_ckpt = safe_open(
                    control_model_file, framework="pt", device=device
                )
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cuda",
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cuda", torch.float16)
        if control_model_file is not None:
            control_model_ckpt = safe_open(
                control_model_file, framework="pt", device=device
            )
            self.model.control_model = self.model.control_model.to(device)
            load_into(
                control_model_ckpt,
                self.model.control_model,
                "",
                device,
                dtype=torch.float16,
                remap=CONTROLNET_MAP,
            )

            self.using_8b_controlnet = (
                self.model.control_model.y_embedder.mlp[0].in_features == 2048
            )
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet
        control_model_ckpt = None


class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)


#################################################################################################
### Main inference logic
#################################################################################################


# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = "a photo of a cat"
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 40
# Seed
SEED = 23
# SEEDTYPE = "fixed"
SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
# MODEL = "models/sd3_medium.safetensors"
# MODEL = "models/sd3.5_large_turbo.safetensors"
MODEL = "models/sd3.5_large.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# ControlNet
CONTROLNET_COND_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.8
# Output file path
OUTDIR = "outputs"
# SAMPLER
SAMPLER = "dpmpp_2m"
# MODEL FOLDER
MODEL_FOLDER = "models"


class SD3Inferencer:
    """Stable Diffusion 3.5 推論エンジン"""

    def __init__(self, model_path: Optional[str] = None):
        """初期化
        
        Args:
            model_path (Optional[str], optional): モデルファイルのパス. デフォルトはNone.
        """
        self.verbose = False
        self.model_path = model_path
        self.tokenizer = None
        self.t5xxl = None
        self.clip_l = None
        self.clip_g = None
        self.sd3 = None
        self.vae = None

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(
        self,
        model=MODEL,
        vae=VAEFile,
        shift=SHIFT,
        controlnet_ckpt=None,
        model_folder: str = MODEL_FOLDER,
        text_encoder_device: str = "cpu",
        verbose=False,
        load_tokenizers: bool = True,
    ):
        self.verbose = verbose
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        if load_tokenizers:
            print("Loading Google T5-v1-XXL...")
            self.t5xxl = T5XXL(model_folder, text_encoder_device, torch.float32)
            print("Loading OpenAI CLIP L...")
            self.clip_l = ClipL(model_folder)
            print("Loading OpenCLIP bigG...")
            self.clip_g = ClipG(model_folder, text_encoder_device)
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, controlnet_ckpt, verbose, "cuda")
        print("Loading VAE model...")
        self.vae = VAE(vae or model)
        print("Models loaded.")

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = float(sampling.sigma_max)
        end = float(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(float(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        self.print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        if isinstance(cond, dict):
            # 既に辞書形式の場合はそのまま返す
            return {
                "c_crossattn": cond["c_crossattn"].half().cuda(),
                "y": cond["y"].half().cuda()
            }
        # タプル形式の場合は変換する
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        controlnet_cond=None,
        denoise=1.0,
        skip_layer_config={},
        progress_callback=None
    ) -> torch.Tensor:
        """サンプリングを実行する
        
        Args:
            latent (torch.Tensor): 入力潜在表現
            seed (int): 乱数シード
            conditioning (tuple): 条件付け情報
            neg_cond (tuple): 負の条件付け情報
            steps (int): サンプリングステップ数
            cfg_scale (float): CFGスケール
            sampler (str, optional): サンプラー名. デフォルトは"dpmpp_2m"
            controlnet_cond (torch.Tensor, optional): ControlNet条件. デフォルトはNone
            denoise (float, optional): デノイズ率. デフォルトは1.0
            skip_layer_config (dict, optional): スキップレイヤー設定. デフォルトは{}
            progress_callback (callable, optional): 進捗コールバック関数. デフォルトはNone
            
        Returns:
            torch.Tensor: 生成された潜在表現
        """
        self.print("Sampling...")
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            "controlnet_cond": controlnet_cond,
        }
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        
        # カスタムコールバック関数を作成して、progress_callbackに中継
        def wrapped_callback(i, total):
            if progress_callback:
                progress_callback(i, total)
        
        denoiser = (
            SkipLayerCFGDenoiser
            if skip_layer_config.get("scale", 0) > 0
            else CFGDenoiser
        )
        
        # 進捗コールバックを渡す
        latent = sample_fn(
            denoiser(self.sd3.model, steps, skip_layer_config, progress_callback=wrapped_callback),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
            callback=wrapped_callback  # ここでコールバックを渡す
        )
        
        latent = SD3LatentFormat().process_out(latent)
        self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent

    def vae_encode(
        self, image, using_2b_controlnet: bool = False, controlnet_type: int = 0
    ) -> torch.Tensor:
        self.print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).cuda()
        if using_2b_controlnet:
            image_torch = image_torch * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_torch = image_torch * 255 * 0.5 + 0.5
        else:
            image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        self.print("Encoded")
        return latent

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        self.print("Decoded")
        return out_image

    def _image_to_latent(
        self,
        image,
        width,
        height,
        using_2b_controlnet: bool = False,
        controlnet_type: int = 0,
    ) -> torch.Tensor:
        image_data = Image.open(image)
        image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image_data, using_2b_controlnet, controlnet_type)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def gen_image(
        self,
        prompts=[PROMPT],
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        sampler=SAMPLER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        controlnet_cond_image=CONTROLNET_COND_IMAGE,
        init_image=INIT_IMAGE,
        denoise=DENOISE,
        skip_layer_config={},
    ):
        controlnet_cond = None
        if init_image:
            latent = self._image_to_latent(init_image, width, height)
        else:
            latent = self.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()
        if controlnet_cond_image:
            using_2b, control_type = False, 0
            if self.sd3.model.control_model is not None:
                using_2b = not self.sd3.using_8b_controlnet
                control_type = int(self.sd3.model.control_model.control_type.item())
            controlnet_cond = self._image_to_latent(
                controlnet_cond_image, width, height, using_2b, control_type
            )
        neg_cond = self.get_cond("")
        seed_num = None
        pbar = tqdm(enumerate(prompts), total=len(prompts), position=0, leave=True)
        for i, prompt in pbar:
            if seed_type == "roll":
                seed_num = seed if seed_num is None else seed_num + 1
            elif seed_type == "rand":
                seed_num = torch.randint(0, 100000, (1,)).item()
            else:  # fixed
                seed_num = seed
            conditioning = self.get_cond(prompt)
            sampled_latent = self.do_sampling(
                latent,
                seed_num,
                conditioning,
                neg_cond,
                steps,
                cfg_scale,
                sampler,
                controlnet_cond,
                denoise if init_image else 1.0,
                skip_layer_config,
            )
            image = self.vae_decode(sampled_latent)
            save_path = os.path.join(out_dir, f"{i:06d}.png")
            self.print(f"Saving to to {save_path}")
            image.save(save_path)
            self.print("Done")

    def run_inference(
        self,
        prompt: str,
        steps: int = 40,
        cfg_scale: float = 4.5,
        sampler: str = "euler",
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        progress_callback=None
    ) -> List[Image.Image]:
        """画像生成の推論を実行する
        
        Args:
            prompt (str): 生成プロンプト
            steps (int, optional): サンプリングステップ数. デフォルトは40.
            cfg_scale (float, optional): CFGスケール. デフォルトは4.5.
            sampler (str, optional): サンプラー名. デフォルトは"euler".
            width (int, optional): 画像の幅. デフォルトは1024.
            height (int, optional): 画像の高さ. デフォルトは1024.
            seed (int, optional): 乱数シード. デフォルトはNone（ランダム）.
            progress_callback (callable, optional): 進捗コールバック関数 (step, total_steps) を引数に取る. デフォルトはNone.
        
        Returns:
            List[Image.Image]: 生成された画像のリスト
        """
        # シードの設定
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        logger.info(f"画像生成開始: prompt='{prompt}', steps={steps}, cfg_scale={cfg_scale}, seed={seed}")
        
        # 空の条件付け
        neg_cond = self.get_cond("")
        
        # プロンプトの条件付け
        conditioning = self.get_cond(prompt)
        
        # 潜在表現の初期化
        latent = self.get_empty_latent(1, width, height, seed)
        
        # サンプリング
        sampled_latent = self.do_sampling(
            latent,
            seed,
            conditioning,
            neg_cond,
            steps,
            cfg_scale,
            sampler,
            None,  # controlnet_cond
            1.0,   # denoise
            {},    # skip_layer_config
            progress_callback=progress_callback
        )
        
        # VAEデコード
        image = self.vae_decode(sampled_latent)
        
        return [image]


CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_medium": {
        "shift": 3.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,
            "end": 0.20,
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    },
    "sd3.5_large": {
        "shift": 3.0,
        "steps": 40,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_large_turbo": {"shift": 3.0, "cfg": 1.0, "steps": 4, "sampler": "euler"},
    "sd3.5_large_controlnet_blur": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_canny": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_depth": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
}


@torch.no_grad()
def main(
    prompt=PROMPT,
    model=MODEL,
    out_dir=OUTDIR,
    postfix=None,
    seed=SEED,
    seed_type=SEEDTYPE,
    sampler=None,
    steps=None,
    cfg=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    controlnet_ckpt=None,
    controlnet_cond_image=None,
    vae=VAEFile,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    skip_layer_cfg=False,
    verbose=False,
    model_folder=MODEL_FOLDER,
    text_encoder_device="cpu",
    **kwargs,
):
    assert not kwargs, f"Unknown arguments: {kwargs}"

    config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {})
    _shift = shift or config.get("shift", 3)
    _steps = steps or config.get("steps", 50)
    _cfg = cfg or config.get("cfg", 5)
    _sampler = sampler or config.get("sampler", "dpmpp_2m")

    if skip_layer_cfg:
        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(model))[0], {}
        ).get("skip_layer_config", {})
        cfg = skip_layer_config.get("cfg", cfg)
    else:
        skip_layer_config = {}

    if controlnet_ckpt is not None:
        controlnet_config = CONFIGS.get(
            os.path.splitext(os.path.basename(controlnet_ckpt))[0], {}
        )
        _shift = shift or controlnet_config.get("shift", shift)
        _steps = steps or controlnet_config.get("steps", steps)
        _cfg = cfg or controlnet_config.get("cfg", cfg)
        _sampler = sampler or controlnet_config.get("sampler", sampler)

    inferencer = SD3Inferencer(model)

    inferencer.load(
        model,
        vae,
        _shift,
        controlnet_ckpt,
        model_folder,
        text_encoder_device,
        verbose,
    )

    if isinstance(prompt, str):
        if os.path.splitext(prompt)[-1] == ".txt":
            with open(prompt, "r") as f:
                prompts = [l.strip() for l in f.readlines()]
        else:
            prompts = [prompt]

    sanitized_prompt = re.sub(r"[^\w\-\.]", "_", prompt)
    out_dir = os.path.join(
        out_dir,
        (
            os.path.splitext(os.path.basename(model))[0]
            + (
                "_" + os.path.splitext(os.path.basename(controlnet_ckpt))[0]
                if controlnet_ckpt is not None
                else ""
            )
        ),
        os.path.splitext(os.path.basename(sanitized_prompt))[0][:50]
        + (postfix or datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")),
    )

    os.makedirs(out_dir, exist_ok=False)

    inferencer.gen_image(
        prompts,
        width,
        height,
        _steps,
        _cfg,
        _sampler,
        seed,
        seed_type,
        out_dir,
        controlnet_cond_image,
        init_image,
        denoise,
        skip_layer_config,
    )


def load_model(model_path: str, shift: float = 3.0, controlnet_ckpt: Optional[str] = None, verbose: bool = False, device: str = "cuda") -> BaseModel:
    """モデルをロードするヘルパー関数
    
    Args:
        model_path (str): モデルファイルのパス
        shift (float): シフト値、デフォルトは3.0
        controlnet_ckpt (str, optional): ControlNetモデルのパス
        verbose (bool): 詳細なログを出力するかどうか
        device (str): 使用するデバイス
        
    Returns:
        BaseModel: ロードされたSD3モデル
    """
    return SD3(model_path, shift, controlnet_ckpt, verbose, device).model

def load_tokenizer() -> SD3Tokenizer:
    """トークナイザーをロードするヘルパー関数
    
    Returns:
        SD3Tokenizer: ロードされたトークナイザー
    """
    return SD3Tokenizer()

def load_vae(model_path: str, dtype: torch.dtype = torch.float16) -> SDVAE:
    """VAEモデルをロードするヘルパー関数
    
    Args:
        model_path (str): モデルファイルのパス
        dtype (torch.dtype): モデルのデータ型
        
    Returns:
        SDVAE: ロードされたVAEモデル
    """
    return VAE(model_path, dtype).model


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs with progress tracking"""

    def __init__(self, model, steps=None, skip_layer_config=None, progress_callback=None):
        """Initialize the CFG Denoiser
        
        Args:
            model: The base model to use for denoising
            steps (int, optional): Total number of steps. Defaults to None.
            skip_layer_config (dict, optional): Configuration for layer skipping. Defaults to None.
            progress_callback (callable, optional): Progress callback function. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.steps = steps
        self.skip_layer_config = skip_layer_config or {}
        self.progress_callback = progress_callback
        self.current_step = 0

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond=None,
        uncond=None,
        cond_scale=None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with progress tracking
        
        Args:
            x (torch.Tensor): Input tensor
            sigma (torch.Tensor): Noise level
            cond (dict, optional): Conditioning. Defaults to None.
            uncond (dict, optional): Unconditioning. Defaults to None.
            cond_scale (float, optional): Conditioning scale. Defaults to None.
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Denoised output
        """
        # Update progress
        self.current_step += 1
        if self.progress_callback is not None:
            total_steps = self.steps or 1
            self.progress_callback(self.current_step, total_steps)

        # Handle unconditional case
        if cond is None or uncond is None:
            return self.model(x, sigma, **kwargs)

        # Run conditional and unconditional in a batch
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        c_crossattn = torch.cat([cond["c_crossattn"], uncond["c_crossattn"]])
        y = torch.cat([cond["y"], uncond["y"]])

        # Apply model
        out = self.model.apply_model(
            x_in,
            sigma_in,
            c_crossattn=c_crossattn,
            y=y,
            **kwargs
        )

        # Split results and apply conditioning
        out_cond, out_uncond = out.chunk(2)
        return out_uncond + (out_cond - out_uncond) * cond_scale


if __name__ == "__main__":
    fire.Fire(main)