import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# モックに使うクラスをインポート
import unittest.mock as mock

# モジュールをインポートするためのパスを設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# 外部依存モジュールのモック
sys.modules['dit_embedder'] = mock.MagicMock()
sys.modules['mmditx'] = mock.MagicMock()

# テスト対象のモジュールをインポート
from generator.sd3_impls import (
    ModelSamplingDiscreteFlow,
    BaseModel,
    CFGDenoiser,
    SkipLayerCFGDenoiser,
    SD3LatentFormat,
    sample_euler,
    sample_dpmpp_2m,
    SDVAE,
    ResnetBlock,
    AttnBlock,
    Downsample,
    Upsample,
    VAEEncoder,
    VAEDecoder
)

# テスト設定
MODEL_FOLDER = "/home/persona/Tempdata/HuggingFace"
MODEL_FILE = "/home/persona/Tempdata/HuggingFace/sd3.5_large.safetensors"

# モックの設定
class MockSafeTensor:
    def __init__(self):
        self.tensors = {
            "x_embedder.proj.weight": torch.randn(256, 16, 2, 2),
            "pos_embed": torch.randn(1, 256, 256),
            "y_embedder.mlp.0.weight": torch.randn(1024, 2048),
            "context_embedder.weight": torch.randn(4096, 2048),
            "joint_blocks.0.context_block.attn.ln_k.weight": torch.randn(256),
        }
        self.keys_list = list(self.tensors.keys())
        self.keys_list.append("joint_blocks.0.x_block.attn2.ln_k.weight")

    def get_tensor(self, key):
        if key in self.tensors:
            return self.tensors[key]
        # 存在しないキーの場合はダミーデータを返す
        return torch.randn(256, 256)
    
    def keys(self):
        return self.keys_list

# ModelSamplingDiscreteFlowのテスト
class TestModelSamplingDiscreteFlow:
    @pytest.fixture
    def model(self):
        return ModelSamplingDiscreteFlow(shift=1.0)
    
    def test_initialization(self, model):
        assert model.shift == 1.0
        assert model.sigmas.shape[0] == 1000
    
    def test_sigma_min_max(self, model):
        assert model.sigma_min == model.sigmas[0]
        assert model.sigma_max == model.sigmas[-1]
    
    def test_timestep(self, model):
        sigma = torch.tensor(0.5)
        timestep = model.timestep(sigma)
        assert timestep == sigma * 1000
    
    def test_sigma(self, model):
        timestep = torch.tensor(500)
        sigma = model.sigma(timestep)
        expected = timestep / 1000.0
        assert torch.isclose(sigma, expected)
    
    def test_calculate_denoised(self, model):
        sigma = torch.tensor([0.5])
        model_output = torch.ones((1, 3, 64, 64))
        model_input = torch.ones((1, 3, 64, 64)) * 2
        result = model.calculate_denoised(sigma, model_output, model_input)
        # model_input - model_output * sigma
        expected = torch.ones((1, 3, 64, 64)) * 1.5
        assert torch.allclose(result, expected)
    
    def test_noise_scaling(self, model):
        sigma = torch.tensor(0.5)
        noise = torch.ones((1, 3, 64, 64))
        latent = torch.ones((1, 3, 64, 64)) * 2
        result = model.noise_scaling(sigma, noise, latent, False)
        # sigma * noise + (1.0 - sigma) * latent
        expected = 0.5 * 1 + 0.5 * 2
        assert torch.allclose(result, torch.ones((1, 3, 64, 64)) * expected)

# SD3LatentFormatのテスト
class TestSD3LatentFormat:
    @pytest.fixture
    def latent_format(self):
        return SD3LatentFormat()
    
    def test_initialization(self, latent_format):
        assert latent_format.scale_factor == 1.5305
        assert latent_format.shift_factor == 0.0609
    
    def test_process_in(self, latent_format):
        latent = torch.ones((1, 16, 64, 64))
        result = latent_format.process_in(latent)
        expected = (latent - latent_format.shift_factor) * latent_format.scale_factor
        assert torch.allclose(result, expected)
    
    def test_process_out(self, latent_format):
        latent = torch.ones((1, 16, 64, 64))
        result = latent_format.process_out(latent)
        expected = (latent / latent_format.scale_factor) + latent_format.shift_factor
        assert torch.allclose(result, expected)
    
    def test_decode_latent_to_preview(self, latent_format):
        x0 = torch.rand((1, 16, 64, 64))
        result = latent_format.decode_latent_to_preview(x0)
        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)

# ResnetBlockのテスト
class TestResnetBlock:
    def test_initialization(self):
        block = ResnetBlock(in_channels=64, out_channels=128)
        assert block.in_channels == 64
        assert block.out_channels == 128
        assert block.nin_shortcut is not None
    
    def test_initialization_same_channels(self):
        block = ResnetBlock(in_channels=64, out_channels=64)
        assert block.in_channels == 64
        assert block.out_channels == 64
        assert block.nin_shortcut is None
    
    def test_forward(self):
        block = ResnetBlock(in_channels=64, out_channels=64)
        x = torch.randn((1, 64, 32, 32))
        # モックでforward呼び出しをテスト
        with mock.patch.object(block.norm1, 'forward', return_value=x), \
             mock.patch.object(block.conv1, 'forward', return_value=x), \
             mock.patch.object(block.norm2, 'forward', return_value=x), \
             mock.patch.object(block.conv2, 'forward', return_value=x), \
             mock.patch.object(block.swish, 'forward', return_value=x):
            result = block.forward(x)
            assert result.shape == x.shape
            # 入出力チャンネル数が同じなので、x + hidden（= x）
            assert torch.allclose(result, x + x)

# VAEクラス関連のテスト
class TestVAE:
    @pytest.fixture
    def vae(self):
        return SDVAE(dtype=torch.float32, device="cpu")
    
    def test_initialization(self, vae):
        assert isinstance(vae.encoder, VAEEncoder)
        assert isinstance(vae.decoder, VAEDecoder)
    
    def test_encode_decode(self, vae):
        # エンコード・デコードのモックテスト
        with mock.patch.object(vae.encoder, 'forward', return_value=torch.randn(1, 32, 16, 16)), \
             mock.patch.object(vae.decoder, 'forward', return_value=torch.randn(1, 3, 128, 128)):
            
            # エンコードのテスト
            image = torch.randn((1, 3, 128, 128))
            latent = vae.encode(image)
            assert latent.shape[0] == 1
            assert latent.shape[1] > 0
            
            # デコードのテスト
            latent = torch.randn((1, 16, 16, 16))
            decoded = vae.decode(latent)
            assert decoded.shape[1] == 3  # RGB channels

# サンプラー関数のモックテスト
@mock.patch('torch.autocast')
def test_sample_euler(mock_autocast):
    model = mock.MagicMock()
    model.return_value = torch.zeros((1, 16, 64, 64))
    
    x = torch.randn((1, 16, 64, 64))
    sigmas = torch.linspace(1.0, 0.0, 11)
    
    with mock.patch('generator.sd3_impls.tqdm', lambda x, *args, **kwargs: x):
        result = sample_euler(model, x, sigmas)
    
    assert result.shape == x.shape
    assert model.call_count >= len(sigmas) - 1

@mock.patch('torch.autocast')
def test_sample_dpmpp_2m(mock_autocast):
    model = mock.MagicMock()
    model.return_value = torch.zeros((1, 16, 64, 64))
    
    x = torch.randn((1, 16, 64, 64))
    sigmas = torch.linspace(1.0, 0.0, 11)
    
    with mock.patch('generator.sd3_impls.tqdm', lambda x, *args, **kwargs: x):
        result = sample_dpmpp_2m(model, x, sigmas)
    
    assert result.shape == x.shape 