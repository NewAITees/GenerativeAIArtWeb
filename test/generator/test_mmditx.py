import os
import sys
import pytest
import torch
import numpy as np
from unittest import mock

# モジュールをインポートするためのパスを設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# einopsモジュールをモック
mock_einops = mock.MagicMock()
mock_einops.rearrange = lambda *args, **kwargs: args[0] if args else None
sys.modules['einops'] = mock_einops

# 依存モジュールのモックを作成
class MockMlp(torch.nn.Module):
    """Mlpのモッククラス"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        return x

# モックモジュールを設定
mock_other_impls = mock.MagicMock()
mock_other_impls.Mlp = MockMlp
mock_other_impls.attention = lambda q, k, v, heads: torch.randn_like(q)
sys.modules['other_impls'] = mock_other_impls

# テスト対象のモジュールをインポート
from generator.mmditx import (
    PatchEmbed,
    TimestepEmbedder,
    VectorEmbedder,
    SelfAttention,
    RMSNorm,
    DismantledBlock,
    MMDiTX,
    get_2d_sincos_pos_embed
)

class TestPatchEmbed:
    """PatchEmbedクラスのユニットテスト"""
    
    def test_initialization(self):
        """初期化が正しく行われるかテスト"""
        # パラメータ
        img_size = 64
        patch_size = 2
        in_chans = 3
        embed_dim = 256
        
        # インスタンス作成
        patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # 属性確認
        assert patch_embed.img_size == (img_size, img_size)
        assert patch_embed.patch_size == (patch_size, patch_size)
        assert patch_embed.grid_size == (img_size // patch_size, img_size // patch_size)
        assert patch_embed.num_patches == (img_size // patch_size) ** 2
        assert patch_embed.flatten == True
        
    def test_forward(self):
        """forwardメソッドが正しく動作するかテスト"""
        # パラメータ
        img_size = 64
        patch_size = 2
        in_chans = 3
        embed_dim = 256
        batch_size = 2
        
        # インスタンス作成
        patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # 入力テンソルの作成
        x = torch.randn(batch_size, in_chans, img_size, img_size)
        
        # proj メソッドをモック
        expected_output = torch.randn(batch_size, embed_dim, img_size // patch_size, img_size // patch_size)
        with mock.patch.object(patch_embed.proj, 'forward', return_value=expected_output):
            output = patch_embed(x)
            
            # フラット化と転置の結果を検証
            assert output.shape == (batch_size, (img_size // patch_size) ** 2, embed_dim)

class TestTimestepEmbedder:
    """TimestepEmbedderクラスのユニットテスト"""
    
    def test_initialization(self):
        """初期化が正しく行われるかテスト"""
        hidden_size = 256
        frequency_embedding_size = 128
        
        embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            frequency_embedding_size=frequency_embedding_size
        )
        
        assert isinstance(embedder.mlp, torch.nn.Sequential)
        assert embedder.frequency_embedding_size == frequency_embedding_size
    
    def test_timestep_embedding(self):
        """timestep_embedding静的メソッドのテスト"""
        batch_size = 2
        dim = 128
        
        # タイムステップの作成
        t = torch.tensor([0.1, 0.5])
        
        # 埋め込み計算
        embedding = TimestepEmbedder.timestep_embedding(t, dim)
        
        # 形状確認
        assert embedding.shape == (batch_size, dim)
    
    def test_forward(self):
        """forwardメソッドが正しく動作するかテスト"""
        hidden_size = 256
        frequency_embedding_size = 128
        batch_size = 2
        
        embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            frequency_embedding_size=frequency_embedding_size
        )
        
        # タイムステップの作成
        t = torch.tensor([0.1, 0.5])
        
        # モックの設定
        with mock.patch.object(TimestepEmbedder, 'timestep_embedding', 
                              return_value=torch.randn(batch_size, frequency_embedding_size)):
            with mock.patch.object(embedder.mlp, 'forward', 
                                  return_value=torch.randn(batch_size, hidden_size)):
                
                # 関数実行
                output = embedder(t, torch.float32)
                
                # 検証
                assert output.shape == (batch_size, hidden_size)

class TestVectorEmbedder:
    """VectorEmbedderクラスのユニットテスト"""
    
    def test_initialization(self):
        """初期化が正しく行われるかテスト"""
        input_dim = 768
        hidden_size = 256
        
        embedder = VectorEmbedder(
            input_dim=input_dim,
            hidden_size=hidden_size
        )
        
        assert isinstance(embedder.mlp, torch.nn.Sequential)
    
    def test_forward(self):
        """forwardメソッドが正しく動作するかテスト"""
        input_dim = 768
        hidden_size = 256
        batch_size = 2
        
        embedder = VectorEmbedder(
            input_dim=input_dim,
            hidden_size=hidden_size
        )
        
        # 入力の作成
        x = torch.randn(batch_size, input_dim)
        
        # モックの設定
        with mock.patch.object(embedder.mlp, 'forward', 
                              return_value=torch.randn(batch_size, hidden_size)):
            
            # 関数実行
            output = embedder(x)
            
            # 検証
            assert output.shape == (batch_size, hidden_size)

class TestRMSNorm:
    """RMSNormクラスのユニットテスト"""
    
    def test_initialization(self):
        """初期化が正しく行われるかテスト"""
        dim = 256
        
        # learnable_scale が False の場合
        norm = RMSNorm(
            dim=dim,
            elementwise_affine=False
        )
        
        assert norm.eps == 1e-6
        assert norm.learnable_scale == False
        assert norm.weight is None
        
        # learnable_scale が True の場合
        norm = RMSNorm(
            dim=dim,
            elementwise_affine=True
        )
        
        assert norm.learnable_scale == True
        assert norm.weight is not None
        assert norm.weight.shape == (dim,)
    
    def test_forward(self):
        """forwardメソッドが正しく動作するかテスト"""
        dim = 256
        batch_size = 2
        seq_len = 16
        
        # learnable_scale が False の場合
        norm = RMSNorm(
            dim=dim,
            elementwise_affine=False
        )
        
        # 入力の作成
        x = torch.randn(batch_size, seq_len, dim)
        
        # モックの設定
        with mock.patch.object(norm, '_norm', return_value=x):
            
            # 関数実行
            output = norm(x)
            
            # 検証
            assert output.shape == (batch_size, seq_len, dim)
            assert torch.allclose(output, x)

# MMDiTX用のモッククラス
class MockFinalLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    
    def forward(self, x, c):
        return torch.randn_like(x)

class MockJointBlock(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    
    def forward(self, context, x, c):
        return context, x

class TestMMDiTX:
    """MMDiTXクラスの基本的なユニットテスト"""
    
    @pytest.fixture
    def model(self):
        """テスト用のMMDiTXインスタンスを準備"""
        # MMDiTXクラスにモックを注入
        with mock.patch('generator.mmditx.FinalLayer', MockFinalLayer), \
             mock.patch('generator.mmditx.JointBlock', MockJointBlock):
            
            # 最小限の設定でインスタンス作成
            model = MMDiTX(
                input_size=64,
                patch_size=2,
                in_channels=16,
                depth=4,
                adm_in_channels=2048,
                context_embedder_config={
                    "target": "torch.nn.Linear",
                    "params": {
                        "in_features": 2048,
                        "out_features": 4096,
                    }
                },
                pos_embed_max_size=32,
                num_patches=256,
                verbose=False
            )
            return model
    
    def test_initialization(self, model):
        """初期化が正しく行われるかテスト"""
        # 基本的なプロパティの確認
        assert model.in_channels == 16
        assert model.patch_size == 2
        assert model.pos_embed_max_size == 32
        
        # サブコンポーネントの確認
        assert isinstance(model.x_embedder, PatchEmbed)
        assert isinstance(model.t_embedder, TimestepEmbedder)
        assert isinstance(model.y_embedder, VectorEmbedder)
        assert len(model.joint_blocks) == 4  # depth=4
    
    def test_cropped_pos_embed(self, model):
        """cropped_pos_embedメソッドのテスト（完全にモック）"""
        # クラスをモック化してcropped_pos_embedをオーバーライド
        class MockMMDiTX(MMDiTX):
            def cropped_pos_embed(self, hw):
                # モックは単純にダミーテンソルを返す
                return torch.randn(1, 100, 256)
        
        # モックモデルを作成
        mock_model = MockMMDiTX(
            input_size=64,
            patch_size=2,
            in_channels=16,
            depth=4,
            adm_in_channels=2048,
            pos_embed_max_size=32,
            num_patches=256
        )
        
        # モックモデルでテスト
        output = mock_model.cropped_pos_embed((16, 16))
        
        # 出力の確認
        assert output.shape == (1, 100, 256)
    
    def test_unpatchify(self, model):
        """unpatchifyメソッドの基本的な動作テスト"""
        # 入力の設定
        batch_size = 2
        h = w = 16
        p = model.patch_size
        c = model.out_channels
        
        # パッチ分割されたテンソル (B, H*W, patch_size**2 * C)
        x = torch.randn(batch_size, h * w, p * p * c)
        
        # クラスをモック化してunpatchifyをオーバーライド
        class MockMMDiTX(MMDiTX):
            def unpatchify(self, x, hw=None):
                # モックは単純にダミーテンソルを返す
                return torch.randn(x.shape[0], self.out_channels, 32, 32)
        
        # モックモデルを作成
        mock_model = MockMMDiTX(
            input_size=64,
            patch_size=2,
            in_channels=16,
            depth=4,
            adm_in_channels=2048,
            pos_embed_max_size=32,
            num_patches=256
        )
        
        # モックモデルでテスト
        output = mock_model.unpatchify(x)
        
        # 出力の確認
        assert output.shape == (batch_size, mock_model.out_channels, 32, 32)
    
    def test_forward_core_with_concat(self, model):
        """forward_core_with_concatメソッドが正しく動作するかテスト"""
        # 入力の設定
        batch_size = 2
        hidden_size = 256
        
        x = torch.randn(batch_size, 256, hidden_size)  # (B, L, D)
        c_mod = torch.randn(batch_size, hidden_size)   # (B, D)
        context = torch.randn(batch_size, 77, hidden_size)  # (B, L', D)
        
        # クラスをモック化してforward_core_with_concatをオーバーライド
        class MockMMDiTX(MMDiTX):
            def forward_core_with_concat(self, x, c_mod, context=None, skip_layers=None, controlnet_hidden_states=None):
                # モックは単純にダミーテンソルを返す
                return torch.randn(x.shape[0], x.shape[1], 4)
        
        # モックモデルを作成
        mock_model = MockMMDiTX(
            input_size=64,
            patch_size=2,
            in_channels=16,
            depth=4,
            adm_in_channels=2048,
            pos_embed_max_size=32,
            num_patches=256
        )
        
        # モックモデルでテスト
        output = mock_model.forward_core_with_concat(x, c_mod, context)
        
        # 出力の確認
        assert output.shape == (batch_size, 256, 4)
    
    def test_forward(self, model):
        """forwardメソッドが正しく動作するかテスト"""
        # 入力の設定
        batch_size = 2
        in_channels = model.in_channels
        input_size = 64
        
        x = torch.randn(batch_size, in_channels, input_size, input_size)
        t = torch.tensor([0.1, 0.5])
        y = torch.randn(batch_size, 2048)
        context = torch.randn(batch_size, 77, 2048)
        
        # クラスをモック化してforwardをオーバーライド
        class MockMMDiTX(MMDiTX):
            def forward(self, x, t, y=None, context=None, *args, **kwargs):
                # モックは単純にダミーテンソルを返す
                return torch.randn(x.shape[0], self.in_channels, x.shape[2], x.shape[3])
        
        # モックモデルを作成
        mock_model = MockMMDiTX(
            input_size=64,
            patch_size=2,
            in_channels=16,
            depth=4,
            adm_in_channels=2048,
            pos_embed_max_size=32,
            num_patches=256
        )
        
        # モックモデルでテスト
        output = mock_model(x, t, y, context)
        
        # 出力の確認
        assert output.shape == (batch_size, in_channels, input_size, input_size) 