import os
import sys
import pytest
import torch
import numpy as np
from unittest import mock

# モジュールをインポートするためのパスを設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# 依存モジュールのモックを作成
class MockDismantledBlock(torch.nn.Module):
    """DismantledBlockのモッククラス"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    
    def forward(self, x, *args, **kwargs):
        return x

# モックモジュールを設定
mock_mmditx = mock.MagicMock()
mock_mmditx.DismantledBlock = MockDismantledBlock
sys.modules['mmditx'] = mock_mmditx

# テスト対象のモジュールをインポート
from generator.dit_embedder import ControlNetEmbedder

class TestControlNetEmbedder:
    """ControlNetEmbedderクラスのユニットテスト"""
    
    @pytest.fixture
    def embedder(self):
        """テスト用のControlNetEmbedderインスタンスを準備"""
        # テストパラメータ
        img_size = 64
        patch_size = 2
        in_chans = 16
        attention_head_dim = 64
        num_attention_heads = 4
        pooled_projection_size = 2048
        num_layers = 3
        device = "cpu"
        dtype = torch.float32
        
        # インスタンス作成
        return ControlNetEmbedder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            pooled_projection_size=pooled_projection_size,
            num_layers=num_layers,
            device=device,
            dtype=dtype
        )
    
    def test_initialization(self, embedder):
        """初期化が正しく行われるかテスト"""
        # 基本的なプロパティの確認
        assert embedder.hidden_size == 4 * 64  # num_attention_heads * attention_head_dim
        assert len(embedder.transformer_blocks) == 3
        assert len(embedder.controlnet_blocks) == 3
        assert embedder.using_8b_controlnet is False
        
        # コンポーネントの型の確認
        assert isinstance(embedder.control_type, torch.Tensor)
        assert embedder.control_type.item() == 0  # デフォルトはblur (0)
    
    def test_forward(self, embedder):
        """forwardメソッドが正しく動作するかテスト"""
        # 入力テンソルの作成
        batch_size = 2
        hidden_size = embedder.hidden_size
        seq_len = 16  # 4x4のパッチ分割を想定
        
        x = torch.randn(batch_size, hidden_size, 4, 4)  # パッチ埋め込み後の形状を想定
        x_cond = torch.randn(batch_size, 16, 8, 8)      # 条件画像
        y = torch.randn(batch_size, 2048)               # コンディショニングベクトル
        timestep = torch.tensor([0.5])                   # タイムステップ
        scale = 1.0                                      # スケーリング係数
        
        # 依存するメソッドとクラスをモック
        with mock.patch.object(embedder.x_embedder, 'forward', return_value=x), \
             mock.patch.object(embedder.t_embedder, 'forward', return_value=torch.randn(batch_size, hidden_size)), \
             mock.patch.object(embedder.y_embedder, 'forward', return_value=torch.randn(batch_size, hidden_size)), \
             mock.patch.object(embedder.pos_embed_input, 'forward', return_value=torch.randn(batch_size, hidden_size, 4, 4)):
            
            # transformer_blocksとcontrolnet_blocksのforwardをモック
            for block in embedder.transformer_blocks:
                block.forward = mock.MagicMock(return_value=torch.randn(batch_size, seq_len, hidden_size))
            
            for block in embedder.controlnet_blocks:
                original_forward = block.forward
                block.forward = mock.MagicMock(return_value=torch.randn(batch_size, seq_len, hidden_size))
            
            # forwardメソッドを呼び出し
            outputs = embedder.forward(x, x_cond, y, scale, timestep)
            
            # 出力の検証
            assert isinstance(outputs, list)
            assert len(outputs) == 3  # 3つのブロックからの出力
            for output in outputs:
                assert isinstance(output, torch.Tensor)
    
    def test_8b_controlnet_mode(self, embedder):
        """8Bモードで正しく動作するかテスト"""
        # 8Bモードを有効化
        embedder.using_8b_controlnet = True
        
        # 入力テンソルの作成
        batch_size = 2
        hidden_size = embedder.hidden_size
        seq_len = 16
        
        x = torch.randn(batch_size, hidden_size, 4, 4)
        x_cond = torch.randn(batch_size, 16, 8, 8)
        y = torch.randn(batch_size, 2048)
        timestep = torch.tensor([0.5])
        scale = 1.0
        
        # 依存するメソッドとクラスをモック
        with mock.patch.object(embedder.pos_embed_input, 'forward', return_value=torch.randn(batch_size, hidden_size, 4, 4)):
            
            # transformer_blocksとcontrolnet_blocksのforwardをモック
            for block in embedder.transformer_blocks:
                block.forward = mock.MagicMock(return_value=torch.randn(batch_size, seq_len, hidden_size))
            
            for block in embedder.controlnet_blocks:
                block.forward = mock.MagicMock(return_value=torch.randn(batch_size, seq_len, hidden_size))
            
            # forwardメソッドを呼び出し
            outputs = embedder.forward(x, x_cond, y, scale, timestep)
            
            # 出力の検証
            assert isinstance(outputs, list)
            assert len(outputs) == 3
            for output in outputs:
                assert isinstance(output, torch.Tensor)
                # スケーリングが適用されていることを確認
                assert output.shape == (batch_size, seq_len, hidden_size) 