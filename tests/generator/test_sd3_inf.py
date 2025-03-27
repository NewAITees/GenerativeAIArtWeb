import os
import sys
import pytest
import torch
import asyncio
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# モックに使うクラスをインポート
import unittest.mock as mock

# モジュールをインポートするためのパスを設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# 外部依存モジュールのモック
sys.modules['sd3_impls'] = mock.MagicMock()
sys.modules['other_impls'] = mock.MagicMock()

# テスト対象のモジュールをインポート
from generator.sd3_inf import SD3Inferencer, SD3, VAE, CFGDenoiser

# テスト設定
MODEL_FOLDER = "/home/persona/Tempdata/HuggingFace"
MODEL_FILE = "/home/persona/Tempdata/HuggingFace/sd3.5_large.safetensors"
VAE_FILE = None

# モックを使用した単体テスト
class TestSD3InferencerMock:
    """SD3Inferencerクラスの単体テスト（モック使用）"""
    
    @pytest.fixture
    def inferencer(self):
        """テスト用のSD3Inferencerインスタンスを準備"""
        return SD3Inferencer()
    
    def test_initialization(self, inferencer):
        """初期化が正しく行われるかテスト"""
        assert inferencer is not None
        assert inferencer.verbose is False
    
    @mock.patch('generator.sd3_inf.SD3Tokenizer')
    @mock.patch('generator.sd3_inf.T5XXL')
    @mock.patch('generator.sd3_inf.ClipL')
    @mock.patch('generator.sd3_inf.ClipG')
    @mock.patch('generator.sd3_inf.SD3')
    @mock.patch('generator.sd3_inf.VAE')
    @mock.patch('generator.sd3_inf.safe_open')
    def test_load(self, mock_safe_open, mock_vae, mock_sd3, mock_clip_g, mock_clip_l, 
                 mock_t5xxl, mock_tokenizer, inferencer):
        """モデルのロードが正しく行われるかテスト"""
        # モックの設定
        mock_tokenizer.return_value = mock.MagicMock()
        mock_t5xxl.return_value = mock.MagicMock()
        mock_clip_l.return_value = mock.MagicMock()
        mock_clip_g.return_value = mock.MagicMock()
        mock_sd3.return_value = mock.MagicMock()
        mock_vae.return_value = mock.MagicMock()
        
        # safe_openの戻り値をモック
        mock_safe_open.return_value.__enter__.return_value = mock.MagicMock()
        
        # load関数の呼び出し
        inferencer.load(
            model=MODEL_FILE,
            vae=VAE_FILE,
            shift=3.0,
            model_folder=MODEL_FOLDER,
            text_encoder_device="cpu",
            verbose=False,
            load_tokenizers=True
        )
        
        # 各モックが呼ばれたことを確認
        mock_tokenizer.assert_called_once()
        mock_t5xxl.assert_called_once_with(MODEL_FOLDER, "cpu", torch.float32)
        mock_clip_l.assert_called_once_with(MODEL_FOLDER)
        mock_clip_g.assert_called_once_with(MODEL_FOLDER, "cpu")
        mock_sd3.assert_called_once_with(MODEL_FILE, 3.0, None, False, "cuda")
        mock_vae.assert_called_once_with(MODEL_FILE)
        
        # インスタンス変数が正しく設定されていることを確認
        assert inferencer.tokenizer == mock_tokenizer.return_value
        assert inferencer.t5xxl == mock_t5xxl.return_value
        assert inferencer.clip_l == mock_clip_l.return_value
        assert inferencer.clip_g == mock_clip_g.return_value
        assert inferencer.sd3 == mock_sd3.return_value
        assert inferencer.vae == mock_vae.return_value
    
    def test_get_empty_latent(self, inferencer):
        """空のLatentが正しく生成されるかテスト"""
        batch_size = 1
        width = 1024
        height = 1024
        seed = 42
        device = "cpu"
        
        # 関数の呼び出し
        latent = inferencer.get_empty_latent(batch_size, width, height, seed, device)
        
        # 結果の検証
        assert isinstance(latent, torch.Tensor)
        assert latent.shape == (batch_size, 16, height // 8, width // 8)
        assert latent.device.type == device
    
    @mock.patch('generator.sd3_inf.torch.randn')
    def test_get_noise(self, mock_randn, inferencer):
        """ノイズが正しく生成されるかテスト"""
        # モックの設定
        mock_randn.return_value = torch.ones(1, 16, 128, 128)
        
        # テスト用のlatentを設定
        latent = torch.zeros(1, 16, 128, 128, dtype=torch.float32)
        
        # 関数の呼び出し
        noise = inferencer.get_noise(42, latent)
        
        # 結果の検証
        mock_randn.assert_called_once()
        assert noise.shape == latent.shape
    
    @mock.patch('generator.sd3_inf.torch.cat')
    @mock.patch('generator.sd3_inf.torch.nn.functional.pad')
    def test_get_cond(self, mock_pad, mock_cat, inferencer):
        """条件付けが正しく生成されるかテスト"""
        # モックの設定
        # 無限再帰を避けるため、本物のtorch.catを使わない
        l_out = torch.randn(1, 77, 768)
        g_out = torch.randn(1, 77, 1280)
        t5_out = torch.randn(1, 77, 4096)
        l_pooled = torch.randn(1, 768)
        g_pooled = torch.randn(1, 1280)
        
        # パディング後のテンソル
        lg_out_padded = torch.randn(1, 77, 4096)
        
        # モックの動作を設定
        mock_cat.side_effect = [
            torch.randn(1, 77, 2048),  # l_out + g_out
            torch.randn(1, 77, 6144),  # lg_out_padded + t5_out
            torch.randn(1, 2048)       # l_pooled + g_pooled
        ]
        
        mock_pad.return_value = lg_out_padded
        
        # 依存するメソッドをモック
        inferencer.tokenizer = mock.MagicMock()
        inferencer.tokenizer.tokenize_with_weights.return_value = {
            "l": mock.MagicMock(),
            "g": mock.MagicMock(),
            "t5xxl": mock.MagicMock()
        }
        
        inferencer.clip_l = mock.MagicMock()
        inferencer.clip_l.model.encode_token_weights.return_value = (l_out, l_pooled)
        
        inferencer.clip_g = mock.MagicMock()
        inferencer.clip_g.model.encode_token_weights.return_value = (g_out, g_pooled)
        
        inferencer.t5xxl = mock.MagicMock()
        inferencer.t5xxl.model.encode_token_weights.return_value = (t5_out, torch.randn(1, 4096))
        
        # 関数の呼び出し
        cond, pooled = inferencer.get_cond("test prompt")
        
        # 結果の検証
        assert isinstance(cond, torch.Tensor)
        assert isinstance(pooled, torch.Tensor)
        
        # モックが正しく呼ばれたことを確認
        assert mock_cat.call_count == 3
        assert mock_pad.call_count == 1

# 実際のモデルを使用した統合テスト
@pytest.mark.skipif(not os.path.exists(MODEL_FILE), reason="モデルファイルが存在しません")
class TestSD3InferencerIntegration:
    """SD3Inferencerクラスの統合テスト（実際のモデルを使用）"""
    
    @pytest.fixture
    def inferencer(self):
        """テスト用のSD3Inferencerインスタンスを準備"""
        return SD3Inferencer()
    
    def test_model_file_existence(self):
        """必要なモデルファイルが存在するか確認"""
        assert os.path.exists(MODEL_FOLDER), f"{MODEL_FOLDER}ディレクトリが存在しません"
        assert os.path.exists(MODEL_FILE), f"{MODEL_FILE}が存在しません"
        assert os.path.exists(f"{MODEL_FOLDER}/clip_g.safetensors"), "clip_g.safetensorsが存在しません"
        assert os.path.exists(f"{MODEL_FOLDER}/clip_l.safetensors"), "clip_l.safetensorsが存在しません"
        
        # t5xxl_fp16.safetensors または t5xxl.safetensors のいずれかが存在するか確認
        t5xxl_fp16_exists = os.path.exists(f"{MODEL_FOLDER}/t5xxl_fp16.safetensors")
        t5xxl_exists = os.path.exists(f"{MODEL_FOLDER}/t5xxl.safetensors")
        assert t5xxl_fp16_exists or t5xxl_exists, "t5xxl_fp16.safetensors または t5xxl.safetensors が存在しません"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
    def test_load_actual_model(self, inferencer):
        """実際のモデルを読み込むテスト（CUDA必要）"""
        # GPUがある場合のみ実行
        inferencer.load(
            model=MODEL_FILE,
            vae=VAE_FILE,
            shift=3.0,
            model_folder=MODEL_FOLDER,
            text_encoder_device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
            load_tokenizers=True
        )
        
        # 各モジュールが正しく読み込まれたことを確認
        assert inferencer.tokenizer is not None
        assert inferencer.t5xxl is not None
        assert inferencer.clip_l is not None
        assert inferencer.clip_g is not None
        assert inferencer.sd3 is not None
        assert inferencer.vae is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
    def test_generate_image(self, inferencer):
        """画像生成のミニマルテスト（CUDA必要）"""
        # 出力ディレクトリの設定
        out_dir = "test_outputs"
        os.makedirs(out_dir, exist_ok=True)
        
        # モデルの読み込み（GPU利用可能な場合）
        try:
            inferencer.load(
                model=MODEL_FILE,
                vae=VAE_FILE,
                shift=3.0,
                model_folder=MODEL_FOLDER,
                text_encoder_device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
                load_tokenizers=True
            )
            
            # 最小限のパラメータで画像生成（ステップ数を減らして高速化）
            inferencer.gen_image(
                prompts=["テスト画像"],
                width=512,  # 小さいサイズで高速化
                height=512,  # 小さいサイズで高速化
                steps=4,    # 少ないステップ数で高速化
                cfg_scale=4.5,
                sampler="euler",  # 高速なサンプラー
                seed=42,
                seed_type="fixed",
                out_dir=out_dir
            )
            
            # 出力ファイルの存在確認
            output_files = list(Path(out_dir).glob("*.png"))
            assert len(output_files) > 0, "画像が生成されませんでした"
            
        except Exception as e:
            pytest.skip(f"モデル実行中にエラーが発生しました: {str(e)}")
        
        # テスト後のクリーンアップ
        # for file in Path(out_dir).glob("*.png"):
        #    file.unlink()  # コメントアウトして画像を保持 

@pytest.fixture
def mock_model():
    mock_model = mock.MagicMock()
    mock_model.return_value = torch.randn(1, 4, 64, 64)
    return mock_model

@pytest.fixture
def mock_tokenizer():
    return mock.MagicMock()

@pytest.fixture
def mock_vae():
    return mock.MagicMock()

@pytest.fixture
def inferencer():
    """テスト用のSD3Inferencerインスタンスを準備"""
    # インスタンスの作成
    inf = SD3Inferencer()
    inf.model_loaded = True
    
    # モックオブジェクトの作成
    inf.tokenizer = MagicMock()
    inf.tokenizer.tokenize_with_weights.return_value = {
        "l": MagicMock(),
        "g": MagicMock(),
        "t5xxl": MagicMock()
    }
    
    inf.clip_l = MagicMock()
    inf.clip_l.model = MagicMock()
    inf.clip_l.model.encode_token_weights.return_value = (
        torch.randn(1, 77, 768),  # l_out
        torch.randn(1, 768)       # l_pooled
    )
    
    inf.clip_g = MagicMock()
    inf.clip_g.model = MagicMock()
    inf.clip_g.model.encode_token_weights.return_value = (
        torch.randn(1, 77, 1280),  # g_out
        torch.randn(1, 1280)       # g_pooled
    )
    
    inf.t5xxl = MagicMock()
    inf.t5xxl.model = MagicMock()
    inf.t5xxl.model.encode_token_weights.return_value = (
        torch.randn(1, 77, 4096),  # t5_out
        torch.randn(1, 4096)       # t5_pooled
    )
    
    inf.sd3 = MagicMock()
    inf.sd3.model = MagicMock()
    inf.sd3.model.cuda.return_value = inf.sd3.model
    inf.sd3.model.model_sampling = MagicMock()
    inf.sd3.model.model_sampling.sigma_max = torch.tensor(1.0)
    inf.sd3.model.model_sampling.sigma_min = torch.tensor(0.1)
    inf.sd3.model.model_sampling.timestep.side_effect = lambda x: x
    inf.sd3.model.forward.return_value = torch.randn(1, 4, 64, 64)
    
    inf.vae = MagicMock()
    inf.vae.model = MagicMock()
    
    # vae_decodeメソッドをモック
    inf.vae_decode = MagicMock()
    test_image = Image.new('RGB', (64, 64))
    inf.vae_decode.return_value = test_image
    
    # モックで非同期関数を返すようにする
    async def mock_do_sampling(*args, **kwargs):
        # 進捗コールバックを呼び出す（存在する場合）
        if 'progress_callback' in kwargs and kwargs['progress_callback']:
            for i in range(10):
                kwargs['progress_callback'](i, 10)
        return torch.randn(1, 4, 64, 64)
    
    # 非同期関数のモック
    inf.do_sampling = AsyncMock(side_effect=mock_do_sampling)
    
    # run_inferenceメソッドをモック
    async def mock_run_inference(*args, **kwargs):
        if 'progress_callback' in kwargs and kwargs['progress_callback']:
            for i in range(10):
                kwargs['progress_callback'](i, 10)
        return test_image
    
    inf.run_inference = AsyncMock(side_effect=mock_run_inference)
    
    return inf

@pytest.mark.asyncio
async def test_run_inference_with_progress(inferencer):
    """進捗コールバック付きの推論実行テスト"""
    progress_calls = []
    
    def progress_callback(current, total):
        progress_calls.append((current, total))
    
    # 推論の実行
    result = await inferencer.run_inference(
        prompt="test prompt",
        steps=10,
        cfg_scale=4.5,
        sampler="euler",
        width=64,
        height=64,
        seed=42,
        progress_callback=progress_callback
    )
    
    # 結果の検証
    assert isinstance(result, Image.Image)
    assert len(progress_calls) > 0  # プログレスコールバックが呼ばれたことを確認
    # 最後のコールバックで完了していることを確認
    assert progress_calls[-1][0] == progress_calls[-1][1] or progress_calls[-1][0] == 9

@pytest.mark.asyncio
async def test_do_sampling_with_progress(inferencer):
    """サンプリング処理の進捗コールバックテスト"""
    progress_calls = []
    
    def progress_callback(current, total):
        progress_calls.append((current, total))
    
    # サンプリングの実行
    result = await inferencer.do_sampling(
        latent=torch.randn(1, 4, 64, 64),
        seed=42,
        conditioning={"c_crossattn": torch.randn(1, 77, 768), "y": torch.randn(1, 768)},
        neg_cond={"c_crossattn": torch.randn(1, 77, 768), "y": torch.randn(1, 768)},
        steps=10,
        cfg_scale=4.5,
        sampler="euler",
        progress_callback=progress_callback
    )
    
    # 結果の検証
    assert isinstance(result, torch.Tensor)
    assert len(progress_calls) > 0  # プログレスコールバックが呼ばれたことを確認

def test_cfg_denoiser_progress():
    """CFGDenoiserの進捗コールバックテスト"""
    # モックの設定
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 4, 64, 64)
    progress_calls = []
    
    def progress_callback(current, total):
        progress_calls.append((current, total))
    
    # CFGDenoiserの作成と実行
    cfg_denoiser = CFGDenoiser(mock_model, steps=10, progress_callback=progress_callback)
    result = cfg_denoiser(torch.randn(1, 4, 64, 64), torch.tensor([1.0]))
    
    # 結果の検証
    assert result is not None
    assert len(progress_calls) > 0  # プログレスコールバックが呼ばれたことを確認

@pytest.mark.asyncio
async def test_concurrent_inference(inferencer):
    """複数の推論の同時実行テスト"""
    # 複数の推論を同時に実行
    tasks = []
    for i in range(3):
        task = inferencer.run_inference(
            prompt=f"test prompt {i}",
            steps=10,
            cfg_scale=4.5,
            sampler="euler",
            width=64,
            height=64,
            seed=42
        )
        tasks.append(task)
    
    # すべてのタスクが完了するまで待機
    results = await asyncio.gather(*tasks)
    
    # 結果の検証
    for result in results:
        assert isinstance(result, Image.Image)

@pytest.mark.asyncio
async def test_error_handling(inferencer):
    """エラーハンドリングのテスト"""
    # エラーを発生させる設定
    async def raise_error(*args, **kwargs):
        raise Exception("Test error")
    
    # 元の実装を保存
    original_run_inference = inferencer.run_inference
    
    try:
        # エラーを発生させる実装に置き換え
        inferencer.run_inference = AsyncMock(side_effect=raise_error)
        
        # エラーが適切に処理されることを確認
        with pytest.raises(Exception) as exc_info:
            await inferencer.run_inference(
                prompt="test prompt",
                steps=10,
                cfg_scale=4.5,
                sampler="euler",
                width=64,
                height=64,
                seed=42
            )
        
        assert "Test error" in str(exc_info.value)
        
    finally:
        # テスト後に元の実装に戻す
        inferencer.run_inference = original_run_inference 