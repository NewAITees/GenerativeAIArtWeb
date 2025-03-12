import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# プロジェクトのルートディレクトリをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# SD3Inferencerをモック化
sys.modules['src.generator.sd3_inf'] = MagicMock()
from src.generator.sd3_inf import SD3Inferencer

# インポート前に必要なモジュールをモック化
from src.web.app import GradioInterface


class TestGradioInterface(unittest.TestCase):
    """GradioインターフェースクラスのテストクラS"""
    
    def setUp(self):
        """テスト前の準備"""
        # save_imageメソッドをモック化
        self.patcher = patch.object(GradioInterface, 'save_image')
        self.mock_save_image = self.patcher.start()
        
        # GradioInterfaceのインスタンス作成
        self.app = GradioInterface()
        
        # テスト用に別のモック設定があるメソッドは本物のインスタンスの状態を保存
        self.original_inferencer = self.app.inferencer
        self.original_model_loaded = self.app.model_loaded
        
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.patcher.stop()
    
    def test_initialization(self):
        """初期化が正しく行われるかテスト"""
        # 新しいインスタンスを作成してテスト
        app = GradioInterface()
        self.assertFalse(app.model_loaded)
        self.assertIsNone(app.inferencer)
        self.assertEqual(app.models_dir, project_root / "models")
        self.assertEqual(app.outputs_dir, project_root / "outputs")
    
    @patch('os.path.exists')
    def test_load_model_success(self, mock_exists):
        """モデル読み込みが成功するケースのテスト"""
        # osパスチェックをモック化して常にTrueを返す
        mock_exists.return_value = True
        
        # テスト用に空のSD3Inferencerモックを設定
        with patch('src.web.app.SD3Inferencer') as mock_inferencer:
            # モックインスタンスの設定
            mock_inferencer_instance = MagicMock()
            mock_inferencer.return_value = mock_inferencer_instance
            
            # 元の状態に戻す
            self.app.model_loaded = False
            self.app.inferencer = None
            
            model_path = "models/test_model.safetensors"
            result = self.app.load_model(model_path)
            
            # 結果の検証
            self.assertIn("test_model.safetensors", result)
            self.assertTrue(self.app.model_loaded)
            mock_inferencer.assert_called_once_with(model_path)
    
    @patch('os.path.exists')
    def test_load_model_failure_not_exists(self, mock_exists):
        """モデルファイルが存在しない場合のテスト"""
        # osパスチェックをモック化してFalseを返す
        mock_exists.return_value = False
        
        # 元の状態に戻す
        self.app.model_loaded = False
        self.app.inferencer = None
        
        model_path = "models/nonexistent_model.safetensors"
        result = self.app.load_model(model_path)
        
        # 結果の検証
        self.assertIn("Error", result)
        self.assertFalse(self.app.model_loaded)
    
    def test_generate_image_success(self):
        """画像生成が成功するケースのテスト"""
        # インスタンス属性にモックを設定
        self.app.inferencer = MagicMock()
        
        # モックの設定
        mock_image = MagicMock()
        self.app.inferencer.run_inference.return_value = [mock_image]
        
        # 事前にモデルを読み込み済みとする
        self.app.model_loaded = True
        
        # 画像生成の実行
        image, status = self.app.generate_image(
            prompt="test prompt",
            model_path="models/test_model.safetensors",
            steps=40,
            cfg_scale=4.5,
            sampler="euler",
            width=1024,
            height=1024,
            seed=None
        )
        
        # 結果の検証
        self.assertEqual(image, mock_image)
        self.assertIn("画像を生成しました", status)
        self.app.inferencer.run_inference.assert_called_once()
        self.mock_save_image.assert_called_once()
    
    def test_save_image(self):
        """画像保存機能のテスト"""
        # モック画像の作成（本物のPILImageをモック）
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        
        # パッチを停止して本物のメソッドをテスト
        self.patcher.stop()
        
        # 保存を実行
        self.app.save_image(mock_image, "test_path.png")
        
        # 検証
        mock_image.save.assert_called_once_with("test_path.png")
        
        # パッチを再開
        self.patcher = patch.object(GradioInterface, 'save_image')
        self.mock_save_image = self.patcher.start()
    
    def test_create_interface(self):
        """インターフェースが正しく作成されるかテスト"""
        interface = self.app.create_interface()
        self.assertIsNotNone(interface)


if __name__ == "__main__":
    unittest.main() 