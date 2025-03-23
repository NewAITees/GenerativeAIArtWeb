import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# プロジェクトのルートディレクトリをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# SD3Inferencerをモック化
sys.modules['src.generator.sd3_inf'] = MagicMock()
from src.generator.sd3_inf import SD3Inferencer

# LLMジェネレーターをモック化
sys.modules['src.prompt.llm_generator'] = MagicMock()
sys.modules['src.prompt.json_builder'] = MagicMock()

# 画像処理モジュールをモック化
sys.modules['src.utils.upscaler'] = MagicMock()
sys.modules['src.utils.watermark'] = MagicMock()
sys.modules['src.utils.file_manager'] = MagicMock()

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
        # 設定ディレクトリが初期化されているか確認
        self.assertEqual(app.settings_dir, project_root / "settings")
    
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

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"settings": {"prompt": "test", "steps": 40}}')
    def test_load_settings(self, mock_file, mock_exists):
        """設定読み込み機能のテスト"""
        # ファイルが存在するとして設定
        mock_exists.return_value = True
        
        # テスト対象のメソッド呼び出し
        settings = self.app.load_settings("test_preset")
        
        # 検証
        self.assertEqual(settings["prompt"], "test")
        self.assertEqual(settings["steps"], 40)
        mock_file.assert_called_once()

    @patch('pathlib.Path.exists')
    def test_load_settings_not_exist(self, mock_exists):
        """存在しない設定ファイルを読み込む場合のテスト"""
        # ファイルが存在しないとして設定
        mock_exists.return_value = False
        
        # テスト対象のメソッド呼び出し
        settings = self.app.load_settings("nonexistent_preset")
        
        # 検証
        self.assertEqual(settings, {})

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_save_settings(self, mock_mkdir, mock_file):
        """設定保存機能のテスト"""
        # テスト用設定データ
        test_settings = {
            "prompt": "test prompt",
            "steps": 40,
            "cfg_scale": 4.5,
            "sampler": "euler",
            "width": 1024,
            "height": 1024
        }
        
        # テスト対象のメソッド呼び出し
        result = self.app.save_settings("test_preset", test_settings)
        
        # 検証
        self.assertTrue(result)
        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_file.assert_called_once()
        # 書き込まれたJSONを検証
        mock_file().write.assert_called_once()
        written_data = mock_file().write.call_args[0][0]
        written_settings = json.loads(written_data)
        self.assertEqual(written_settings["prompt"], "test prompt")
        self.assertEqual(written_settings["steps"], 40)

    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.exists')
    def test_get_available_presets(self, mock_exists, mock_glob):
        """利用可能なプリセット一覧取得のテスト"""
        # ディレクトリが存在する設定
        mock_exists.return_value = True
        
        # 戻り値の設定
        preset_paths = [
            Path('preset1.json'),
            Path('preset2.json'),
            Path('preset3.json')
        ]
        mock_glob.return_value = preset_paths
        
        # テスト対象のメソッド呼び出し
        presets = self.app.get_available_presets()
        
        # 検証
        self.assertEqual(len(presets), 3)
        self.assertIn('preset1', presets)
        self.assertIn('preset2', presets)
        self.assertIn('preset3', presets)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_get_available_presets_empty(self, mock_mkdir, mock_exists):
        """プリセットディレクトリが存在しない場合のテスト"""
        # ディレクトリが存在しない設定
        mock_exists.return_value = False
        
        # テスト対象のメソッド呼び出し
        presets = self.app.get_available_presets()
        
        # 検証
        self.assertEqual(len(presets), 0)
        mock_mkdir.assert_called_once_with(exist_ok=True)

    @patch('src.web.app.LLMPromptGenerator')
    def test_generate_prompt_llm(self, mock_llm_generator):
        """LLMを使ったプロンプト生成機能のテスト"""
        # モックの設定
        mock_generator_instance = MagicMock()
        mock_llm_generator.return_value = mock_generator_instance
        mock_generator_instance.generate_prompt.return_value = "A beautiful sunset with detailed clouds"
        
        # テスト対象のメソッド呼び出し
        result = self.app.generate_prompt_llm("sunset")
        
        # 検証
        self.assertEqual(result, "A beautiful sunset with detailed clouds")
        mock_generator_instance.generate_prompt.assert_called_once_with("sunset")

    @patch('src.web.app.JSONPromptBuilder')
    def test_generate_prompt_json(self, mock_json_builder):
        """JSONベースのプロンプト構築機能のテスト"""
        # モックの設定
        mock_builder_instance = MagicMock()
        mock_json_builder.return_value = mock_builder_instance
        mock_builder_instance.build_prompt.return_value = "A fantasy landscape with mountains, river, sunset"
        
        # テスト用データ
        test_elements = {
            "subject": "landscape",
            "style": "fantasy",
            "elements": ["mountains", "river"],
            "lighting": "sunset"
        }
        
        # テスト対象のメソッド呼び出し
        result = self.app.generate_prompt_json(test_elements)
        
        # 検証
        self.assertEqual(result, "A fantasy landscape with mountains, river, sunset")
        mock_builder_instance.build_prompt.assert_called_once_with(test_elements)

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"elements": {"subject": ["portrait", "landscape"], "style": ["fantasy", "realistic"]}}')
    def test_load_prompt_elements(self, mock_file, mock_exists):
        """プロンプト要素ファイルの読み込みテスト"""
        # ファイルが存在するとして設定
        mock_exists.return_value = True
        
        # テスト対象のメソッド呼び出し
        elements = self.app.load_prompt_elements()
        
        # 検証
        self.assertIn("subject", elements)
        self.assertIn("style", elements)
        self.assertEqual(len(elements["subject"]), 2)
        self.assertEqual(len(elements["style"]), 2)
        self.assertIn("portrait", elements["subject"])
        self.assertIn("landscape", elements["subject"])
        mock_file.assert_called_once()

    @patch('pathlib.Path.exists')
    def test_load_prompt_elements_not_exist(self, mock_exists):
        """存在しないプロンプト要素ファイルの読み込みテスト"""
        # ファイルが存在しないとして設定
        mock_exists.return_value = False
        
        # テスト対象のメソッド呼び出し
        elements = self.app.load_prompt_elements()
        
        # 検証（デフォルト値が返されるか）
        self.assertIsNotNone(elements)
        self.assertIn("subject", elements)
        self.assertIn("style", elements)

    @patch('src.web.app.ImageUpscaler')
    def test_upscale_image(self, mock_upscaler):
        """画像アップスケール機能のテスト"""
        # モックの設定
        mock_upscaler_instance = MagicMock()
        mock_upscaler.return_value = mock_upscaler_instance
        
        # 入力と出力の画像をモック
        mock_input_image = MagicMock()
        mock_output_image = MagicMock()
        mock_upscaler_instance.upscale.return_value = mock_output_image
        
        # テスト対象のメソッド呼び出し
        result = self.app.upscale_image(mock_input_image, scale=2.0)
        
        # 検証
        self.assertEqual(result, mock_output_image)
        mock_upscaler_instance.upscale.assert_called_once_with(mock_input_image, scale=2.0)

    @patch('src.web.app.WatermarkProcessor')
    def test_add_watermark(self, mock_watermark):
        """ウォーターマーク追加機能のテスト"""
        # モックの設定
        mock_watermark_instance = MagicMock()
        mock_watermark.return_value = mock_watermark_instance
        
        # 入力と出力の画像をモック
        mock_input_image = MagicMock()
        mock_output_image = MagicMock()
        mock_watermark_instance.add_watermark.return_value = mock_output_image
        
        # テスト対象のメソッド呼び出し
        result = self.app.add_watermark(
            mock_input_image,
            text="Test Watermark",
            position="bottom-right",
            opacity=0.5
        )
        
        # 検証
        self.assertEqual(result, mock_output_image)
        mock_watermark_instance.add_watermark.assert_called_once_with(
            mock_input_image,
            text="Test Watermark",
            position="bottom-right",
            opacity=0.5
        )

    @patch('src.web.app.FileManager')
    def test_save_image_custom(self, mock_file_manager):
        """カスタムファイル保存機能のテスト"""
        # モックの設定
        mock_manager_instance = MagicMock()
        mock_file_manager.return_value = mock_manager_instance
        mock_manager_instance.save_image.return_value = "custom/path/image.png"
        
        # モックを適用したGradioInterfaceの新しいインスタンスを作成
        app = GradioInterface()
        # file_managerプロパティを直接オーバーライド
        app.file_manager = mock_manager_instance
        
        # 入力画像をモック
        mock_image = MagicMock()
        
        # テスト対象のメソッド呼び出し
        result = app.save_image_custom(
            mock_image,
            directory="custom/path",
            filename_pattern="{prompt}_{date}",
            metadata={"prompt": "test prompt"}
        )
        
        # 検証
        self.assertEqual(result, "custom/path/image.png")
        mock_manager_instance.save_image.assert_called_once_with(
            mock_image,
            directory="custom/path",
            filename_pattern="{prompt}_{date}",
            metadata={"prompt": "test prompt"}
        )

    @patch('src.web.app.FileManager')
    def test_get_save_directories(self, mock_file_manager):
        """保存ディレクトリ一覧取得機能のテスト"""
        # モックの設定
        mock_manager_instance = MagicMock()
        mock_file_manager.return_value = mock_manager_instance
        mock_manager_instance.get_directories.return_value = ["dir1", "dir2", "dir3"]
        
        # モックを適用したGradioInterfaceの新しいインスタンスを作成
        app = GradioInterface()
        # file_managerプロパティを直接オーバーライド
        app.file_manager = mock_manager_instance
        
        # テスト対象のメソッド呼び出し
        result = app.get_save_directories()
        
        # 検証
        self.assertEqual(len(result), 3)
        self.assertIn("dir1", result)
        self.assertIn("dir2", result)
        self.assertIn("dir3", result)
        mock_manager_instance.get_directories.assert_called_once()


if __name__ == "__main__":
    unittest.main() 