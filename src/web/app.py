import os
import sys
import json
import argparse
import gradio as gr
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, cast
from src.models.web import GenerationRequest, GenerationResponse, UpscaleRequest, UIState
from src.models.settings import AppSettings, ImageGenerationSettings

# プロジェクトのルートディレクトリをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 実際の環境では、各モジュールをインポート
try:
    from src.generator.sd3_inf import SD3Inferencer
    from src.prompt.llm_generator import LLMPromptGenerator
    from src.prompt.json_builder import JSONPromptBuilder
    from src.utils.upscaler import Upscaler
    from src.utils.watermark import Watermarker
    from src.utils.file_manager import FileManager
except ImportError as e:
    logger.warning(f"モジュールのインポートエラー: {e}")
    # モックを作成して開発を続ける
    SD3Inferencer = None
    LLMPromptGenerator = None
    JSONPromptBuilder = None
    Upscaler = None
    Watermarker = None
    FileManager = None


class GradioInterface:
    """SD3.5モデルを利用した画像生成Webアプリケーション用のGradioインターフェース"""
    
    def __init__(self):
        """Gradioインターフェースの初期化"""
        self.model_loaded = False
        self.inferencer = None
        self.models_dir = project_root / "models"
        self.outputs_dir = project_root / "outputs"
        self.settings_dir = project_root / "settings"
        self.prompt_elements_path = project_root / "src" / "prompt" / "elements.json"
        
        # 各種ディレクトリの作成
        self.outputs_dir.mkdir(exist_ok=True)
        self.settings_dir.mkdir(exist_ok=True)
        
        # モデルパスの取得
        self.model_paths = self._get_model_paths()
        
        # ユーティリティクラスの初期化
        if FileManager:
            self.file_manager = FileManager(self.outputs_dir)
        
        # 進捗コールバック用の状態管理
        self.current_task = None
        self.task_progress = 0
        self.task_total = 0
    
    def _get_model_paths(self):
        """モデルディレクトリから利用可能なモデルファイル一覧を取得"""
        model_files = []
        
        if self.models_dir.exists():
            for file in self.models_dir.glob("*.safetensors"):
                if "sd3" in file.name.lower():
                    model_files.append(str(file))
                    
        return model_files or ["models/sd3.5_large.safetensors"]
    
    async def load_model(self, model_path):
        """モデルを非同期に読み込む"""
        try:
            if not os.path.exists(model_path):
                return f"Error: モデルファイル '{model_path}' が見つかりません"
            
            logger.info(f"モデルを読み込み中: {model_path}")
            
            # モデル読み込みを別スレッドで実行
            self.inferencer = await asyncio.to_thread(
                lambda: SD3Inferencer(model_path)
            )
            self.model_loaded = True
            return f"モデル '{os.path.basename(model_path)}' を読み込みました"
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return f"Error: モデル読み込み中にエラーが発生しました: {e}"
    
    def update_progress(self, current: int, total: int):
        """進捗状況を更新する"""
        self.task_progress = current
        self.task_total = total
        return current / total
    
    async def generate_image(
        self, 
        prompt: str, 
        model_path: str, 
        steps: int, 
        cfg_scale: float, 
        sampler: str, 
        width: int, 
        height: int, 
        seed: Optional[int],
        progress=gr.Progress()
    ) -> Tuple[Optional[Any], str]:
        """画像生成を非同期に実行する"""
        try:
            # リクエストをpydanticモデルに変換して検証
            request = GenerationRequest(
                prompt=prompt,
                model_path=model_path,
                settings=ImageGenerationSettings(
                    steps=steps,
                    cfg_scale=cfg_scale,
                    sampler=sampler,
                    width=width,
                    height=height,
                    seed=seed
                )
            )
            
            # モデルが読み込まれていない場合は読み込む
            if not self.model_loaded or self.inferencer is None:
                load_result = await self.load_model(request.model_path)
                if load_result.startswith("Error"):
                    return None, load_result
            
            logger.info(f"画像生成開始: prompt='{request.prompt}', steps={request.settings.steps}, cfg_scale={request.settings.cfg_scale}")
            
            # 画像生成実行（進捗表示付き）
            def generation_task():
                return self.inferencer.run_inference(
                    prompt=request.prompt,
                    steps=request.settings.steps,
                    cfg_scale=request.settings.cfg_scale,
                    sampler=request.settings.sampler,
                    width=request.settings.width,
                    height=request.settings.height,
                    seed=request.settings.seed,
                    progress_callback=self.update_progress
                )
            
            # 進捗表示を初期化
            self.task_progress = 0
            self.task_total = request.settings.steps
            
            # 画像生成を別スレッドで実行
            images = await asyncio.to_thread(generation_task)
            
            # 画像を保存
            save_path = self.outputs_dir / f"gradio_output_{os.urandom(4).hex()}.png"
            await asyncio.to_thread(self.save_image, images[0], str(save_path))
            
            # レスポンスを構築
            response = GenerationResponse(
                success=True,
                message=f"画像を生成しました: {request.prompt}",
                image_path=str(save_path),
                metadata={
                    "prompt": request.prompt,
                    "steps": request.settings.steps,
                    "cfg_scale": request.settings.cfg_scale,
                    "sampler": request.settings.sampler,
                    "width": request.settings.width,
                    "height": request.settings.height,
                    "seed": request.settings.seed
                }
            )
            
            return images[0], response.message
            
        except Exception as e:
            logger.error(f"画像生成エラー: {e}")
            response = GenerationResponse(
                success=False,
                message=f"Error: 画像生成中にエラーが発生しました",
                error=str(e)
            )
            return None, response.message

    def save_image(self, image, path):
        """画像を保存する"""
        try:
            # PILイメージの場合は直接保存
            if hasattr(image, 'save'):
                image.save(path)
            else:
                # PILイメージでない場合はログを出力して処理をスキップ
                logger.warning(f"非PILイメージを保存しようとしました: {type(image)}")
        except Exception as e:
            logger.error(f"画像保存エラー: {e}")
    
    def load_settings(self, preset_name):
        """設定プリセットを読み込む"""
        settings_path = self.settings_dir / f"{preset_name}.json"
        if not settings_path.exists():
            logger.warning(f"設定ファイルが見つかりません: {settings_path}")
            return {}
        
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                # テスト対応のため、settingsという階層がある場合は中身を返す
                return settings.get("settings", settings)
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            return {}
    
    def save_settings(self, preset_name, settings):
        """設定プリセットを保存する"""
        self.settings_dir.mkdir(exist_ok=True)
        settings_path = self.settings_dir / f"{preset_name}.json"
        
        try:
            # json.dumpではなくjson.dumpsでJSON文字列を作成してから書き込む
            json_str = json.dumps(settings, ensure_ascii=False, indent=2)
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return True
        except Exception as e:
            logger.error(f"設定保存エラー: {e}")
            return False
    
    def get_available_presets(self):
        """利用可能なプリセット一覧を取得"""
        if not self.settings_dir.exists():
            self.settings_dir.mkdir(exist_ok=True)
            return []
        
        preset_files = self.settings_dir.glob("*.json")
        return [p.stem for p in preset_files]
    
    def get_available_llm_models(self):
        """利用可能なLLMモデル一覧を取得する"""
        try:
            # ollamaクライアントを作成して利用可能なモデルを取得
            import ollama
            client = ollama.Client()
            models = client.list()
            model_names = [model['name'] for model in models.get('models', [])]
            return model_names if model_names else ["llama3"]
        except Exception as e:
            logger.error(f"LLMモデル一覧取得エラー: {e}")
            # エラー時はデフォルトモデルを返す
            return ["llama3"]

    def generate_prompt_llm(self, base_prompt, model_name="llama3", style="指定なし"):
        """LLMを使用してプロンプトを生成・拡張する"""
        if not LLMPromptGenerator:
            logger.warning("LLMPromptGenerator モジュールが利用できません")
            return f"拡張プロンプト: {base_prompt}"
        
        try:
            generator = LLMPromptGenerator(model_name=model_name)
            style_param = None if style == "指定なし" else style
            return generator.generate_prompt(base_prompt, style=style_param)
        except Exception as e:
            logger.error(f"プロンプト生成エラー: {e}")
            return base_prompt
    
    def generate_prompt_json(self, elements):
        """JSONベースのプロンプト構築機能を使用してプロンプトを生成する"""
        if not JSONPromptBuilder:
            logger.warning("JSONPromptBuilder モジュールが利用できません")
            return f"JSONプロンプト: {elements}"
        
        try:
            builder = JSONPromptBuilder()
            return builder.build_prompt(elements)
        except Exception as e:
            logger.error(f"JSONプロンプト構築エラー: {e}")
            return str(elements)
    
    def load_prompt_elements(self):
        """プロンプト要素のJSONファイルを読み込む"""
        if not self.prompt_elements_path.exists():
            logger.warning(f"プロンプト要素ファイルが見つかりません: {self.prompt_elements_path}")
            # デフォルト値を返す
            return {
                "subject": ["portrait", "landscape", "still life", "abstract"],
                "style": ["realistic", "fantasy", "anime", "oil painting", "watercolor"],
                "lighting": ["daylight", "sunset", "studio", "dramatic", "neon"],
                "camera": ["50mm", "portrait lens", "wide angle", "macro", "telephoto"]
            }
        
        try:
            with open(self.prompt_elements_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("elements", {})
        except Exception as e:
            logger.error(f"プロンプト要素読み込みエラー: {e}")
            return {}
    
    async def upscale_image(self, image, scale=2.0):
        """画像を非同期にアップスケールする"""
        if not Upscaler:
            logger.warning("Upscaler モジュールが利用できません")
            return image
        
        try:
            upscaler = Upscaler()
            return await asyncio.to_thread(upscaler.upscale, image, scale=scale)
        except Exception as e:
            logger.error(f"アップスケールエラー: {e}")
            return image
    
    async def add_watermark(self, image, text="Generated with SD3.5", position="bottom-right", opacity=0.5):
        """画像に非同期にウォーターマークを追加する"""
        if not Watermarker:
            logger.warning("Watermarker モジュールが利用できません")
            return image
        
        try:
            processor = Watermarker()
            return await asyncio.to_thread(
                processor.add_watermark,
                image,
                text=text,
                position=position,
                opacity=opacity
            )
        except Exception as e:
            logger.error(f"ウォーターマーク追加エラー: {e}")
            return image
    
    async def save_image_custom(self, image, directory=None, filename_pattern=None, metadata=None):
        """カスタムファイル保存を非同期に実行する"""
        if not FileManager:
            logger.warning("FileManager モジュールが利用できません")
            save_path = self.outputs_dir / f"custom_{os.urandom(4).hex()}.png"
            await asyncio.to_thread(self.save_image, image, str(save_path))
            return str(save_path)
        
        try:
            if metadata is None:
                metadata = {}
            
            # ファイルマネージャーのインスタンスを使用して画像を保存
            result = await asyncio.to_thread(
                self.file_manager.save_image,
                image,
                directory=directory,
                filename_pattern=filename_pattern,
                metadata=metadata
            )
            return result
        except Exception as e:
            logger.error(f"カスタムファイル保存エラー: {e}")
            save_path = self.outputs_dir / f"error_{os.urandom(4).hex()}.png"
            await asyncio.to_thread(self.save_image, image, str(save_path))
            return str(save_path)
    
    def get_save_directories(self):
        """保存ディレクトリ一覧を取得する"""
        if not FileManager:
            logger.warning("FileManager モジュールが利用できません")
            return ["outputs"]
        
        try:
            # ファイルマネージャーのインスタンスを使用してディレクトリ一覧を取得
            dirs = self.file_manager.get_directories()
            # テスト対応のため、実際の戻り値を返す
            return dirs
        except Exception as e:
            logger.error(f"ディレクトリ一覧取得エラー: {e}")
            return ["outputs"]
    
    def create_interface(self):
        """Gradioインターフェースを作成する"""
        with gr.Blocks(title="SD3.5 Image Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# SD3.5 Image Generator")
            
            with gr.Tabs():
                # メイン画像生成タブ
                with gr.TabItem("画像生成"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # 入力コントロール
                            prompt = gr.Textbox(
                                label="プロンプト",
                                placeholder="画像の説明を入力してください...",
                                lines=3
                            )
                            
                            with gr.Row():
                                model_path = gr.Dropdown(
                                    label="モデル",
                                    choices=self.model_paths,
                                    value=self.model_paths[0] if self.model_paths else None
                                )
                                
                                load_model_btn = gr.Button("モデル読み込み")
                            
                            with gr.Row():
                                with gr.Column():
                                    steps = gr.Slider(
                                        label="生成ステップ数",
                                        minimum=10,
                                        maximum=100,
                                        value=40,
                                        step=1
                                    )
                                    cfg_scale = gr.Slider(
                                        label="CFGスケール",
                                        minimum=1.0,
                                        maximum=10.0,
                                        value=4.5,
                                        step=0.1
                                    )
                                    sampler = gr.Dropdown(
                                        label="サンプラー",
                                        choices=["euler", "dpmpp_2m"],
                                        value="euler"
                                    )
                                
                                with gr.Column():
                                    width = gr.Slider(
                                        label="幅",
                                        minimum=256,
                                        maximum=1536,
                                        value=1024,
                                        step=64
                                    )
                                    height = gr.Slider(
                                        label="高さ",
                                        minimum=256,
                                        maximum=1536,
                                        value=1024,
                                        step=64
                                    )
                                    seed = gr.Number(
                                        label="シード（空白はランダム）",
                                        value=None
                                    )
                            
                            # 設定保存/読み込み
                            with gr.Accordion("設定保存/読込", open=False):
                                with gr.Row():
                                    presets = gr.Dropdown(
                                        label="プリセット",
                                        choices=self.get_available_presets(),
                                        value=None
                                    )
                                    preset_name = gr.Textbox(
                                        label="新規プリセット名",
                                        placeholder="保存するプリセット名を入力"
                                    )
                                
                                with gr.Row():
                                    load_preset_btn = gr.Button("読み込み")
                                    save_preset_btn = gr.Button("保存")
                                    refresh_presets_btn = gr.Button("更新")
                            
                            # 生成ボタン
                            generate_btn = gr.Button("画像生成", variant="primary")
                        
                        with gr.Column(scale=3):
                            # 出力表示
                            image_output = gr.Image(label="生成画像")
                            status_output = gr.Textbox(label="ステータス", interactive=False)
                            
                            # 画像処理オプション
                            with gr.Accordion("画像処理オプション", open=False):
                                with gr.Row():
                                    upscale_factor = gr.Dropdown(
                                        label="アップスケール倍率",
                                        choices=["1.0", "1.5", "2.0", "4.0"],
                                        value="1.0"
                                    )
                                    upscale_btn = gr.Button("アップスケール")
                                
                                with gr.Row():
                                    watermark_text = gr.Textbox(
                                        label="ウォーターマークテキスト",
                                        value="Generated with SD3.5"
                                    )
                                    watermark_position = gr.Dropdown(
                                        label="位置",
                                        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
                                        value="bottom-right"
                                    )
                                    watermark_opacity = gr.Slider(
                                        label="不透明度",
                                        minimum=0.1,
                                        maximum=1.0,
                                        value=0.5,
                                        step=0.1
                                    )
                                    add_watermark_btn = gr.Button("ウォーターマーク追加")
                                
                                with gr.Row():
                                    save_directory = gr.Dropdown(
                                        label="保存ディレクトリ",
                                        choices=self.get_save_directories(),
                                        value="outputs"
                                    )
                                    filename_pattern = gr.Textbox(
                                        label="ファイル名パターン",
                                        value="{prompt}_{date}"
                                    )
                                    custom_save_btn = gr.Button("カスタム保存")
                
                # プロンプト生成タブ
                with gr.TabItem("プロンプト生成"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### LLMプロンプト生成")
                            
                            # モデル選択を追加
                            llm_model = gr.Dropdown(
                                label="LLMモデル",
                                choices=self.get_available_llm_models(),
                                value="llama3"
                            )
                            
                            base_prompt = gr.Textbox(
                                label="基本プロンプト",
                                placeholder="基本的なアイデアを入力してください",
                                lines=2
                            )
                            
                            # スタイル選択を追加
                            prompt_style = gr.Dropdown(
                                label="スタイル",
                                choices=["指定なし", "photorealistic", "anime", "oil painting", "watercolor", "digital art"],
                                value="指定なし"
                            )
                            
                            generate_llm_btn = gr.Button("LLMでプロンプト生成")
                            llm_output = gr.Textbox(
                                label="生成されたプロンプト",
                                lines=4,
                                interactive=False
                            )
                            use_llm_prompt_btn = gr.Button("このプロンプトを使用")

                        with gr.Column():
                            gr.Markdown("### JSONプロンプト生成")
                            
                            # JSONプロンプト要素を動的に読み込み
                            elements = self.load_prompt_elements()
                            
                            json_subject = gr.Dropdown(
                                label="被写体",
                                choices=elements.get("subject", ["portrait", "landscape"]),
                                value=elements.get("subject", ["portrait"])[0] if elements.get("subject") else None
                            )
                            
                            json_style = gr.Dropdown(
                                label="スタイル",
                                choices=elements.get("style", ["realistic", "fantasy"]),
                                value=elements.get("style", ["realistic"])[0] if elements.get("style") else None
                            )
                            
                            json_elements = gr.CheckboxGroup(
                                label="追加要素",
                                choices=elements.get("elements", ["mountains", "forest", "river", "sunset"]),
                                value=[]
                            )
                            
                            json_lighting = gr.Dropdown(
                                label="ライティング",
                                choices=elements.get("lighting", ["daylight", "sunset", "studio"]),
                                value=elements.get("lighting", ["daylight"])[0] if elements.get("lighting") else None
                            )
                            
                            generate_json_btn = gr.Button("JSONからプロンプト構築")
                            json_output = gr.Textbox(
                                label="構築されたプロンプト",
                                lines=4,
                                interactive=False
                            )
                            use_json_prompt_btn = gr.Button("このプロンプトを使用")
                
                # 設定タブ
                with gr.TabItem("設定"):
                    gr.Markdown("### アプリケーション設定")
                    with gr.Row():
                        refresh_models_btn = gr.Button("モデル一覧更新")
                        refresh_dirs_btn = gr.Button("ディレクトリ一覧更新")
            
            # イベントハンドラの設定（非同期対応）
            load_model_btn.click(
                fn=self.load_model,
                inputs=[model_path],
                outputs=[status_output]
            )
            
            generate_btn.click(
                fn=self.generate_image,
                inputs=[prompt, model_path, steps, cfg_scale, sampler, width, height, seed],
                outputs=[image_output, status_output],
                show_progress="full"
            )
            
            # 設定保存/読み込み
            save_preset_btn.click(
                fn=lambda name, p, s, c, sm, w, h, sd: (
                    self.save_settings(name, {
                        "prompt": p,
                        "steps": s,
                        "cfg_scale": c,
                        "sampler": sm,
                        "width": w,
                        "height": h,
                        "seed": sd
                    }),
                    f"設定 '{name}' を保存しました"
                ),
                inputs=[preset_name, prompt, steps, cfg_scale, sampler, width, height, seed],
                outputs=[status_output]
            )
            
            def load_preset_fn(preset_name):
                settings = self.load_settings(preset_name)
                if not settings:
                    return [None] * 7 + [f"Error: プリセット '{preset_name}' を読み込めませんでした"]
                
                return [
                    settings.get("prompt", ""),
                    settings.get("steps", 40),
                    settings.get("cfg_scale", 4.5),
                    settings.get("sampler", "euler"),
                    settings.get("width", 1024),
                    settings.get("height", 1024),
                    settings.get("seed", None),
                    f"設定 '{preset_name}' を読み込みました"
                ]
            
            load_preset_btn.click(
                fn=load_preset_fn,
                inputs=[presets],
                outputs=[prompt, steps, cfg_scale, sampler, width, height, seed, status_output]
            )
            
            refresh_presets_btn.click(
                fn=lambda: gr.Dropdown.update(choices=self.get_available_presets()),
                inputs=[],
                outputs=[presets]
            )
            
            # LLMプロンプト生成のイベントハンドラ
            generate_llm_btn.click(
                fn=lambda prompt, model, style: self.generate_prompt_llm(prompt, model, style),
                inputs=[base_prompt, llm_model, prompt_style],
                outputs=[llm_output]
            )
            
            # 生成されたプロンプトを使用するイベントハンドラ
            use_llm_prompt_btn.click(
                fn=lambda x: x,
                inputs=[llm_output],
                outputs=[prompt]
            )
            
            generate_json_btn.click(
                fn=lambda subject, style, elements, lighting: self.generate_prompt_json({
                    "subject": subject,
                    "style": style,
                    "elements": elements,
                    "lighting": lighting
                }),
                inputs=[json_subject, json_style, json_elements, json_lighting],
                outputs=[json_output]
            )
            
            use_json_prompt_btn.click(
                fn=lambda x: x,
                inputs=[json_output],
                outputs=[prompt]
            )
            
            # 画像処理（非同期対応）
            upscale_btn.click(
                fn=self.upscale_image,
                inputs=[image_output, upscale_factor],
                outputs=[image_output, status_output],
                show_progress="full"
            )
            
            add_watermark_btn.click(
                fn=self.add_watermark,
                inputs=[image_output, watermark_text, watermark_position, watermark_opacity],
                outputs=[image_output, status_output],
                show_progress="full"
            )
            
            custom_save_btn.click(
                fn=self.save_image_custom,
                inputs=[image_output, save_directory, filename_pattern, prompt],
                outputs=[status_output],
                show_progress="full"
            )
            
            # 設定更新
            refresh_models_btn.click(
                fn=lambda: (
                    self._get_model_paths(),
                    gr.Dropdown.update(choices=self._get_model_paths()),
                    "モデル一覧を更新しました"
                ),
                inputs=[],
                outputs=[model_path, status_output]
            )
            
            refresh_dirs_btn.click(
                fn=lambda: (
                    gr.Dropdown.update(choices=self.get_save_directories()),
                    "ディレクトリ一覧を更新しました"
                ),
                inputs=[],
                outputs=[save_directory, status_output]
            )
        
        return interface
    
    def launch(self, share=False, debug=False):
        """Gradioアプリを起動する"""
        interface = self.create_interface()
        interface.launch(share=share, debug=debug)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SD3.5 画像生成ウェブアプリケーション")
    parser.add_argument("--share", action="store_true", help="公開用リンクの生成（Gradio）")
    parser.add_argument("--debug", action="store_true", help="デバッグモードで実行")
    
    args = parser.parse_args()
    
    # Gradioインターフェースの作成と起動
    app = GradioInterface()
    app.launch(share=args.share, debug=args.debug)


if __name__ == "__main__":
    main() 