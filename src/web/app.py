import os
import sys
import argparse
import gradio as gr
import logging
from pathlib import Path

# プロジェクトのルートディレクトリをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 実際の環境では、ジェネレーターモジュールをインポート
try:
    from src.generator.sd3_inf import SD3Inferencer
except ImportError:
    logger.warning("SD3Inferencerをインポートできませんでした。モックが必要です。")
    SD3Inferencer = None


class GradioInterface:
    """SD3.5モデルを利用した画像生成Webアプリケーション用のGradioインターフェース"""
    
    def __init__(self):
        """Gradioインターフェースの初期化"""
        self.model_loaded = False
        self.inferencer = None
        self.models_dir = project_root / "models"
        self.outputs_dir = project_root / "outputs"
        
        # 出力ディレクトリの作成
        self.outputs_dir.mkdir(exist_ok=True)
        
        # モデルパスの取得
        self.model_paths = self._get_model_paths()
        
    def _get_model_paths(self):
        """モデルディレクトリから利用可能なモデルファイル一覧を取得"""
        model_files = []
        
        if self.models_dir.exists():
            for file in self.models_dir.glob("*.safetensors"):
                if "sd3" in file.name.lower():
                    model_files.append(str(file))
                    
        return model_files or ["models/sd3.5_large.safetensors"]
    
    def load_model(self, model_path):
        """モデルを読み込む"""
        try:
            if not os.path.exists(model_path):
                return f"Error: モデルファイル '{model_path}' が見つかりません"
            
            logger.info(f"モデルを読み込み中: {model_path}")
            self.inferencer = SD3Inferencer(model_path)
            self.model_loaded = True
            return f"モデル '{os.path.basename(model_path)}' を読み込みました"
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return f"Error: モデル読み込み中にエラーが発生しました: {e}"
    
    def generate_image(self, prompt, model_path, steps, cfg_scale, sampler, width, height, seed):
        """画像生成を実行する"""
        try:
            # モデルが読み込まれていない場合は読み込む
            if not self.model_loaded or self.inferencer is None:
                load_result = self.load_model(model_path)
                if load_result.startswith("Error"):
                    return None, load_result
            
            # パラメータの型変換
            steps = int(steps)
            cfg_scale = float(cfg_scale)
            width = int(width)
            height = int(height)
            seed = int(seed) if seed else None
            
            logger.info(f"画像生成開始: prompt='{prompt}', steps={steps}, cfg_scale={cfg_scale}")
            
            # 画像生成実行
            images = self.inferencer.run_inference(
                prompt=prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                width=width,
                height=height,
                seed=seed
            )
            
            # 画像を保存
            save_path = self.outputs_dir / f"gradio_output_{os.urandom(4).hex()}.png"
            self.save_image(images[0], str(save_path))
            
            return images[0], f"画像を生成しました: {prompt}"
            
        except Exception as e:
            logger.error(f"画像生成エラー: {e}")
            return None, f"Error: 画像生成中にエラーが発生しました: {e}"

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
    
    def create_interface(self):
        """Gradioインターフェースを作成する"""
        with gr.Blocks(title="SD3.5画像生成") as interface:
            gr.Markdown("# SD3.5 画像生成ウェブアプリケーション")
            gr.Markdown("Stable Diffusion 3.5モデルを使用して、テキストプロンプトから高品質な画像を生成します。")
            
            with gr.Tab("画像生成"):
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
                        
                        # 生成ボタン
                        generate_btn = gr.Button("画像生成", variant="primary")
                    
                    with gr.Column(scale=3):
                        # 出力表示
                        image_output = gr.Image(label="生成画像")
                        status_output = gr.Textbox(label="ステータス", interactive=False)
            
            # イベントハンドラの設定
            load_model_btn.click(
                fn=self.load_model,
                inputs=[model_path],
                outputs=[status_output]
            )
            
            generate_btn.click(
                fn=self.generate_image,
                inputs=[prompt, model_path, steps, cfg_scale, sampler, width, height, seed],
                outputs=[image_output, status_output]
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