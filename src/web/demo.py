import os
import sys
import argparse
from pathlib import Path
import logging
from PIL import Image
import numpy as np

# プロジェクトのルートディレクトリをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# GradioInterfaceクラスをインポート
from src.web.app import GradioInterface

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoGradioInterface(GradioInterface):
    """デモ用のGradioインターフェース"""
    
    def __init__(self):
        """初期化"""
        super().__init__()
        # デモモードを設定
        self.demo_mode = True
        
    def load_model(self, model_path):
        """デモ用のモデル読み込み関数（実際には読み込まない）"""
        logger.info(f"デモモード: モデル '{model_path}' の読み込みをシミュレート")
        self.model_loaded = True
        return f"デモモード: モデル '{os.path.basename(model_path)}' を読み込みました"
    
    def generate_image(self, prompt, model_path, steps, cfg_scale, sampler, width, height, seed):
        """デモ用の画像生成関数（実際には生成しない）"""
        try:
            logger.info(f"デモモード: 画像生成をシミュレート - プロンプト: '{prompt}'")
            
            # デモ用の画像を作成（カラフルなノイズ）
            # 色の選択はプロンプトの長さに基づいて行う（単純なデモ効果）
            color_index = len(prompt) % 3
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 赤、緑、青
            base_color = colors[color_index]
            
            # ノイズのある画像を作成
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            # ベースカラーで塗りつぶし
            img_array[:, :, 0] = base_color[0]
            img_array[:, :, 1] = base_color[1]
            img_array[:, :, 2] = base_color[2]
            
            # ノイズを追加
            noise = np.random.randint(0, 100, (height, width, 3), dtype=np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
            
            # 中央にテキストを追加するための枠を作成
            center_h, center_w = height // 2, width // 2
            text_height, text_width = height // 3, width // 2
            
            # 黒い枠を作成
            img_array[center_h - text_height // 2:center_h + text_height // 2,
                     center_w - text_width // 2:center_w + text_width // 2] = (0, 0, 0)
            
            # PILイメージに変換
            image = Image.fromarray(img_array)
            
            # 画像を保存
            save_path = self.outputs_dir / f"demo_output_{os.urandom(4).hex()}.png"
            image.save(str(save_path))
            
            return image, f"デモモード: 画像を生成しました: {prompt} (実際には生成されていません)"
        
        except Exception as e:
            logger.error(f"デモエラー: {e}")
            return None, f"Error: デモ実行中にエラーが発生しました: {e}"


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SD3.5 画像生成Webアプリケーション (デモモード)")
    parser.add_argument("--share", action="store_true", help="公開用リンクの生成（Gradio）")
    parser.add_argument("--debug", action="store_true", help="デバッグモードで実行")
    
    args = parser.parse_args()
    
    logger.info("デモモードでGradioアプリを起動します...")
    
    # デモ用Gradioインターフェースの作成と起動
    app = DemoGradioInterface()
    app.launch(share=args.share, debug=args.debug)


if __name__ == "__main__":
    main() 