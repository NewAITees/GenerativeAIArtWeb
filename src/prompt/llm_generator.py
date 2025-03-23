"""
LLMを使用したプロンプト生成機能

このモジュールでは、大規模言語モデル (LLM) を利用して、
簡単なプロンプト入力から詳細な画像生成プロンプトを生成・拡張する機能を提供します。
"""

import os
import logging
from pathlib import Path
import ollama
from typing import List, Dict, Any, Optional

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()


class LLMPromptGenerator:
    """LLMを使用したプロンプト生成クラス"""
    
    def __init__(self, model_name: str = "llama3", host: Optional[str] = None, timeout: int = 30):
        """初期化メソッド
        
        Args:
            model_name (str, optional): 使用するLLMモデル名。デフォルトは"llama3"。
            host (str, optional): ollamaサーバーのホスト。指定がない場合はデフォルト設定を使用。
            timeout (int, optional): API呼び出しのタイムアウト秒数。デフォルトは30秒。
        """
        self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self.initialized = False
        self.client = None
        
        # ollamaクライアントの初期化
        self._initialize_client()
        
        # システムプロンプトの設定
        self.system_prompt = """あなたはStable Diffusion 3.5モデルの画像生成に最適化された詳細なプロンプトを作成する専門家です。
        以下のガイドラインに従って、入力されたテキストからプロンプトを生成してください：

        1. プロンプトは視覚的に具体的かつ詳細であること
        2. 画像の被写体、スタイル、照明、カメラ設定などの要素を含めること
        3. 画質向上のための要素（解像度、詳細さ、リアリズムなど）を含めること
        4. 入力が短い場合でも、視覚的に魅力的なプロンプトに拡張すること
        5. 先頭に「a photo of」「an image of」などの表現を含めるとよい結果が得られます
        6. 芸術スタイルやテクニック（oil painting, watercolor, photorealistic, anime styleなど）を明示すると特定の見た目になります
        7. レンダリング品質に関する表現（8k resolution, highly detailed, professional photographyなど）を含めると高品質になります
        8. 特に指定がなければ、写真のような高品質な表現を目指すこと

        元の入力意図を維持しながら、これらの要素を自然に組み込んだプロンプトを生成してください。
        生成するテキストはプロンプトのみとし、説明や前置きは含めないでください。"""
    
    def _initialize_client(self):
        """ollamaクライアントを初期化する"""
        try:
            # ollamaクライアントの作成
            client_kwargs = {}
            if self.host:
                client_kwargs['host'] = self.host
            
            self.client = ollama.Client(**client_kwargs)
            
            # モデルの存在確認（初期化チェック）
            models = self.client.list()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if self.model_name in model_names:
                self.initialized = True
                logger.info(f"ollamaクライアントを初期化しました。モデル '{self.model_name}' が利用可能です。")
            else:
                logger.warning(f"モデル '{self.model_name}' が見つかりません。利用可能なモデル: {', '.join(model_names)}")
                self.initialized = False
        except Exception as e:
            logger.error(f"ollamaクライアント初期化エラー: {e}")
            self.initialized = False
    
    def generate_prompt(self, base_prompt: str, style: Optional[str] = None, max_tokens: int = 300) -> str:
        """基本プロンプトから詳細なプロンプトを生成する
        
        Args:
            base_prompt (str): 基本的なプロンプト（例: "猫"）
            style (str, optional): 特定のスタイル指定（例: "photorealistic", "anime"）
            max_tokens (int, optional): 生成する最大トークン数。デフォルトは300。
        
        Returns:
            str: 生成された詳細なプロンプト
        """
        if not base_prompt:
            return ""
        
        # モデルが初期化されていない場合はモックレスポンスを返す
        if not self.initialized or self.client is None:
            logger.warning("ollamaクライアントが初期化されていないため、モックレスポンスを返します")
            return self._mock_generate(base_prompt, style)
        
        try:
            # プロンプトの生成（スタイル指定がある場合は追加）
            prompt = base_prompt
            if style:
                prompt = f"{base_prompt} in {style} style"
            
            # ollamaのAPIを呼び出し
            response = self.client.generate(
                model=self.model_name,
                prompt=f"以下の入力からSD3.5用の画像生成プロンプトを作成してください: {prompt}",
                system=self.system_prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": max_tokens
                }
            )
            
            # レスポンスから生成されたテキストを抽出
            generated_text = response.get('response', '')
            
            # 生成されたテキストが空の場合はモックレスポンスを返す
            if not generated_text:
                logger.warning("空のレスポンスが返されました。モックレスポンスを返します。")
                return self._mock_generate(base_prompt, style)
            
            return generated_text.strip()
        
        except Exception as e:
            logger.error(f"プロンプト生成エラー: {e}")
            # エラー時はモックレスポンスを返す
            return self._mock_generate(base_prompt, style)
    
    def batch_enhance(self, prompts: List[str], style: Optional[str] = None) -> List[str]:
        """複数のプロンプトを一括で強化する
        
        Args:
            prompts (List[str]): 強化するプロンプトのリスト
            style (str, optional): 適用するスタイル
        
        Returns:
            List[str]: 強化されたプロンプトのリスト
        """
        return [self.generate_prompt(prompt, style) for prompt in prompts]
    
    def _mock_generate(self, base_prompt: str, style: Optional[str] = None) -> str:
        """モックのプロンプト生成（開発用）
        
        Args:
            base_prompt (str): 基本的なプロンプト
            style (str, optional): 適用するスタイル
        
        Returns:
            str: モックで生成されたプロンプト
        """
        # 基本的なプロンプトに追加要素を付加して返す
        style_text = f" in {style} style" if style else ""
        
        # 基本的なモック応答（プロンプトの種類に応じたテンプレート）
        prompt_map = {
            "猫": f"A photorealistic portrait of a cute cat with detailed fur{style_text}, sitting in a sunlit garden, surrounded by colorful flowers, 8k resolution, sharp focus, professional lighting",
            "犬": f"A high-quality photograph of a happy dog playing in a park{style_text}, professional camera, golden hour lighting, bokeh background, shallow depth of field, ultra-detailed",
            "風景": f"A breathtaking landscape{style_text} with majestic mountains, flowing river, lush green trees, dramatic sky with clouds, sunset lighting, 8k resolution, studio quality",
            "ポートレート": f"A professional portrait photograph of a person{style_text} with perfect lighting, studio setting, high detail skin texture, shallow depth of field, shot on Canon EOS R5",
            "都市": f"An aerial view of a futuristic city{style_text} with skyscrapers, neon lights, busy streets, dramatic sunset, cinematic composition, 8k resolution, hyperrealistic",
            "宇宙": f"A stunning view of space{style_text} with colorful nebulae, distant galaxies, stars twinkling, cosmic dust, captured by Hubble Space Telescope, ultra HD, astronomical photography"
        }
        
        # プロンプトマップにある場合はマップの内容を返す
        if base_prompt in prompt_map:
            return prompt_map[base_prompt]
        
        # 基本的な品質向上要素を追加
        quality_elements = ", 8k resolution, highly detailed, professional photography, perfect lighting, sharp focus"
        style_elements = f", {style} style" if style else ", dramatic composition, photorealistic, studio quality"
        
        return f"A detailed image of {base_prompt}{quality_elements}{style_elements}"
    
    def save_prompt(self, data: Dict[str, Any], file_path: str) -> bool:
        """プロンプトデータをJSONファイルとして保存する
        
        Args:
            data (Dict[str, Any]): 保存するデータ
            file_path (str): 保存先ファイルパス
        
        Returns:
            bool: 保存成功時はTrue、失敗時はFalse
        """
        try:
            import json
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"プロンプトデータを保存しました: {file_path}")
            return True
        except Exception as e:
            logger.error(f"プロンプトデータ保存エラー: {e}")
            return False
    
    def load_prompt(self, file_path: str) -> Dict[str, Any]:
        """プロンプトデータをJSONファイルから読み込む
        
        Args:
            file_path (str): 読み込むファイルパス
        
        Returns:
            Dict[str, Any]: 読み込んだデータ
        """
        try:
            import json
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"プロンプトデータを読み込みました: {file_path}")
            return data
        except Exception as e:
            logger.error(f"プロンプトデータ読み込みエラー: {e}")
            return {} 