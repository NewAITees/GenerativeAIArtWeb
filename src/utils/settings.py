"""
設定管理機能

このモジュールでは、アプリケーション設定の保存、読み込み、管理機能を提供します。
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Union, List

from src.models.settings import AppSettings, ImageGenerationSettings

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()


class SettingsManager:
    """設定管理クラス"""
    
    def __init__(self, settings_dir=None):
        """初期化メソッド
        
        Args:
            settings_dir (str or Path, optional): 設定ファイルの保存ディレクトリ。
                指定がない場合はプロジェクトルートのsettingsディレクトリを使用。
        """
        self.settings_dir = Path(settings_dir) if settings_dir else project_root / "settings"
        
        # 設定ディレクトリが存在しない場合は作成
        self.settings_dir.mkdir(exist_ok=True)
    
    def save_profile(self, profile_name: str, settings: Dict[str, Any]) -> bool:
        """設定プロファイルを保存する
        
        Args:
            profile_name (str): プロファイル名
            settings (dict): 保存する設定
        
        Returns:
            bool: 保存成功時はTrue、失敗時はFalse
        """
        try:
            # 設定をpydanticモデルに変換して検証
            app_settings = AppSettings.model_validate(settings)
            
            # 設定ファイルのパスを作成
            profile_path = self.settings_dir / f"{profile_name}.json"
            
            # 設定を保存
            with open(str(profile_path), "w", encoding="utf-8") as f:
                json.dump(app_settings.model_dump(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"設定プロファイルを保存しました: {profile_path}")
            return True
        except Exception as e:
            logger.error(f"設定プロファイル保存エラー: {e}")
            return False
    
    def load_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """設定プロファイルを読み込む
        
        Args:
            profile_name (str): プロファイル名
        
        Returns:
            dict: 設定辞書。ファイルが存在しない場合はNone。
        """
        # プロファイルファイルのパスを作成
        profile_path = self.settings_dir / f"{profile_name}.json"
        
        # プロファイルファイルが存在するか確認
        if not profile_path.exists():
            logger.warning(f"設定プロファイルが見つかりません: {profile_path}")
            return None
        
        try:
            # プロファイルの読み込み
            with open(str(profile_path), "r", encoding="utf-8") as f:
                settings_dict = json.load(f)
            
            # pydanticモデルで検証
            app_settings = AppSettings.model_validate(settings_dict)
            
            # 辞書として返す
            return app_settings.model_dump()
        except Exception as e:
            logger.error(f"設定プロファイル読み込みエラー: {e}")
            return None
    
    def list_profiles(self):
        """利用可能な設定プロファイルのリストを取得する
        
        Returns:
            list: プロファイル名のリスト（拡張子なし）
        """
        # 設定ディレクトリが存在しない場合は空リストを返す
        if not self.settings_dir.exists():
            return []
        
        # JSONファイルを検索
        profiles = []
        for file in os.listdir(self.settings_dir):
            if file.endswith('.json'):
                profile_name = os.path.splitext(file)[0]
                profiles.append(profile_name)
        
        return profiles
    
    def delete_profile(self, profile_name):
        """設定プロファイルを削除する
        
        Args:
            profile_name (str): 削除するプロファイル名
        
        Returns:
            bool: 削除成功時はTrue、失敗時はFalse
        """
        # プロファイルファイルのパスを作成
        profile_path = self.settings_dir / f"{profile_name}.json"
        
        # プロファイルファイルが存在するか確認
        if not profile_path.exists():
            logger.warning(f"削除する設定プロファイルが見つかりません: {profile_path}")
            return False
        
        try:
            # ファイルを削除
            os.remove(profile_path)
            logger.info(f"設定プロファイルを削除しました: {profile_path}")
            return True
        except Exception as e:
            logger.error(f"設定プロファイル削除エラー: {e}")
            return False
    
    def get_default_settings(self) -> Dict[str, Any]:
        """デフォルト設定を取得する
        
        Returns:
            dict: デフォルト設定
        """
        # pydanticモデルのデフォルト値を使用
        default_settings = AppSettings()
        return default_settings.model_dump()
    
    def update_profile(self, profile_name, updates, base_settings=None):
        """既存のプロファイルを更新する
        
        Args:
            profile_name (str): 更新するプロファイル名
            updates (dict): 更新する設定値
            base_settings (dict, optional): 基本となる設定。指定がなければ既存プロファイルを読み込む。
        
        Returns:
            dict: 更新後の設定。失敗時はNone。
        """
        # 基本設定の取得
        if base_settings is None:
            base_settings = self.load_profile(profile_name)
            
            # プロファイルが存在しない場合はデフォルト設定から作成
            if base_settings is None:
                base_settings = self.get_default_settings()
        
        # 設定の更新
        updated_settings = base_settings.copy()
        updated_settings.update(updates)
        
        # 更新した設定を保存
        if self.save_profile(profile_name, updated_settings):
            return updated_settings
        else:
            return None
    
    def export_settings(self, settings, output_path):
        """設定をファイルにエクスポートする
        
        Args:
            settings (dict): エクスポートする設定
            output_path (str): 出力ファイルパス
        
        Returns:
            bool: エクスポート成功時はTrue、失敗時はFalse
        """
        try:
            # 設定をJSONファイルとして保存
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            
            logger.info(f"設定をエクスポートしました: {output_path}")
            return True
        except Exception as e:
            logger.error(f"設定エクスポートエラー: {e}")
            return False
    
    def import_settings(self, input_path):
        """設定をファイルからインポートする
        
        Args:
            input_path (str): 入力ファイルパス
        
        Returns:
            dict: インポートした設定。失敗時はNone。
        """
        # ファイルが存在するか確認
        if not os.path.exists(input_path):
            logger.warning(f"インポートするファイルが見つかりません: {input_path}")
            return None
        
        try:
            # 設定をJSONファイルから読み込み
            with open(input_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            
            logger.info(f"設定をインポートしました: {input_path}")
            return settings
        except Exception as e:
            logger.error(f"設定インポートエラー: {e}")
            return None 