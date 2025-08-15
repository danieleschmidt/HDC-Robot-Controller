#!/usr/bin/env python3
"""
Internationalization (i18n) Support for Pipeline Guard
Multi-language support and localization for global deployment

Features:
- Multi-language message translation
- Locale-specific formatting (dates, numbers, currencies)
- Regional compliance and regulatory requirements
- Time zone handling
- Right-to-left (RTL) language support
- Cultural adaptation for alerts and notifications
"""

import json
import os
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import locale
from pathlib import Path

class SupportedLanguage(Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"

class Region(Enum):
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    LATIN_AMERICA = "latam"
    MIDDLE_EAST = "me"
    AFRICA = "af"

@dataclass
class LocaleConfig:
    """Locale configuration for specific region/language"""
    language: SupportedLanguage
    region: Region
    date_format: str
    time_format: str
    number_format: str
    currency: str
    timezone: str
    rtl: bool = False  # Right-to-left text direction
    formal_address: bool = True  # Use formal addressing in messages

class I18nManager:
    """Internationalization manager for Pipeline Guard"""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH,
                 translations_dir: str = "translations"):
        self.default_language = default_language
        self.translations_dir = Path(translations_dir)
        self.current_locale = None
        
        # Translation cache
        self.translations = {}
        
        # Load default translations
        self._load_translations()
        
        # Locale configurations
        self.locale_configs = self._initialize_locale_configs()
        
        # Set default locale
        self.set_locale(default_language, Region.NORTH_AMERICA)
    
    def _load_translations(self):
        """Load translation files"""
        # Create default translations if directory doesn't exist
        if not self.translations_dir.exists():
            self.translations_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations()
        
        # Load translation files
        for lang in SupportedLanguage:
            translation_file = self.translations_dir / f"{lang.value}.json"
            if translation_file.exists():
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations[lang.value] = json.load(f)
            else:
                # Use English as fallback
                self.translations[lang.value] = self.translations.get('en', {})
    
    def _create_default_translations(self):
        """Create default translation files"""
        
        # English translations (base)
        english_translations = {
            # System messages
            "system.startup": "Pipeline Guard system starting up",
            "system.shutdown": "Pipeline Guard system shutting down",
            "system.healthy": "System is healthy",
            "system.degraded": "System performance is degraded",
            "system.critical": "System is in critical state",
            
            # Pipeline status
            "pipeline.status.healthy": "Pipeline is healthy",
            "pipeline.status.degraded": "Pipeline performance is degraded",
            "pipeline.status.critical": "Pipeline is in critical state",
            "pipeline.status.failed": "Pipeline has failed",
            
            # Alerts
            "alert.performance.title": "Performance Degradation Detected",
            "alert.performance.message": "Pipeline {pipeline_id} duration increased by {percentage}%",
            "alert.error.title": "Error Rate Spike",
            "alert.error.message": "Pipeline {pipeline_id} error count: {error_count}",
            "alert.resource.title": "Resource Usage Alert",
            "alert.resource.message": "High resource usage detected: CPU {cpu}%, Memory {memory}%",
            "alert.security.title": "Security Alert",
            "alert.security.message": "Security event detected from {source_ip}",
            
            # Repair actions
            "repair.attempt": "Attempting to repair pipeline {pipeline_id}",
            "repair.success": "Pipeline {pipeline_id} repair completed successfully",
            "repair.failed": "Pipeline {pipeline_id} repair failed: {reason}",
            "repair.strategy.restart": "Restarting services",
            "repair.strategy.scale": "Scaling resources",
            "repair.strategy.rollback": "Rolling back deployment",
            
            # Time and dates
            "time.ago.seconds": "{seconds} seconds ago",
            "time.ago.minutes": "{minutes} minutes ago",
            "time.ago.hours": "{hours} hours ago",
            "time.ago.days": "{days} days ago",
            
            # Numbers and metrics
            "metrics.duration": "Duration: {duration}",
            "metrics.success_rate": "Success Rate: {rate}%",
            "metrics.error_count": "Errors: {count}",
            "metrics.health_score": "Health Score: {score}",
            
            # Dashboard
            "dashboard.title": "Pipeline Guard Dashboard",
            "dashboard.overview": "System Overview",
            "dashboard.pipelines": "Pipelines",
            "dashboard.alerts": "Alerts",
            "dashboard.metrics": "Metrics",
            
            # Notifications
            "notification.subject.alert": "[Pipeline Guard] Alert: {title}",
            "notification.subject.repair": "[Pipeline Guard] Repair: {pipeline_id}",
            "notification.greeting.formal": "Dear {name}",
            "notification.greeting.informal": "Hi {name}",
            "notification.closing.formal": "Best regards,\\nPipeline Guard System",
            "notification.closing.informal": "Cheers,\\nPipeline Guard",
            
            # Compliance
            "compliance.gdpr.notice": "This system processes personal data in accordance with GDPR",
            "compliance.data_retention": "Data is retained for {days} days as per policy",
            "compliance.audit_log": "All actions are logged for audit purposes",
            
            # Errors
            "error.not_found": "Resource not found",
            "error.unauthorized": "Unauthorized access",
            "error.rate_limit": "Rate limit exceeded",
            "error.invalid_input": "Invalid input provided",
            "error.system_error": "Internal system error"
        }
        
        # Spanish translations
        spanish_translations = {
            "system.startup": "Sistema Pipeline Guard iniciándose",
            "system.shutdown": "Sistema Pipeline Guard cerrándose",
            "system.healthy": "El sistema está saludable",
            "system.degraded": "El rendimiento del sistema está degradado",
            "system.critical": "El sistema está en estado crítico",
            
            "pipeline.status.healthy": "El pipeline está saludable",
            "pipeline.status.degraded": "El rendimiento del pipeline está degradado",
            "pipeline.status.critical": "El pipeline está en estado crítico",
            "pipeline.status.failed": "El pipeline ha fallado",
            
            "alert.performance.title": "Degradación de Rendimiento Detectada",
            "alert.performance.message": "La duración del pipeline {pipeline_id} aumentó en {percentage}%",
            "alert.error.title": "Pico de Tasa de Error",
            "alert.error.message": "Conteo de errores del pipeline {pipeline_id}: {error_count}",
            "alert.resource.title": "Alerta de Uso de Recursos",
            "alert.resource.message": "Alto uso de recursos detectado: CPU {cpu}%, Memoria {memory}%",
            
            "repair.attempt": "Intentando reparar el pipeline {pipeline_id}",
            "repair.success": "Reparación del pipeline {pipeline_id} completada exitosamente",
            "repair.failed": "Reparación del pipeline {pipeline_id} falló: {reason}",
            
            "dashboard.title": "Panel de Control de Pipeline Guard",
            "dashboard.overview": "Resumen del Sistema",
            "dashboard.pipelines": "Pipelines",
            "dashboard.alerts": "Alertas",
            "dashboard.metrics": "Métricas",
            
            "notification.greeting.formal": "Estimado/a {name}",
            "notification.greeting.informal": "Hola {name}",
            "notification.closing.formal": "Saludos cordiales,\\nSistema Pipeline Guard",
            "notification.closing.informal": "Saludos,\\nPipeline Guard",
            
            "compliance.gdpr.notice": "Este sistema procesa datos personales de acuerdo con GDPR",
            "error.not_found": "Recurso no encontrado",
            "error.unauthorized": "Acceso no autorizado"
        }
        
        # French translations
        french_translations = {
            "system.startup": "Démarrage du système Pipeline Guard",
            "system.shutdown": "Arrêt du système Pipeline Guard",
            "system.healthy": "Le système est en bonne santé",
            "system.degraded": "Les performances du système sont dégradées",
            "system.critical": "Le système est dans un état critique",
            
            "pipeline.status.healthy": "Le pipeline est en bonne santé",
            "pipeline.status.degraded": "Les performances du pipeline sont dégradées",
            "pipeline.status.critical": "Le pipeline est dans un état critique",
            "pipeline.status.failed": "Le pipeline a échoué",
            
            "alert.performance.title": "Dégradation des Performances Détectée",
            "alert.performance.message": "La durée du pipeline {pipeline_id} a augmenté de {percentage}%",
            "alert.error.title": "Pic du Taux d'Erreur",
            "alert.error.message": "Nombre d'erreurs du pipeline {pipeline_id}: {error_count}",
            
            "repair.attempt": "Tentative de réparation du pipeline {pipeline_id}",
            "repair.success": "Réparation du pipeline {pipeline_id} terminée avec succès",
            "repair.failed": "Réparation du pipeline {pipeline_id} échouée: {reason}",
            
            "dashboard.title": "Tableau de Bord Pipeline Guard",
            "dashboard.overview": "Vue d'Ensemble du Système",
            "dashboard.pipelines": "Pipelines",
            "dashboard.alerts": "Alertes",
            "dashboard.metrics": "Métriques",
            
            "notification.greeting.formal": "Cher/Chère {name}",
            "notification.greeting.informal": "Salut {name}",
            "notification.closing.formal": "Cordialement,\\nSystème Pipeline Guard",
            "notification.closing.informal": "À bientôt,\\nPipeline Guard",
            
            "error.not_found": "Ressource non trouvée",
            "error.unauthorized": "Accès non autorisé"
        }
        
        # German translations
        german_translations = {
            "system.startup": "Pipeline Guard System startet",
            "system.shutdown": "Pipeline Guard System wird heruntergefahren",
            "system.healthy": "System ist gesund",
            "system.degraded": "Systemleistung ist beeinträchtigt",
            "system.critical": "System befindet sich in kritischem Zustand",
            
            "pipeline.status.healthy": "Pipeline ist gesund",
            "pipeline.status.degraded": "Pipeline-Leistung ist beeinträchtigt",
            "pipeline.status.critical": "Pipeline befindet sich in kritischem Zustand",
            "pipeline.status.failed": "Pipeline ist fehlgeschlagen",
            
            "alert.performance.title": "Leistungsabfall Erkannt",
            "alert.performance.message": "Pipeline {pipeline_id} Dauer um {percentage}% gestiegen",
            "alert.error.title": "Fehlerrate-Spitze",
            "alert.error.message": "Pipeline {pipeline_id} Fehleranzahl: {error_count}",
            
            "repair.attempt": "Versuch, Pipeline {pipeline_id} zu reparieren",
            "repair.success": "Pipeline {pipeline_id} Reparatur erfolgreich abgeschlossen",
            "repair.failed": "Pipeline {pipeline_id} Reparatur fehlgeschlagen: {reason}",
            
            "dashboard.title": "Pipeline Guard Dashboard",
            "dashboard.overview": "Systemübersicht",
            "dashboard.pipelines": "Pipelines",
            "dashboard.alerts": "Warnungen",
            "dashboard.metrics": "Metriken",
            
            "notification.greeting.formal": "Sehr geehrte/r {name}",
            "notification.greeting.informal": "Hallo {name}",
            "notification.closing.formal": "Mit freundlichen Grüßen,\\nPipeline Guard System",
            "notification.closing.informal": "Viele Grüße,\\nPipeline Guard",
            
            "error.not_found": "Ressource nicht gefunden",
            "error.unauthorized": "Unbefugter Zugriff"
        }
        
        # Japanese translations
        japanese_translations = {
            "system.startup": "Pipeline Guardシステムが起動中です",
            "system.shutdown": "Pipeline Guardシステムが終了中です",
            "system.healthy": "システムは正常です",
            "system.degraded": "システムパフォーマンスが低下しています",
            "system.critical": "システムが重要な状態にあります",
            
            "pipeline.status.healthy": "パイプラインは正常です",
            "pipeline.status.degraded": "パイプラインのパフォーマンスが低下しています",
            "pipeline.status.critical": "パイプラインが重要な状態にあります",
            "pipeline.status.failed": "パイプラインが失敗しました",
            
            "alert.performance.title": "パフォーマンス低下を検出",
            "alert.performance.message": "パイプライン{pipeline_id}の実行時間が{percentage}%増加しました",
            "alert.error.title": "エラー率の急増",
            "alert.error.message": "パイプライン{pipeline_id}のエラー数: {error_count}",
            
            "repair.attempt": "パイプライン{pipeline_id}の修復を試行中",
            "repair.success": "パイプライン{pipeline_id}の修復が正常に完了しました",
            "repair.failed": "パイプライン{pipeline_id}の修復が失敗しました: {reason}",
            
            "dashboard.title": "Pipeline Guardダッシュボード",
            "dashboard.overview": "システム概要",
            "dashboard.pipelines": "パイプライン",
            "dashboard.alerts": "アラート",
            "dashboard.metrics": "メトリクス",
            
            "notification.greeting.formal": "{name}様",
            "notification.greeting.informal": "{name}さん",
            "notification.closing.formal": "よろしくお願いいたします。\\nPipeline Guardシステム",
            "notification.closing.informal": "よろしく、\\nPipeline Guard",
            
            "error.not_found": "リソースが見つかりません",
            "error.unauthorized": "認証されていないアクセス"
        }
        
        # Chinese Simplified translations
        chinese_translations = {
            "system.startup": "Pipeline Guard系统正在启动",
            "system.shutdown": "Pipeline Guard系统正在关闭",
            "system.healthy": "系统状态良好",
            "system.degraded": "系统性能下降",
            "system.critical": "系统处于紧急状态",
            
            "pipeline.status.healthy": "管道状态良好",
            "pipeline.status.degraded": "管道性能下降",
            "pipeline.status.critical": "管道处于紧急状态",
            "pipeline.status.failed": "管道失败",
            
            "alert.performance.title": "检测到性能下降",
            "alert.performance.message": "管道{pipeline_id}持续时间增加{percentage}%",
            "alert.error.title": "错误率激增",
            "alert.error.message": "管道{pipeline_id}错误数量: {error_count}",
            
            "repair.attempt": "正在尝试修复管道{pipeline_id}",
            "repair.success": "管道{pipeline_id}修复成功完成",
            "repair.failed": "管道{pipeline_id}修复失败: {reason}",
            
            "dashboard.title": "Pipeline Guard仪表板",
            "dashboard.overview": "系统概览",
            "dashboard.pipelines": "管道",
            "dashboard.alerts": "警报",
            "dashboard.metrics": "指标",
            
            "notification.greeting.formal": "尊敬的{name}",
            "notification.greeting.informal": "你好{name}",
            "notification.closing.formal": "此致敬礼，\\nPipeline Guard系统",
            "notification.closing.informal": "祝好，\\nPipeline Guard",
            
            "error.not_found": "资源未找到",
            "error.unauthorized": "未授权访问"
        }
        
        # Save translation files
        translations_map = {
            "en": english_translations,
            "es": spanish_translations,
            "fr": french_translations,
            "de": german_translations,
            "ja": japanese_translations,
            "zh-cn": chinese_translations
        }
        
        for lang_code, translations in translations_map.items():
            file_path = self.translations_dir / f"{lang_code}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
    
    def _initialize_locale_configs(self) -> Dict[str, LocaleConfig]:
        """Initialize locale configurations for different regions"""
        return {
            "en_na": LocaleConfig(
                language=SupportedLanguage.ENGLISH,
                region=Region.NORTH_AMERICA,
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                number_format="1,234.56",
                currency="USD",
                timezone="America/New_York"
            ),
            "en_eu": LocaleConfig(
                language=SupportedLanguage.ENGLISH,
                region=Region.EUROPE,
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                currency="EUR",
                timezone="Europe/London"
            ),
            "es_latam": LocaleConfig(
                language=SupportedLanguage.SPANISH,
                region=Region.LATIN_AMERICA,
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                currency="USD",
                timezone="America/Mexico_City"
            ),
            "fr_eu": LocaleConfig(
                language=SupportedLanguage.FRENCH,
                region=Region.EUROPE,
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1 234,56",
                currency="EUR",
                timezone="Europe/Paris"
            ),
            "de_eu": LocaleConfig(
                language=SupportedLanguage.GERMAN,
                region=Region.EUROPE,
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                currency="EUR",
                timezone="Europe/Berlin"
            ),
            "ja_apac": LocaleConfig(
                language=SupportedLanguage.JAPANESE,
                region=Region.ASIA_PACIFIC,
                date_format="%Y/%m/%d",
                time_format="%H:%M",
                number_format="1,234.56",
                currency="JPY",
                timezone="Asia/Tokyo"
            ),
            "zh-cn_apac": LocaleConfig(
                language=SupportedLanguage.CHINESE_SIMPLIFIED,
                region=Region.ASIA_PACIFIC,
                date_format="%Y-%m-%d",
                time_format="%H:%M",
                number_format="1,234.56",
                currency="CNY",
                timezone="Asia/Shanghai"
            ),
            "ar_me": LocaleConfig(
                language=SupportedLanguage.ARABIC,
                region=Region.MIDDLE_EAST,
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1,234.56",
                currency="USD",
                timezone="Asia/Dubai",
                rtl=True
            )
        }
    
    def set_locale(self, language: SupportedLanguage, region: Region):
        """Set current locale"""
        locale_key = f"{language.value}_{region.value}"
        
        if locale_key in self.locale_configs:
            self.current_locale = self.locale_configs[locale_key]
        else:
            # Use default English locale
            self.current_locale = self.locale_configs["en_na"]
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate message key with optional parameters"""
        if not self.current_locale:
            return key
        
        lang_code = self.current_locale.language.value
        translations = self.translations.get(lang_code, {})
        
        # Get translation or fall back to English
        message = translations.get(key)
        if not message and lang_code != 'en':
            message = self.translations.get('en', {}).get(key, key)
        elif not message:
            message = key
        
        # Format with parameters
        if kwargs:
            try:
                message = message.format(**kwargs)
            except (KeyError, ValueError):
                # If formatting fails, return original message
                pass
        
        return message
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to current locale"""
        if not self.current_locale:
            return dt.isoformat()
        
        # Convert to locale timezone
        try:
            import pytz
            tz = pytz.timezone(self.current_locale.timezone)
            localized_dt = dt.astimezone(tz)
        except ImportError:
            localized_dt = dt
        
        # Format according to locale
        date_str = localized_dt.strftime(self.current_locale.date_format)
        time_str = localized_dt.strftime(self.current_locale.time_format)
        
        return f"{date_str} {time_str}"
    
    def format_number(self, number: Union[int, float], decimal_places: int = 2) -> str:
        """Format number according to current locale"""
        if not self.current_locale:
            return str(number)
        
        # Format based on locale number format
        if self.current_locale.number_format == "1,234.56":
            # US/UK format
            return f"{number:,.{decimal_places}f}"
        elif self.current_locale.number_format == "1.234,56":
            # European format
            formatted = f"{number:,.{decimal_places}f}"
            # Swap , and .
            formatted = formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
            return formatted
        elif self.current_locale.number_format == "1 234,56":
            # French format
            formatted = f"{number:,.{decimal_places}f}"
            # Replace , with space and . with ,
            formatted = formatted.replace(",", " ").replace(".", ",")
            return formatted
        else:
            return f"{number:.{decimal_places}f}"
    
    def format_currency(self, amount: float) -> str:
        """Format currency according to current locale"""
        if not self.current_locale:
            return f"${amount:.2f}"
        
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "JPY": "¥",
            "CNY": "¥",
            "GBP": "£"
        }
        
        symbol = currency_symbols.get(self.current_locale.currency, self.current_locale.currency)
        formatted_amount = self.format_number(amount, 2)
        
        # Currency symbol placement varies by locale
        if self.current_locale.language in [SupportedLanguage.ENGLISH]:
            return f"{symbol}{formatted_amount}"
        else:
            return f"{formatted_amount} {symbol}"
    
    def get_notification_greeting(self, name: str, formal: bool = None) -> str:
        """Get localized notification greeting"""
        if formal is None:
            formal = self.current_locale.formal_address if self.current_locale else True
        
        greeting_key = "notification.greeting.formal" if formal else "notification.greeting.informal"
        return self.translate(greeting_key, name=name)
    
    def get_notification_closing(self, formal: bool = None) -> str:
        """Get localized notification closing"""
        if formal is None:
            formal = self.current_locale.formal_address if self.current_locale else True
        
        closing_key = "notification.closing.formal" if formal else "notification.closing.informal"
        return self.translate(closing_key)
    
    def get_compliance_notice(self) -> str:
        """Get compliance notice for current region"""
        if not self.current_locale:
            return ""
        
        # Different compliance requirements by region
        if self.current_locale.region == Region.EUROPE:
            return self.translate("compliance.gdpr.notice")
        elif self.current_locale.region == Region.NORTH_AMERICA:
            return self.translate("compliance.data_retention", days=365)
        else:
            return self.translate("compliance.audit_log")
    
    def is_rtl_language(self) -> bool:
        """Check if current language is right-to-left"""
        return self.current_locale and self.current_locale.rtl
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        language_names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Español",
            SupportedLanguage.FRENCH: "Français",
            SupportedLanguage.GERMAN: "Deutsch",
            SupportedLanguage.JAPANESE: "日本語",
            SupportedLanguage.CHINESE_SIMPLIFIED: "简体中文",
            SupportedLanguage.CHINESE_TRADITIONAL: "繁體中文",
            SupportedLanguage.KOREAN: "한국어",
            SupportedLanguage.PORTUGUESE: "Português",
            SupportedLanguage.RUSSIAN: "Русский",
            SupportedLanguage.ARABIC: "العربية",
            SupportedLanguage.HINDI: "हिन्दी"
        }
        
        return [
            {
                "code": lang.value,
                "name": language_names.get(lang, lang.value),
                "native_name": language_names.get(lang, lang.value)
            }
            for lang in SupportedLanguage
        ]

# Global i18n instance
_i18n_manager = None

def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance"""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager

def translate(key: str, **kwargs) -> str:
    """Convenience function for translation"""
    return get_i18n_manager().translate(key, **kwargs)

def set_language(language: SupportedLanguage, region: Region = Region.NORTH_AMERICA):
    """Convenience function to set language"""
    get_i18n_manager().set_locale(language, region)

# Example usage and testing
if __name__ == "__main__":
    # Initialize i18n manager
    i18n = I18nManager()
    
    # Test English (default)
    print("=== English (US) ===")
    i18n.set_locale(SupportedLanguage.ENGLISH, Region.NORTH_AMERICA)
    print(i18n.translate("system.startup"))
    print(i18n.translate("alert.performance.message", pipeline_id="web-app", percentage=25))
    print(i18n.format_datetime(datetime.now()))
    print(i18n.format_currency(1234.56))
    
    # Test Spanish
    print("\\n=== Español (LATAM) ===")
    i18n.set_locale(SupportedLanguage.SPANISH, Region.LATIN_AMERICA)
    print(i18n.translate("system.startup"))
    print(i18n.translate("alert.performance.message", pipeline_id="web-app", percentage=25))
    print(i18n.format_datetime(datetime.now()))
    print(i18n.format_currency(1234.56))
    
    # Test French
    print("\\n=== Français (EU) ===")
    i18n.set_locale(SupportedLanguage.FRENCH, Region.EUROPE)
    print(i18n.translate("system.startup"))
    print(i18n.translate("alert.performance.message", pipeline_id="web-app", percentage=25))
    print(i18n.format_datetime(datetime.now()))
    print(i18n.format_currency(1234.56))
    
    # Test Japanese
    print("\\n=== 日本語 (APAC) ===")
    i18n.set_locale(SupportedLanguage.JAPANESE, Region.ASIA_PACIFIC)
    print(i18n.translate("system.startup"))
    print(i18n.translate("alert.performance.message", pipeline_id="web-app", percentage=25))
    print(i18n.format_datetime(datetime.now()))
    print(i18n.format_currency(1234.56))
    
    # Test notification formatting
    print("\\n=== Notification Examples ===")
    i18n.set_locale(SupportedLanguage.ENGLISH, Region.NORTH_AMERICA)
    print("Formal:", i18n.get_notification_greeting("John Smith", formal=True))
    print("Informal:", i18n.get_notification_greeting("John", formal=False))
    print("Closing:", i18n.get_notification_closing(formal=True))
    
    # Test compliance notices
    print("\\n=== Compliance Notices ===")
    i18n.set_locale(SupportedLanguage.ENGLISH, Region.EUROPE)
    print("EU:", i18n.get_compliance_notice())
    i18n.set_locale(SupportedLanguage.ENGLISH, Region.NORTH_AMERICA)
    print("NA:", i18n.get_compliance_notice())
    
    print("\\nI18n system test completed.")