"""
🌲 Isolation Forest Anomaly Detector

Implements Isolation Forest algorithm for anomaly detection.
Uses ensemble of isolation trees to identify anomalies by their isolation difficulty.
Anomalies are easier to isolate and have shorter average path lengths.

Context7 Features:
- Distributed tree ensemble
- Auto-scaling parameters
- Model versioning and persistence
- Real-time inference optimization
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest detector."""
    n_estimators: int = 200  # Количество деревьев
    max_samples: Union[int, float, str] = 'auto'  # Размер выборки для каждого дерева
    contamination: float = 0.1  # Ожидаемая доля аномалий
    max_features: float = 1.0  # Доля признаков для каждого дерева
    bootstrap: bool = False  # Bootstrap sampling
    n_jobs: int = -1  # Количество процессов для параллельного обучения
    random_state: Optional[int] = 42
    warm_start: bool = False  # Инкрементальное обучение
    
    # Context7 enterprise features
    model_versioning: bool = True  # Версионирование модели
    auto_scaling: bool = True  # Автоматическое масштабирование
    scaler_type: str = 'standard'  # 'standard', 'robust', 'none'
    performance_tracking: bool = True  # Отслеживание производительности
    
    # Crypto-specific parameters
    crypto_optimized: bool = True  # Оптимизация для крипто данных
    volatility_aware: bool = True  # Учет волатильности

class IsolationForestDetector:
    """
    Isolation Forest Anomaly Detector для криптотрейдинга.
    
    Использует ансамбль изолирующих деревьев для выявления аномалий.
    Аномалии легче изолируются и имеют меньшую среднюю глубину пути.
    
    Context7 Features:
    - Enterprise-grade model management
    - Distributed processing capability
    - Real-time inference optimization
    - Comprehensive monitoring
    """
    
    def __init__(self, config: Optional[IsolationForestConfig] = None):
        """
        Инициализация Isolation Forest детектора.
        
        Args:
            config: Конфигурация детектора
        """
        self.config = config or IsolationForestConfig()
        self.fitted = False
        self.model = None
        self.scaler = None
        self._feature_names = None
        self._model_version = "1.0.0"
        self._training_stats = {}
        self._performance_metrics = {}
        
        # Создаем модель
        self._create_model()
        self._create_scaler()
        
        logger.info(
            "IsolationForestDetector initialized",
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            crypto_optimized=self.config.crypto_optimized,
            model_version=self._model_version
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
            y: Optional[np.ndarray] = None) -> 'IsolationForestDetector':
        """
        Обучение детектора на исторических данных.
        
        Args:
            X: Исторические данные для обучения
            y: Истинные метки (опционально, для оценки качества)
            
        Returns:
            self: Обученный детектор
        """
        try:
            X, feature_names = self._validate_and_prepare_input(X)
            self._feature_names = feature_names
            
            # Сохраняем статистики обучающих данных
            self._training_stats = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "feature_means": np.mean(X, axis=0).tolist(),
                "feature_stds": np.std(X, axis=0).tolist()
            }
            
            # Масштабирование признаков
            if self.scaler is not None:
                logger.info("Scaling features", scaler_type=self.config.scaler_type)
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Crypto-specific оптимизация параметров
            if self.config.crypto_optimized:
                self._optimize_for_crypto_data(X_scaled)
            
            # Обучение модели
            logger.info("Training Isolation Forest model")
            self.model.fit(X_scaled)
            
            self.fitted = True
            
            # Оценка качества если есть истинные метки
            if y is not None:
                self._evaluate_performance(X_scaled, y)
            
            logger.info(
                "IsolationForestDetector fitted successfully",
                **self._training_stats,
                model_version=self._model_version,
                crypto_optimized=self.config.crypto_optimized
            )
            
            return self
            
        except Exception as e:
            logger.error("Failed to fit IsolationForestDetector", error=str(e))
            raise
    
    def detect(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обнаружение аномалий в данных.
        
        Args:
            X: Данные для анализа
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_labels, anomaly_scores)
                anomaly_labels: 1 для аномалий, 0 для нормальных точек
                anomaly_scores: Оценки аномальности (чем меньше, тем более аномальная точка)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detecting anomalies")
        
        try:
            X, _ = self._validate_and_prepare_input(X)
            
            # Масштабирование признаков
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Предсказание аномалий
            anomaly_labels = self.model.predict(X_scaled)
            # Преобразуем -1/1 в 0/1
            anomaly_labels = np.where(anomaly_labels == -1, 1, 0)
            
            # Получаем оценки аномальности
            anomaly_scores = self.model.decision_function(X_scaled)
            # Инвертируем оценки (чем меньше, тем более аномально)
            anomaly_scores = -anomaly_scores
            
            n_anomalies = np.sum(anomaly_labels)
            
            logger.debug(
                "Isolation Forest detection completed",
                n_samples=len(X),
                n_anomalies=n_anomalies,
                anomaly_rate=f"{np.mean(anomaly_labels):.3%}",
                min_score=np.min(anomaly_scores),
                max_score=np.max(anomaly_scores)
            )
            
            return anomaly_labels, anomaly_scores
            
        except Exception as e:
            logger.error("Failed to detect anomalies with Isolation Forest", error=str(e))
            raise
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Вероятности аномальности для каждого образца.
        
        Args:
            X: Данные для анализа
            
        Returns:
            np.ndarray: Вероятности аномальности [0, 1]
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted")
        
        try:
            X, _ = self._validate_and_prepare_input(X)
            
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Получаем оценки решения
            decision_scores = self.model.decision_function(X_scaled)
            
            # Преобразуем в вероятности [0, 1]
            # Используем сигмоид для нормализации
            probabilities = 1 / (1 + np.exp(decision_scores))
            
            return probabilities
            
        except Exception as e:
            logger.error("Failed to compute anomaly probabilities", error=str(e))
            raise
    
    def detect_realtime(self, value: Union[float, np.ndarray, Dict]) -> Tuple[bool, float, float]:
        """
        Real-time обнаружение аномалий для одной точки данных.
        
        Args:
            value: Значение для проверки
            
        Returns:
            Tuple[bool, float, float]: (is_anomaly, anomaly_score, probability)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before real-time detection")
        
        try:
            # Подготовка данных
            if isinstance(value, dict):
                if self._feature_names:
                    value = np.array([value[name] for name in self._feature_names])
                else:
                    value = np.array(list(value.values()))
            elif isinstance(value, (int, float)):
                value = np.array([value])
            elif isinstance(value, list):
                value = np.array(value)
            
            value = value.reshape(1, -1)
            
            # Масштабирование
            if self.scaler is not None:
                value_scaled = self.scaler.transform(value)
            else:
                value_scaled = value
            
            # Предсказание
            prediction = self.model.predict(value_scaled)[0]
            is_anomaly = prediction == -1
            
            # Оценки
            anomaly_score = -self.model.decision_function(value_scaled)[0]
            probability = 1 / (1 + np.exp(-anomaly_score))
            
            if is_anomaly:
                logger.warning(
                    "Real-time Isolation Forest anomaly detected",
                    value=value[0].tolist() if value.ndim > 1 else float(value[0]),
                    anomaly_score=anomaly_score,
                    probability=probability
                )
            
            return is_anomaly, anomaly_score, probability
            
        except Exception as e:
            logger.error("Failed real-time Isolation Forest detection", error=str(e))
            raise
    
    def _validate_and_prepare_input(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Валидация и подготовка входных данных."""
        feature_names = None
        
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, pd.Series):
            feature_names = [X.name] if X.name else ["feature_0"]
            X = X.values.reshape(-1, 1)
        elif isinstance(X, list):
            X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            if not feature_names:
                feature_names = ["feature_0"]
        
        # Проверка на NaN и Inf
        if np.any(~np.isfinite(X)):
            warnings.warn("Input contains NaN or Inf values, replacing with median")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        return X, feature_names
    
    def _create_model(self) -> None:
        """Создание модели Isolation Forest."""
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            warm_start=self.config.warm_start,
            verbose=0
        )
    
    def _create_scaler(self) -> None:
        """Создание масштабировщика признаков."""
        if self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.config.scaler_type == 'none':
            self.scaler = None
        else:
            logger.warning(
                "Unknown scaler type, using standard",
                scaler_type=self.config.scaler_type
            )
            self.scaler = StandardScaler()
    
    def _optimize_for_crypto_data(self, X: np.ndarray) -> None:
        """Оптимизация параметров для криптоданных."""
        if not self.config.crypto_optimized:
            return
        
        n_samples, n_features = X.shape
        
        # Адаптация contamination на основе волатильности данных
        if self.config.volatility_aware and n_features > 1:
            volatilities = np.std(X, axis=0)
            avg_volatility = np.mean(volatilities)
            
            # Увеличиваем contamination для более волатильных данных
            if avg_volatility > 1.0:
                new_contamination = min(0.2, self.config.contamination * 1.5)
                logger.info(
                    "Adjusting contamination for high volatility crypto data",
                    old_contamination=self.config.contamination,
                    new_contamination=new_contamination,
                    avg_volatility=avg_volatility
                )
                self.model.contamination = new_contamination
        
        # Оптимизация max_samples для crypto данных
        if isinstance(self.config.max_samples, str) and self.config.max_samples == 'auto':
            # Для криптоданных используем меньшие выборки для лучшего обнаружения аномалий
            optimal_samples = min(512, max(64, n_samples // 10))
            logger.info(
                "Optimizing max_samples for crypto data",
                original="auto",
                optimized=optimal_samples,
                n_samples=n_samples
            )
            self.model.max_samples = optimal_samples
    
    def _evaluate_performance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Оценка производительности модели."""
        try:
            # Получаем предсказания
            predictions = self.model.predict(X)
            decision_scores = self.model.decision_function(X)
            
            # Преобразуем метки
            y_binary = np.where(y == 1, 1, 0)  # Аномалии = 1
            pred_binary = np.where(predictions == -1, 1, 0)  # Изолейшн форест: -1 = аномалия
            
            # Вычисляем метрики
            if len(np.unique(y_binary)) > 1:  # Проверяем наличие обеих классов
                auc_score = roc_auc_score(y_binary, -decision_scores)
                precision, recall, _ = precision_recall_curve(y_binary, -decision_scores)
                auc_pr = np.trapz(precision, recall)
                
                self._performance_metrics = {
                    "auc_roc": auc_score,
                    "auc_pr": auc_pr,
                    "precision_at_contamination": precision[len(precision)//2],  # Примерная оценка
                    "recall_at_contamination": recall[len(recall)//2]
                }
                
                logger.info(
                    "Model performance evaluation",
                    **self._performance_metrics
                )
            
        except Exception as e:
            logger.warning("Failed to evaluate model performance", error=str(e))
    
    def save_model(self, filepath: str) -> None:
        """Сохранение модели на диск."""
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "config": self.config,
                "feature_names": self._feature_names,
                "model_version": self._model_version,
                "training_stats": self._training_stats,
                "performance_metrics": self._performance_metrics
            }
            
            joblib.dump(model_data, filepath)
            
            logger.info(
                "Model saved successfully",
                filepath=filepath,
                model_version=self._model_version
            )
            
        except Exception as e:
            logger.error("Failed to save model", filepath=filepath, error=str(e))
            raise
    
    def load_model(self, filepath: str) -> 'IsolationForestDetector':
        """Загрузка модели с диска."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.config = model_data["config"]
            self._feature_names = model_data.get("feature_names")
            self._model_version = model_data.get("model_version", "unknown")
            self._training_stats = model_data.get("training_stats", {})
            self._performance_metrics = model_data.get("performance_metrics", {})
            self.fitted = True
            
            logger.info(
                "Model loaded successfully",
                filepath=filepath,
                model_version=self._model_version
            )
            
            return self
            
        except Exception as e:
            logger.error("Failed to load model", filepath=filepath, error=str(e))
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистик детектора."""
        if not self.fitted:
            return {"status": "not_fitted"}
        
        return {
            "fitted": True,
            "model_version": self._model_version,
            "config": {
                "n_estimators": self.config.n_estimators,
                "contamination": self.config.contamination,
                "max_features": self.config.max_features,
                "crypto_optimized": self.config.crypto_optimized
            },
            "training_stats": self._training_stats,
            "performance_metrics": self._performance_metrics,
            "feature_names": self._feature_names
        }

# Пример использования для криптотрейдинга
def create_crypto_isolation_forest(
    price_data: pd.DataFrame,
    features: Optional[List[str]] = None,
    contamination: float = 0.05,
    n_estimators: int = 200
) -> IsolationForestDetector:
    """
    Создание Isolation Forest детектора для криптоданных.
    
    Args:
        price_data: DataFrame с ценовыми данными
        features: Список признаков для анализа
        contamination: Ожидаемая доля аномалий
        n_estimators: Количество деревьев
        
    Returns:
        Настроенный IsolationForestDetector
    """
    if features is None:
        features = ['close', 'volume']
        # Добавляем технические индикаторы если есть
        optional_features = ['returns', 'volatility', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
        for feature in optional_features:
            if feature in price_data.columns:
                features.append(feature)
    
    config = IsolationForestConfig(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=0.8,  # Используем 80% признаков для каждого дерева
        crypto_optimized=True,
        volatility_aware=True,
        scaler_type='robust',  # Robust scaler лучше для криптоданных
        auto_scaling=True
    )
    
    detector = IsolationForestDetector(config)
    detector.fit(price_data[features].dropna())
    
    return detector