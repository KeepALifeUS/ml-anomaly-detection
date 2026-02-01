"""
📊 Median Absolute Deviation (MAD) Anomaly Detector

Implements MAD-based anomaly detection - more robust to outliers than Z-score.
MAD is less sensitive to extreme values and works better with non-normal distributions.

Formula: MAD = median(|X - median(X)|)
Modified Z-score: M = 0.6745 * (X - median) / MAD

Context7 Features:
- Robust outlier detection
- Streaming data support
- Auto-adaptive thresholds
- Enterprise monitoring
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
from scipy import stats
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class MADConfig:
    """Configuration for MAD detector."""
    threshold: float = 3.5  # Modified Z-score threshold
    window_size: Optional[int] = None
    rolling_window: bool = False
    min_samples: int = 20
    bilateral: bool = True
    auto_threshold: bool = False
    contamination: float = 0.1
    consistency_constant: float = 1.4826  # For normal distribution consistency

class MADDetector:
    """
    Median Absolute Deviation Anomaly Detector.
    
    Более робастный чем Z-score детектор, особенно для данных с выбросами.
    Использует медиану вместо среднего и MAD вместо стандартного отклонения.
    
    Context7 Features:
    - Distributed processing ready
    - Auto-scaling parameters
    - Real-time detection
    - Robust statistics
    """
    
    def __init__(self, config: Optional[MADConfig] = None):
        """
        Инициализация MAD детектора.
        
        Args:
            config: Конфигурация детектора
        """
        self.config = config or MADConfig()
        self.fitted = False
        self._median = None
        self._mad = None
        self._scaling_factor = self.config.consistency_constant
        
        logger.info(
            "MADDetector initialized",
            threshold=self.config.threshold,
            consistency_constant=self.config.consistency_constant,
            auto_threshold=self.config.auto_threshold
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> 'MADDetector':
        """
        Обучение детектора на исторических данных.
        
        Args:
            X: Исторические данные для обучения
            
        Returns:
            self: Обученный детектор
        """
        try:
            X = self._validate_input(X)
            
            if len(X) < self.config.min_samples:
                raise ValueError(
                    f"Insufficient samples: {len(X)} < {self.config.min_samples}"
                )
            
            # Вычисляем медиану и MAD
            self._median = np.median(X, axis=0)
            
            # MAD = median(|X - median(X)|)
            deviations = np.abs(X - self._median)
            self._mad = np.median(deviations, axis=0)
            
            # Применяем константу консистентности для нормального распределения
            self._mad *= self._scaling_factor
            
            # Защита от деления на ноль
            self._mad = np.where(self._mad == 0, np.finfo(float).eps, self._mad)
            
            # Автоматическая настройка порога
            if self.config.auto_threshold:
                self._auto_tune_threshold(X)
            
            self.fitted = True
            
            logger.info(
                "MADDetector fitted successfully",
                n_samples=len(X),
                n_features=X.shape[1] if X.ndim > 1 else 1,
                median=self._median.tolist() if isinstance(self._median, np.ndarray) else self._median,
                mad=self._mad.tolist() if isinstance(self._mad, np.ndarray) else self._mad
            )
            
            return self
            
        except Exception as e:
            logger.error("Failed to fit MADDetector", error=str(e))
            raise
    
    def detect(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обнаружение аномалий в данных.
        
        Args:
            X: Данные для анализа
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_labels, anomaly_scores)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detecting anomalies")
        
        try:
            X = self._validate_input(X)
            
            if self.config.rolling_window and self.config.window_size:
                anomaly_labels, modified_z_scores = self._detect_rolling(X)
            else:
                modified_z_scores = self._calculate_modified_zscore(X)
                anomaly_labels = self._classify_anomalies(modified_z_scores)
            
            logger.debug(
                "MAD anomaly detection completed",
                n_samples=len(X),
                n_anomalies=np.sum(anomaly_labels),
                anomaly_rate=f"{np.mean(anomaly_labels):.3%}",
                max_score=np.max(np.abs(modified_z_scores))
            )
            
            return anomaly_labels, np.abs(modified_z_scores)
            
        except Exception as e:
            logger.error("Failed to detect anomalies with MAD", error=str(e))
            raise
    
    def detect_realtime(self, value: Union[float, np.ndarray]) -> Tuple[bool, float]:
        """
        Real-time обнаружение аномалий для одной точки данных.
        
        Args:
            value: Значение для проверки
            
        Returns:
            Tuple[bool, float]: (is_anomaly, modified_z_score)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before real-time detection")
        
        try:
            if isinstance(value, (int, float)):
                value = np.array([value])
            elif isinstance(value, list):
                value = np.array(value)
            
            modified_z_score = self._calculate_modified_zscore(value.reshape(1, -1))[0]
            is_anomaly = abs(modified_z_score) > self.config.threshold
            
            if is_anomaly:
                logger.warning(
                    "Real-time MAD anomaly detected",
                    value=value,
                    modified_z_score=modified_z_score,
                    threshold=self.config.threshold
                )
            
            return is_anomaly, abs(modified_z_score)
            
        except Exception as e:
            logger.error("Failed real-time MAD detection", error=str(e))
            raise
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Валидация и конвертация входных данных."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        elif isinstance(X, list):
            X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Проверка на NaN и Inf
        if np.any(~np.isfinite(X)):
            warnings.warn("Input contains NaN or Inf values, removing them")
            X = X[np.isfinite(X).all(axis=1)]
        
        return X
    
    def _calculate_modified_zscore(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление модифицированного Z-score с использованием MAD.
        
        Modified Z-score = 0.6745 * (X - median) / MAD
        """
        # Вычисляем модифицированный Z-score
        modified_z_scores = 0.6745 * (X - self._median) / self._mad
        
        # Для многомерных данных используем максимальный score
        if modified_z_scores.ndim > 1 and modified_z_scores.shape[1] > 1:
            modified_z_scores = np.max(np.abs(modified_z_scores), axis=1)
        else:
            modified_z_scores = modified_z_scores.flatten()
        
        return modified_z_scores
    
    def _classify_anomalies(self, modified_z_scores: np.ndarray) -> np.ndarray:
        """Классификация аномалий на основе модифицированных Z-scores."""
        if self.config.bilateral:
            return (np.abs(modified_z_scores) > self.config.threshold).astype(int)
        else:
            return (modified_z_scores > self.config.threshold).astype(int)
    
    def _detect_rolling(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Обнаружение аномалий с использованием скользящего окна."""
        window_size = self.config.window_size
        n_samples = len(X)
        
        anomaly_labels = np.zeros(n_samples)
        modified_z_scores = np.zeros(n_samples)
        
        for i in range(window_size, n_samples):
            window_data = X[i-window_size:i]
            
            # Вычисляем статистики для окна
            window_median = np.median(window_data, axis=0)
            window_mad = np.median(np.abs(window_data - window_median), axis=0) * self._scaling_factor
            window_mad = np.where(window_mad == 0, np.finfo(float).eps, window_mad)
            
            # Модифицированный Z-score для текущей точки
            modified_z_score = 0.6745 * (X[i] - window_median) / window_mad
            
            # Для многомерных данных
            if isinstance(modified_z_score, np.ndarray) and len(modified_z_score) > 1:
                modified_z_score = np.max(np.abs(modified_z_score))
            
            modified_z_scores[i] = modified_z_score
            anomaly_labels[i] = int(abs(modified_z_score) > self.config.threshold)
        
        return anomaly_labels, modified_z_scores
    
    def _auto_tune_threshold(self, X: np.ndarray) -> None:
        """Автоматическая настройка порога на основе данных."""
        # Вычисляем все модифицированные Z-scores для обучающих данных
        modified_z_scores = self._calculate_modified_zscore(X)
        
        # Используем процентили для определения порога
        percentile = (1 - self.config.contamination) * 100
        threshold = np.percentile(np.abs(modified_z_scores), percentile)
        
        # Минимальный порог = 2.5 (для MAD обычно чуть меньше чем для Z-score)
        self.config.threshold = max(threshold, 2.5)
        
        logger.info(
            "Auto-tuned MAD threshold",
            old_threshold=3.5,
            new_threshold=self.config.threshold,
            contamination=self.config.contamination
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистик детектора."""
        if not self.fitted:
            return {"status": "not_fitted"}
        
        return {
            "fitted": True,
            "threshold": self.config.threshold,
            "bilateral": self.config.bilateral,
            "min_samples": self.config.min_samples,
            "consistency_constant": self.config.consistency_constant,
            "median": self._median.tolist() if isinstance(self._median, np.ndarray) else self._median,
            "mad": self._mad.tolist() if isinstance(self._mad, np.ndarray) else self._mad
        }
    
    def calculate_robustness_score(self, X: np.ndarray) -> float:
        """
        Вычисление оценки робастности детектора для данных.
        
        Args:
            X: Данные для анализа
            
        Returns:
            float: Оценка робастности (0-1, где 1 = максимально робастно)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted")
        
        X = self._validate_input(X)
        
        # Сравниваем MAD с стандартным отклонением
        std_dev = np.std(X, axis=0, ddof=1)
        mad_scaled = self._mad / self._scaling_factor
        
        # Коэффициент робастности: чем ближе MAD к std, тем менее выражены выбросы
        robustness_ratio = np.mean(mad_scaled / (std_dev + np.finfo(float).eps))
        
        # Нормализуем в диапазон [0, 1]
        robustness_score = min(robustness_ratio, 1.0)
        
        return float(robustness_score)

# Пример использования для криптотрейдинга
def create_crypto_mad_detector(
    price_data: pd.DataFrame,
    threshold: float = 3.0,
    window_size: Optional[int] = None
) -> MADDetector:
    """
    Создание MAD детектора оптимизированного для криптоданных.
    
    Args:
        price_data: DataFrame с ценовыми данными
        threshold: Порог для определения аномалий
        window_size: Размер окна для скользящего анализа
        
    Returns:
        Настроенный MADDetector
    """
    config = MADConfig(
        threshold=threshold,
        window_size=window_size,
        rolling_window=window_size is not None,
        auto_threshold=True,
        contamination=0.05,  # 5% аномалий в крипто
        bilateral=True
    )
    
    detector = MADDetector(config)
    
    # Используем цену и объем для обучения
    features = ['close', 'volume']
    if 'returns' in price_data.columns:
        features.append('returns')
    
    detector.fit(price_data[features].values)
    
    return detector