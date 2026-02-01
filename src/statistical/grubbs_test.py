"""
🧪 Grubbs' Test Anomaly Detector

Implements Grubbs' test for detecting outliers in univariate normal distributions.
This is a statistical test that identifies single outliers in a dataset.

Test statistic: G = max|Xi - X̄| / s
Where X̄ is the sample mean and s is the sample standard deviation.

Context7 Features:
- Statistical significance testing
- Single outlier detection
- Normal distribution assumption
- Iterative outlier removal
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
class GrubbsConfig:
    """Configuration for Grubbs test detector."""
    alpha: float = 0.05  # Уровень значимости
    max_outliers: Optional[int] = None  # Максимальное количество выбросов
    iterative: bool = True  # Итеративное удаление выбросов
    two_sided: bool = True  # Двусторонний тест
    min_samples: int = 7  # Минимум для статистической значимости
    normality_check: bool = True  # Проверка нормальности
    normality_alpha: float = 0.01  # Уровень значимости для теста нормальности

class GrubbsTestDetector:
    """
    Grubbs' Test Anomaly Detector.
    
    Статистический тест для обнаружения выбросов в нормально распределенных данных.
    Подходит для небольших выборок с одним или несколькими выбросами.
    
    Context7 Features:
    - Statistical rigor
    - Hypothesis testing approach
    - Iterative outlier detection
    - Distribution validation
    """
    
    def __init__(self, config: Optional[GrubbsConfig] = None):
        """
        Инициализация Grubbs test детектора.
        
        Args:
            config: Конфигурация детектора
        """
        self.config = config or GrubbsConfig()
        self.fitted = False
        self._mean = None
        self._std = None
        self._critical_values = {}
        self._normality_pvalue = None
        
        logger.info(
            "GrubbsTestDetector initialized",
            alpha=self.config.alpha,
            two_sided=self.config.two_sided,
            iterative=self.config.iterative
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> 'GrubbsTestDetector':
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
                    f"Insufficient samples for Grubbs test: {len(X)} < {self.config.min_samples}"
                )
            
            # Для многомерных данных применяем к каждому признаку отдельно
            if X.shape[1] > 1:
                logger.warning(
                    "Grubbs test is univariate, applying to each feature separately"
                )
            
            # Проверяем нормальность распределения
            if self.config.normality_check:
                self._check_normality(X)
            
            # Вычисляем статистики
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0, ddof=1)
            
            # Предвычисляем критические значения для различных размеров выборки
            self._precompute_critical_values()
            
            self.fitted = True
            
            logger.info(
                "GrubbsTestDetector fitted successfully",
                n_samples=len(X),
                n_features=X.shape[1] if X.ndim > 1 else 1,
                normality_pvalue=self._normality_pvalue,
                mean=self._mean.tolist() if isinstance(self._mean, np.ndarray) else self._mean,
                std=self._std.tolist() if isinstance(self._std, np.ndarray) else self._std
            )
            
            return self
            
        except Exception as e:
            logger.error("Failed to fit GrubbsTestDetector", error=str(e))
            raise
    
    def detect(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обнаружение аномалий с помощью теста Граббса.
        
        Args:
            X: Данные для анализа
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_labels, test_statistics)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detecting anomalies")
        
        try:
            X = self._validate_input(X)
            
            if X.shape[1] == 1:
                # Одномерный случай
                anomaly_labels, test_stats = self._detect_univariate(X.flatten())
            else:
                # Многомерный случай - применяем к каждому признаку
                anomaly_labels, test_stats = self._detect_multivariate(X)
            
            logger.debug(
                "Grubbs test completed",
                n_samples=len(X),
                n_anomalies=np.sum(anomaly_labels),
                anomaly_rate=f"{np.mean(anomaly_labels):.3%}",
                max_test_stat=np.max(test_stats)
            )
            
            return anomaly_labels, test_stats
            
        except Exception as e:
            logger.error("Failed to detect anomalies with Grubbs test", error=str(e))
            raise
    
    def detect_single_outlier(self, X: np.ndarray) -> Tuple[bool, int, float, float]:
        """
        Обнаружение одного выброса с помощью теста Граббса.
        
        Args:
            X: Одномерный массив данных
            
        Returns:
            Tuple[bool, int, float, float]: (is_outlier, outlier_index, test_statistic, p_value)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted")
        
        X = X.flatten()
        n = len(X)
        
        if n < self.config.min_samples:
            return False, -1, 0.0, 1.0
        
        # Вычисляем статистику теста
        mean = np.mean(X)
        std = np.std(X, ddof=1)
        
        if std == 0:
            return False, -1, 0.0, 1.0
        
        # G = max|Xi - X̄| / s
        deviations = np.abs(X - mean)
        max_deviation_idx = np.argmax(deviations)
        test_statistic = deviations[max_deviation_idx] / std
        
        # Критическое значение
        critical_value = self._get_critical_value(n)
        
        # P-значение (приблизительное)
        t_stat = test_statistic * np.sqrt((n-2) / (n - 1 - test_statistic**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
        
        is_outlier = test_statistic > critical_value
        
        if is_outlier:
            logger.debug(
                "Grubbs test outlier detected",
                index=max_deviation_idx,
                value=X[max_deviation_idx],
                test_statistic=test_statistic,
                critical_value=critical_value,
                p_value=p_value
            )
        
        return is_outlier, max_deviation_idx, test_statistic, p_value
    
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
    
    def _detect_univariate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Обнаружение выбросов в одномерных данных."""
        X_work = X.copy()
        n_original = len(X)
        anomaly_labels = np.zeros(n_original, dtype=int)
        test_statistics = np.zeros(n_original)
        
        # Создаем маппинг индексов
        remaining_indices = np.arange(n_original)
        
        max_outliers = self.config.max_outliers or len(X) // 10
        outliers_found = 0
        
        while len(X_work) >= self.config.min_samples and outliers_found < max_outliers:
            is_outlier, local_idx, test_stat, p_value = self.detect_single_outlier(X_work)
            
            if not is_outlier or not self.config.iterative:
                break
            
            # Находим глобальный индекс
            global_idx = remaining_indices[local_idx]
            
            # Записываем результат
            anomaly_labels[global_idx] = 1
            test_statistics[global_idx] = test_stat
            
            # Удаляем выброс для следующей итерации
            X_work = np.delete(X_work, local_idx)
            remaining_indices = np.delete(remaining_indices, local_idx)
            outliers_found += 1
            
            if not self.config.iterative:
                break
        
        # Для оставшихся точек вычисляем финальные статистики
        if len(X_work) >= self.config.min_samples:
            mean_final = np.mean(X_work)
            std_final = np.std(X_work, ddof=1)
            
            for idx in remaining_indices:
                if std_final > 0:
                    test_statistics[idx] = abs(X[idx] - mean_final) / std_final
        
        return anomaly_labels, test_statistics
    
    def _detect_multivariate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Обнаружение выбросов в многомерных данных (по каждому признаку отдельно)."""
        n_samples, n_features = X.shape
        anomaly_labels = np.zeros(n_samples, dtype=int)
        test_statistics = np.zeros(n_samples)
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            feature_labels, feature_stats = self._detect_univariate(feature_data)
            
            # Объединяем результаты (аномалия если хотя бы в одном признаке)
            anomaly_labels = np.maximum(anomaly_labels, feature_labels)
            test_statistics = np.maximum(test_statistics, feature_stats)
        
        return anomaly_labels, test_statistics
    
    def _check_normality(self, X: np.ndarray) -> None:
        """Проверка нормальности распределения."""
        if X.shape[1] == 1:
            # Одномерный случай
            _, p_value = stats.shapiro(X.flatten())
            self._normality_pvalue = p_value
        else:
            # Многомерный случай - проверяем каждый признак
            p_values = []
            for feature_idx in range(X.shape[1]):
                _, p_value = stats.shapiro(X[:, feature_idx])
                p_values.append(p_value)
            self._normality_pvalue = min(p_values)
        
        if self._normality_pvalue < self.config.normality_alpha:
            logger.warning(
                "Data may not be normally distributed",
                shapiro_p_value=self._normality_pvalue,
                threshold=self.config.normality_alpha,
                recommendation="Consider using non-parametric methods"
            )
    
    def _precompute_critical_values(self) -> None:
        """Предвычисление критических значений для различных размеров выборки."""
        # Критические значения для теста Граббса (приблизительные)
        alpha = self.config.alpha
        
        for n in range(7, 1001):  # От 7 до 1000 наблюдений
            # Формула для критического значения (приближение)
            t_critical = stats.t.ppf(1 - alpha/(2*n), n-2)
            g_critical = ((n-1) * np.sqrt(t_critical**2 / (n-2 + t_critical**2))) / np.sqrt(n)
            self._critical_values[n] = g_critical
    
    def _get_critical_value(self, n: int) -> float:
        """Получение критического значения для размера выборки n."""
        if n in self._critical_values:
            return self._critical_values[n]
        
        # Для больших выборок используем асимптотическую формулу
        if n > 1000:
            # Приблизительное значение для больших n
            z_critical = stats.norm.ppf(1 - self.config.alpha/2)
            return z_critical * np.sqrt((n-1)**2 / (n * (n-2)))
        
        # Вычисляем на лету для нестандартных размеров
        alpha = self.config.alpha
        t_critical = stats.t.ppf(1 - alpha/(2*n), n-2)
        g_critical = ((n-1) * np.sqrt(t_critical**2 / (n-2 + t_critical**2))) / np.sqrt(n)
        
        return g_critical
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистик детектора."""
        if not self.fitted:
            return {"status": "not_fitted"}
        
        return {
            "fitted": True,
            "alpha": self.config.alpha,
            "two_sided": self.config.two_sided,
            "iterative": self.config.iterative,
            "min_samples": self.config.min_samples,
            "normality_pvalue": self._normality_pvalue,
            "mean": self._mean.tolist() if isinstance(self._mean, np.ndarray) else self._mean,
            "std": self._std.tolist() if isinstance(self._std, np.ndarray) else self._std,
            "critical_values_computed": len(self._critical_values)
        }

# Пример использования для криптотрейдинга
def create_crypto_grubbs_detector(
    price_data: pd.DataFrame,
    feature: str = 'returns',
    alpha: float = 0.05
) -> GrubbsTestDetector:
    """
    Создание Grubbs детектора для криптоданных.
    
    Args:
        price_data: DataFrame с ценовыми данными
        feature: Признак для анализа (лучше всего работает с returns)
        alpha: Уровень значимости
        
    Returns:
        Настроенный GrubbsTestDetector
    """
    # Вычисляем доходности если их нет
    if 'returns' not in price_data.columns and feature == 'returns':
        price_data = price_data.copy()
        price_data['returns'] = price_data['close'].pct_change().dropna()
    
    config = GrubbsConfig(
        alpha=alpha,
        iterative=True,
        two_sided=True,
        normality_check=True,
        max_outliers=max(1, len(price_data) // 50)  # Максимум 2% выбросов
    )
    
    detector = GrubbsTestDetector(config)
    
    # Используем только один признак (Grubbs test одномерный)
    if feature in price_data.columns:
        feature_data = price_data[feature].dropna().values.reshape(-1, 1)
        detector.fit(feature_data)
    else:
        raise ValueError(f"Feature '{feature}' not found in price_data")
    
    return detector