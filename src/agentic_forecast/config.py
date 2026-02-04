from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    time_col: str = "date"
    target_col: str = "y"
    id_cols: List[str] = field(default_factory=lambda: ["item_id", "store_id"])
    horizon: int = 14
    freq: str = "D"
    min_train_points: int = 60
    expected_columns: Optional[List[str]] = None


@dataclass
class DecisionConfig:
    capacity: float = 1_000.0
    budget: float = 10_000.0
    service_level: float = 0.9
    stockout_cost: float = 5.0
    holding_cost: float = 1.0
    unit_cost: float = 2.0


@dataclass
class BacktestConfig:
    splits: int = 3
    step_size: int = 14
    val_size: int = 14


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

