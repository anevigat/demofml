"""Gradient-boosted trees trained inside purged temporal folds (Campaign 3).

Deliberately a sibling of `baseline.py` rather than an extension of it.
`baseline.py` is frozen: its contracts were used by every Campaign 1 and
Campaign 2 run, and its `BaselineConfig` rejects any `model_type` other than
`ridge`. Everything here that is genuinely shared — feature/label alignment,
locked-test rejection, fold row selection, the action rule — is imported from
that module instead of being reimplemented, so the two families differ only
where they are supposed to differ.

The hypothesis this exists to test is stated in
`docs/research/campaign-3-lightgbm-causal-v2-hypothesis-v1.md`, and the process
rules it obeys in `docs/research/campaign-3-protocol-v1.md`.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from demofml.features.causal_v2 import FEATURE_SET_V2_ID, FEATURE_V2_COLUMNS
from demofml.labels.executable import LABEL_SET_V2_ID
from demofml.models.baseline import (  # noqa: PLC2701 - see module docstring
    AlignedResearchData,
    _action,
    align_research_tables,
)
from demofml.validation.splits import (
    CAMPAIGN_3_VALIDATION_SET_ID,
    TemporalFold,
    ValidationPlan,
    select_fold_rows,
)

GBM_MODEL_SET_ID = "gbm-lightgbm-v1"
GBM_PREDICTION_SET_ID = "walk-forward-predictions-v5"
GBM_SELECTION_POLICY = "inner-purged-cv-first-fold-v1"
GBM_SELECTION_METRIC = "mean_executable_return_bps"
# Stage B: identical estimator, different action rule. Stage A lost 10.05% at
# 0.61% realized volatility across 192,499 trades, which is cost bleed rather
# than mispriced risk, so Stage B searches over conviction and spread-regime
# gates alongside the model candidates. gbm-lightgbm-v1 has been used by a run
# and is therefore never edited; this is a new contract beside it.
GBM_MODEL_SET_V2_ID = "gbm-lightgbm-v2"
GBM_PREDICTION_SET_V2_ID = "walk-forward-predictions-v6"
GBM_SELECTION_POLICY_V2 = "inner-purged-cv-conviction-gate-v1"
_SPREAD_ZSCORE_FEATURE = "spread_zscore_72"
_OBJECTIVE = "l2"
_MODEL_TYPE = "lightgbm"


@dataclass(frozen=True)
class GBMCandidate:
    """One fully specified point of the sealed hyperparameter search space."""

    id: str
    num_leaves: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int
    colsample_bytree: float
    subsample: float
    subsample_freq: int
    reg_lambda: float

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("candidate id cannot be empty")
        if self.num_leaves < 2 or self.max_depth < 1:
            raise ValueError("candidate tree size must be positive")
        if self.num_leaves > 2**self.max_depth:
            raise ValueError("candidate num_leaves exceeds its depth limit")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("candidate learning_rate must be in (0, 1]")
        if self.n_estimators < 1 or self.min_child_samples < 1:
            raise ValueError("candidate estimator counts must be positive")
        if not 0.0 < self.colsample_bytree <= 1.0 or not 0.0 < self.subsample <= 1.0:
            raise ValueError("candidate sampling fractions must be in (0, 1]")
        if self.subsample < 1.0 and self.subsample_freq < 1:
            raise ValueError("row subsampling requires a positive subsample_freq")
        if self.reg_lambda < 0.0 or not math.isfinite(self.reg_lambda):
            raise ValueError("candidate reg_lambda must be finite and non-negative")

    def parameters(self, config: GBMConfig) -> dict[str, Any]:
        """Return the exact LightGBM parameters this candidate fixes."""
        return {
            "objective": config.objective,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "colsample_bytree": self.colsample_bytree,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "reg_lambda": self.reg_lambda,
            "random_state": config.random_seed,
            "n_jobs": config.num_threads,
            "deterministic": config.deterministic,
            "force_row_wise": True,
            "verbose": -1,
        }


@dataclass(frozen=True)
class GateCandidate:
    """One named point of the sealed action-rule search space.

    `conviction_quantile` is the fraction of decisions the base rule would
    trade that survive the gate, ranked by predicted return. The cut itself is
    recomputed per fold from that fold's training rows, so only the policy is
    frozen across the walk-forward, never a level fitted on validation data.
    `spread_zscore_maximum` is an absolute bound on `spread_zscore_72`, which
    is already standardized; `None` disables the spread gate.
    """

    id: str
    conviction_quantile: float
    spread_zscore_maximum: float | None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("gate id cannot be empty")
        if not 0.0 < self.conviction_quantile <= 1.0:
            raise ValueError("conviction_quantile must be in (0, 1]")
        if self.spread_zscore_maximum is not None and not math.isfinite(
            self.spread_zscore_maximum
        ):
            raise ValueError("spread_zscore_maximum must be finite when set")


@dataclass(frozen=True)
class GBMConfig:
    """Immutable behavior of the Campaign 3 gradient-boosting line."""

    id: str
    feature_set: str
    label_set: str
    validation_set: str
    horizons_minutes: tuple[int, ...]
    training_scope: str
    model_type: str
    objective: str
    selection_policy: str
    selection_metric: str
    inner_folds: int
    inner_purge_minutes: int
    minimum_inner_training_rows: int
    action_threshold_bps: float
    minimum_training_rows: int
    random_seed: int
    num_threads: int
    deterministic: bool
    locked_test_policy: str
    features: tuple[str, ...]
    candidates: tuple[GBMCandidate, ...]
    gates: tuple[GateCandidate, ...] = ()

    def __post_init__(self) -> None:
        if self.id not in {GBM_MODEL_SET_ID, GBM_MODEL_SET_V2_ID}:
            raise ValueError("model id is not supported")
        if self.feature_set != FEATURE_SET_V2_ID or self.label_set != LABEL_SET_V2_ID:
            raise ValueError("gbm data contracts are incompatible")
        if self.validation_set != CAMPAIGN_3_VALIDATION_SET_ID:
            raise ValueError("gbm requires campaign-3-walk-forward-v1")
        if (
            not self.horizons_minutes
            or tuple(sorted(set(self.horizons_minutes))) != self.horizons_minutes
        ):
            raise ValueError("horizons must be unique and increasing")
        if self.training_scope != "per_symbol":
            raise ValueError("gbm training_scope must be per_symbol")
        if self.model_type != _MODEL_TYPE or self.objective != _OBJECTIVE:
            raise ValueError("gbm model must be lightgbm with the l2 objective")
        expected_policy = (
            GBM_SELECTION_POLICY
            if self.id == GBM_MODEL_SET_ID
            else GBM_SELECTION_POLICY_V2
        )
        if self.selection_policy != expected_policy:
            raise ValueError("gbm selection policy does not match model id")
        if self.selection_metric != GBM_SELECTION_METRIC:
            raise ValueError("gbm selection metric is not supported")
        if not 3 <= self.inner_folds <= 5:
            raise ValueError("inner_folds must be between three and five")
        # The inner boundary carries the same label window as the outer one:
        # an inner training row must not resolve into its inner validation
        # block, exactly as validation/splits.py enforces for real folds.
        if self.inner_purge_minutes < max(self.horizons_minutes) + 5:
            raise ValueError("inner purge must cover the full label window")
        if self.minimum_inner_training_rows < 2 or self.minimum_training_rows < 2:
            raise ValueError("training row minima must be at least two")
        if not math.isfinite(self.action_threshold_bps):
            raise ValueError("action_threshold_bps must be finite")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        # num_threads is part of the contract because LightGBM only reproduces
        # bit-identical trees for a fixed thread count; changing it changes the
        # model and therefore requires a new model id.
        if self.num_threads < 1 or not self.deterministic:
            raise ValueError("gbm requires deterministic fitting on fixed threads")
        if self.locked_test_policy != "forbidden":
            raise ValueError("locked test policy must remain forbidden")
        if self.features != FEATURE_V2_COLUMNS:
            raise ValueError("gbm features do not match causal-v2")
        if len(self.candidates) < 2:
            raise ValueError("a search space needs at least two candidates")
        identifiers = [candidate.id for candidate in self.candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate ids must be unique")
        if self.id == GBM_MODEL_SET_ID:
            if self.gates:
                raise ValueError("gbm-lightgbm-v1 cannot define action gates")
        else:
            if len(self.gates) < 2:
                raise ValueError("a gate search space needs at least two gates")
            gate_ids = [gate.id for gate in self.gates]
            if len(set(gate_ids)) != len(gate_ids):
                raise ValueError("gate ids must be unique")
            if _SPREAD_ZSCORE_FEATURE not in self.features:
                raise ValueError(f"gates require the {_SPREAD_ZSCORE_FEATURE} feature")

    @property
    def action_threshold(self) -> float:
        return self.action_threshold_bps / 10_000.0

    @property
    def prediction_set(self) -> str:
        """Return the prediction contract emitted by this model version."""
        if self.id == GBM_MODEL_SET_ID:
            return GBM_PREDICTION_SET_ID
        return GBM_PREDICTION_SET_V2_ID

    @property
    def spread_zscore_position(self) -> int:
        """Column offset of the spread z-score inside the feature matrix."""
        return self.features.index(_SPREAD_ZSCORE_FEATURE)

    def gate(self, gate_id: str) -> GateCandidate:
        """Return one sealed action gate by id."""
        for gate in self.gates:
            if gate.id == gate_id:
                return gate
        raise KeyError(f"unknown gate: {gate_id}")

    def candidate(self, candidate_id: str) -> GBMCandidate:
        """Return one sealed candidate by id."""
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate
        raise KeyError(f"unknown candidate: {candidate_id}")


def load_gbm_config(path: Path) -> GBMConfig:
    """Load and strictly validate the versioned gradient-boosting definition."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"GBM config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        candidates = tuple(
            GBMCandidate(
                id=str(entry["id"]),
                num_leaves=int(entry["num_leaves"]),
                max_depth=int(entry["max_depth"]),
                learning_rate=float(entry["learning_rate"]),
                n_estimators=int(entry["n_estimators"]),
                min_child_samples=int(entry["min_child_samples"]),
                colsample_bytree=float(entry["colsample_bytree"]),
                subsample=float(entry["subsample"]),
                subsample_freq=int(entry["subsample_freq"]),
                reg_lambda=float(entry["reg_lambda"]),
            )
            for entry in values["candidates"]
        )
        return GBMConfig(
            id=str(values["id"]),
            feature_set=str(values["feature_set"]),
            label_set=str(values["label_set"]),
            validation_set=str(values["validation_set"]),
            horizons_minutes=tuple(int(value) for value in values["horizons_minutes"]),
            training_scope=str(values["training_scope"]),
            model_type=str(values["model_type"]),
            objective=str(values["objective"]),
            selection_policy=str(values["selection_policy"]),
            selection_metric=str(values["selection_metric"]),
            inner_folds=int(values["inner_folds"]),
            inner_purge_minutes=int(values["inner_purge_minutes"]),
            minimum_inner_training_rows=int(values["minimum_inner_training_rows"]),
            action_threshold_bps=float(values["action_threshold_bps"]),
            minimum_training_rows=int(values["minimum_training_rows"]),
            random_seed=int(values["random_seed"]),
            num_threads=int(values["num_threads"]),
            deterministic=bool(values["deterministic"]),
            locked_test_policy=str(values["locked_test_policy"]),
            features=tuple(str(value) for value in values["features"]),
            candidates=candidates,
            gates=tuple(
                GateCandidate(
                    id=str(entry["id"]),
                    conviction_quantile=float(entry["conviction_quantile"]),
                    spread_zscore_maximum=(
                        float(entry["spread_zscore_maximum"])
                        if entry.get("spread_zscore_maximum") is not None
                        else None
                    ),
                )
                for entry in values.get("gates", [])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid gbm config field: {error}") from error


def prediction_schema(config: GBMConfig) -> pa.Schema:
    """Build the prediction schema with complete model provenance."""
    gate_fields = (
        [
            pa.field("selected_gate", pa.string(), nullable=False),
            pa.field("conviction_quantile", pa.float64(), nullable=False),
            pa.field("conviction_threshold", pa.float64(), nullable=False),
        ]
        if config.gates
        else []
    )
    gate_metadata = (
        {
            b"demofml.gates": ",".join(gate.id for gate in config.gates).encode(),
            b"demofml.gate_calibration": b"per_fold_training_quantile",
        }
        if config.gates
        else {}
    )
    return pa.schema(
        [
            pa.field("model_set", pa.string(), nullable=False),
            pa.field("validation_set", pa.string(), nullable=False),
            pa.field("fold_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("decision_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("entry_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("exit_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("horizon_minutes", pa.int16(), nullable=False),
            pa.field("predicted_long_return", pa.float64(), nullable=False),
            pa.field("predicted_short_return", pa.float64(), nullable=False),
            pa.field("selected_candidate", pa.string(), nullable=False),
            *gate_fields,
            pa.field("action", pa.string(), nullable=False),
            pa.field("realized_return", pa.float64(), nullable=False),
        ],
        metadata={
            b"demofml.prediction_set": config.prediction_set.encode(),
            b"demofml.model_set": config.id.encode(),
            b"demofml.feature_set": config.feature_set.encode(),
            b"demofml.label_set": config.label_set.encode(),
            b"demofml.validation_set": config.validation_set.encode(),
            b"demofml.horizons_minutes": ",".join(
                str(value) for value in config.horizons_minutes
            ).encode(),
            b"demofml.action_threshold_bps": str(config.action_threshold_bps).encode(),
            b"demofml.random_seed": str(config.random_seed).encode(),
            b"demofml.num_threads": str(config.num_threads).encode(),
            b"demofml.selection_policy": config.selection_policy.encode(),
            b"demofml.selection_metric": config.selection_metric.encode(),
            b"demofml.inner_folds": str(config.inner_folds).encode(),
            b"demofml.inner_purge_minutes": str(config.inner_purge_minutes).encode(),
            b"demofml.candidates": ",".join(
                candidate.id for candidate in config.candidates
            ).encode(),
            **gate_metadata,
        },
    )


def _resolved(
    indices: NDArray[np.int64],
    long_targets: NDArray[np.float64],
    short_targets: NDArray[np.float64],
) -> NDArray[np.int64]:
    mask = np.isfinite(long_targets[indices]) & np.isfinite(short_targets[indices])
    return indices[mask]


def _fit_predict(
    candidate: GBMCandidate,
    config: GBMConfig,
    training_features: NDArray[np.float64],
    long_targets: NDArray[np.float64],
    short_targets: NDArray[np.float64],
    scoring_features: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit one booster per direction and score them on the same rows.

    Two independent single-output boosters rather than one multi-output model:
    LightGBM has no native multi-output regression, and the long and short legs
    have genuinely different cost structures at the same instant.
    """
    parameters = candidate.parameters(config)
    predictions: list[NDArray[np.float64]] = []
    for targets in (long_targets, short_targets):
        model = LGBMRegressor(**parameters)
        model.fit(training_features, targets)
        predicted = np.asarray(model.predict(scoring_features), dtype=float)
        if predicted.ndim != 1 or predicted.size != scoring_features.shape[0]:
            raise RuntimeError("lightgbm produced an invalid prediction vector")
        if not np.isfinite(predicted).all():
            raise RuntimeError("lightgbm produced a non-finite prediction")
        predictions.append(predicted)
    return predictions[0], predictions[1]


def _executable_return_bps(
    long_predictions: NDArray[np.float64],
    short_predictions: NDArray[np.float64],
    long_targets: NDArray[np.float64],
    short_targets: NDArray[np.float64],
    threshold: float,
) -> float:
    """Mean realized return of the action rule, in basis points."""
    take_long = long_predictions > short_predictions
    best = np.maximum(long_predictions, short_predictions)
    realized = np.where(take_long, long_targets, short_targets)
    traded = np.where(best > threshold, realized, 0.0)
    return float(np.mean(traded)) * 10_000.0


def _inner_blocks(
    indices: NDArray[np.int64], inner_folds: int
) -> list[NDArray[np.int64]]:
    """Split ordered training positions into contiguous equal-time-order blocks."""
    return [
        block
        for block in np.array_split(indices, inner_folds + 1)
        if block.size > 0
    ]


def select_candidate(
    data: AlignedResearchData,
    fold: TemporalFold,
    horizon: int,
    config: GBMConfig,
) -> str:
    """Choose one candidate using only rows inside `fold`'s training window.

    Expanding inner cross-validation with the same label-window purge as the
    outer split, so nothing observed here can have resolved into the block it
    is scored on. Ties break toward the earlier candidate in the sealed search
    space, which makes the choice a deterministic function of the contract.
    """
    long_targets = data.long_targets[horizon]
    short_targets = data.short_targets[horizon]
    selection = select_fold_rows(data.decision_times, fold)
    training_indices = _resolved(
        np.asarray(selection.train, dtype=np.int64), long_targets, short_targets
    )
    blocks = _inner_blocks(training_indices, config.inner_folds)
    if len(blocks) < 2:
        raise ValueError("inner cross-validation needs at least two blocks")
    purge = timedelta(minutes=config.inner_purge_minutes)
    scores: dict[str, list[float]] = {
        candidate.id: [] for candidate in config.candidates
    }
    for position in range(1, len(blocks)):
        inner_validation = blocks[position]
        inner_start = data.decision_times[int(inner_validation[0])]
        fit_end = inner_start - purge
        inner_training = np.concatenate(blocks[:position])
        keep = np.fromiter(
            (data.decision_times[int(index)] < fit_end for index in inner_training),
            dtype=bool,
            count=inner_training.size,
        )
        inner_training = inner_training[keep]
        if inner_training.size < config.minimum_inner_training_rows:
            continue
        for candidate in config.candidates:
            long_predictions, short_predictions = _fit_predict(
                candidate,
                config,
                data.features[inner_training],
                long_targets[inner_training],
                short_targets[inner_training],
                data.features[inner_validation],
            )
            scores[candidate.id].append(
                _executable_return_bps(
                    long_predictions,
                    short_predictions,
                    long_targets[inner_validation],
                    short_targets[inner_validation],
                    config.action_threshold,
                )
            )
    scored = [
        (candidate.id, scores[candidate.id])
        for candidate in config.candidates
        if scores[candidate.id]
    ]
    if not scored:
        raise ValueError(
            f"fold {fold.id} horizon {horizon} has no usable inner training rows"
        )
    best_id, best_score = scored[0][0], float(np.mean(scored[0][1]))
    for candidate_id, values in scored[1:]:
        mean_score = float(np.mean(values))
        if mean_score > best_score:
            best_id, best_score = candidate_id, mean_score
    return best_id


def _conviction_threshold(
    training_scores: NDArray[np.float64], gate: GateCandidate, floor: float
) -> float:
    """Return the score cut for one gate, computed from training rows only.

    The cut is a quantile of the training-window score distribution restricted
    to decisions the base rule would already take, so `conviction_quantile=1.0`
    reproduces the ungated rule exactly. Scores here are in-sample, which makes
    them more dispersed than out-of-fold scores; the gate is relative rather
    than absolute precisely so that shift moves the cut rather than the policy.
    """
    tradeable = training_scores[training_scores > floor]
    if tradeable.size == 0:
        return float("inf")
    return float(np.quantile(tradeable, 1.0 - gate.conviction_quantile))


def _gate_admits(
    scores: NDArray[np.float64],
    spread_zscores: NDArray[np.float64],
    gate: GateCandidate,
    threshold: float,
    floor: float,
) -> NDArray[np.bool_]:
    """Rows the gate allows to trade. A missing spread z-score never trades."""
    admitted = (scores > floor) & (scores >= threshold)
    if gate.spread_zscore_maximum is not None:
        admitted &= np.isfinite(spread_zscores) & (
            spread_zscores <= gate.spread_zscore_maximum
        )
    return admitted


def _gated_return_bps(
    long_predictions: NDArray[np.float64],
    short_predictions: NDArray[np.float64],
    spread_zscores: NDArray[np.float64],
    long_targets: NDArray[np.float64],
    short_targets: NDArray[np.float64],
    gate: GateCandidate,
    threshold: float,
    floor: float,
) -> float:
    scores = np.maximum(long_predictions, short_predictions)
    admitted = _gate_admits(scores, spread_zscores, gate, threshold, floor)
    realized = np.where(
        long_predictions > short_predictions, long_targets, short_targets
    )
    return float(np.mean(np.where(admitted, realized, 0.0))) * 10_000.0


def _fit_score_both(
    candidate: GBMCandidate,
    config: GBMConfig,
    training_features: NDArray[np.float64],
    long_targets: NDArray[np.float64],
    short_targets: NDArray[np.float64],
    scoring_features: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fit once, then score the training rows and the scoring rows together."""
    combined = np.vstack([training_features, scoring_features])
    long_all, short_all = _fit_predict(
        candidate, config, training_features, long_targets, short_targets, combined
    )
    split = training_features.shape[0]
    return long_all[:split], short_all[:split], long_all[split:], short_all[split:]


def select_candidate_and_gate(
    data: AlignedResearchData,
    fold: TemporalFold,
    horizon: int,
    config: GBMConfig,
) -> tuple[str, str]:
    """Choose a model candidate and action gate inside `fold`'s training window.

    Gates are evaluated on predictions the candidates already produced, so
    widening the action-rule search space costs no additional model fits: the
    inner cross-validation is exactly as expensive as Stage A's.
    """
    long_targets = data.long_targets[horizon]
    short_targets = data.short_targets[horizon]
    spread_column = data.features[:, config.spread_zscore_position]
    floor = config.action_threshold
    selection = select_fold_rows(data.decision_times, fold)
    training_indices = _resolved(
        np.asarray(selection.train, dtype=np.int64), long_targets, short_targets
    )
    blocks = _inner_blocks(training_indices, config.inner_folds)
    if len(blocks) < 2:
        raise ValueError("inner cross-validation needs at least two blocks")
    purge = timedelta(minutes=config.inner_purge_minutes)
    scores: dict[tuple[str, str], list[float]] = {
        (candidate.id, gate.id): []
        for candidate in config.candidates
        for gate in config.gates
    }
    for position in range(1, len(blocks)):
        inner_validation = blocks[position]
        fit_end = data.decision_times[int(inner_validation[0])] - purge
        inner_training = np.concatenate(blocks[:position])
        keep = np.fromiter(
            (data.decision_times[int(index)] < fit_end for index in inner_training),
            dtype=bool,
            count=inner_training.size,
        )
        inner_training = inner_training[keep]
        if inner_training.size < config.minimum_inner_training_rows:
            continue
        for candidate in config.candidates:
            train_long, train_short, valid_long, valid_short = _fit_score_both(
                candidate,
                config,
                data.features[inner_training],
                long_targets[inner_training],
                short_targets[inner_training],
                data.features[inner_validation],
            )
            training_scores = np.maximum(train_long, train_short)
            for gate in config.gates:
                threshold = _conviction_threshold(training_scores, gate, floor)
                scores[(candidate.id, gate.id)].append(
                    _gated_return_bps(
                        valid_long,
                        valid_short,
                        spread_column[inner_validation],
                        long_targets[inner_validation],
                        short_targets[inner_validation],
                        gate,
                        threshold,
                        floor,
                    )
                )
    scored = [
        (key, values)
        for key, values in scores.items()
        if values
    ]
    if not scored:
        raise ValueError(
            f"fold {fold.id} horizon {horizon} has no usable inner training rows"
        )
    ordered = [
        (candidate.id, gate.id)
        for candidate in config.candidates
        for gate in config.gates
    ]
    ranked = sorted(scored, key=lambda item: ordered.index(item[0]))
    best_key, best_score = ranked[0][0], float(np.mean(ranked[0][1]))
    for key, values in ranked[1:]:
        mean_score = float(np.mean(values))
        if mean_score > best_score:
            best_key, best_score = key, mean_score
    return best_key


def _run_gated_walk_forward(
    data: AlignedResearchData,
    plan: ValidationPlan,
    config: GBMConfig,
) -> pa.Table:
    """Stage B: the Stage A estimator behind a sealed, per-fold-calibrated gate."""
    folds = plan.folds()
    selected = {
        horizon: select_candidate_and_gate(data, folds[0], horizon, config)
        for horizon in config.horizons_minutes
    }
    spread_column = data.features[:, config.spread_zscore_position]
    floor = config.action_threshold
    rows: list[dict[str, object]] = []
    for fold in folds:
        selection = select_fold_rows(data.decision_times, fold)
        if not selection.validation:
            raise ValueError(f"fold {fold.id} has no validation rows")
        training_indices = np.asarray(selection.train, dtype=np.int64)
        validation_indices = np.asarray(selection.validation, dtype=np.int64)
        for horizon in config.horizons_minutes:
            long_targets = data.long_targets[horizon]
            short_targets = data.short_targets[horizon]
            usable_training = _resolved(training_indices, long_targets, short_targets)
            usable_validation = _resolved(
                validation_indices, long_targets, short_targets
            )
            if usable_training.size < config.minimum_training_rows:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has insufficient training rows"
                )
            if usable_validation.size == 0:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has no resolved "
                    "validation labels"
                )
            candidate_id, gate_id = selected[horizon]
            gate = config.gate(gate_id)
            train_long, train_short, valid_long, valid_short = _fit_score_both(
                config.candidate(candidate_id),
                config,
                data.features[usable_training],
                long_targets[usable_training],
                short_targets[usable_training],
                data.features[usable_validation],
            )
            threshold = _conviction_threshold(
                np.maximum(train_long, train_short), gate, floor
            )
            validation_scores = np.maximum(valid_long, valid_short)
            admitted = _gate_admits(
                validation_scores,
                spread_column[usable_validation],
                gate,
                threshold,
                floor,
            )
            for position, row_index in enumerate(usable_validation):
                predicted_long = float(valid_long[position])
                predicted_short = float(valid_short[position])
                action = (
                    ("long" if predicted_long > predicted_short else "short")
                    if bool(admitted[position])
                    else "flat"
                )
                entry_time = data.entry_times[row_index]
                exit_time = data.exit_times[horizon][row_index]
                decision_time = data.decision_times[row_index]
                if (
                    not isinstance(entry_time, datetime)
                    or not isinstance(exit_time, datetime)
                    or not decision_time <= entry_time < exit_time
                ):
                    raise RuntimeError("resolved label execution times are invalid")
                realized = (
                    float(long_targets[row_index])
                    if action == "long"
                    else float(short_targets[row_index])
                    if action == "short"
                    else 0.0
                )
                rows.append(
                    {
                        "model_set": config.id,
                        "validation_set": plan.id,
                        "fold_id": fold.id,
                        "symbol": data.symbol,
                        "decision_time": decision_time,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "horizon_minutes": horizon,
                        "predicted_long_return": predicted_long,
                        "predicted_short_return": predicted_short,
                        "selected_candidate": candidate_id,
                        "selected_gate": gate_id,
                        "conviction_quantile": gate.conviction_quantile,
                        "conviction_threshold": (
                            0.0 if not math.isfinite(threshold) else threshold
                        ),
                        "action": action,
                        "realized_return": realized,
                    }
                )
    return pa.Table.from_pylist(rows, schema=prediction_schema(config))


def run_walk_forward_gbm(
    features: pa.Table,
    labels: pa.Table,
    plan: ValidationPlan,
    config: GBMConfig,
) -> pa.Table:
    """Train and score every development fold without touching the lock.

    The candidate is chosen once per horizon, on the first fold's training
    window, and then frozen for the whole walk-forward. That window precedes
    every validation fold in an expanding plan, so the choice satisfies the
    protocol's rule that selection happens only inside a training window, while
    keeping the run to one fit per fold instead of one per candidate per fold.
    """
    data = align_research_tables(features, labels, plan, config)
    if config.gates:
        return _run_gated_walk_forward(data, plan, config)
    folds = plan.folds()
    selected = {
        horizon: select_candidate(data, folds[0], horizon, config)
        for horizon in config.horizons_minutes
    }
    rows: list[dict[str, object]] = []
    for fold in folds:
        selection = select_fold_rows(data.decision_times, fold)
        if not selection.validation:
            raise ValueError(f"fold {fold.id} has no validation rows")
        training_indices = np.asarray(selection.train, dtype=np.int64)
        validation_indices = np.asarray(selection.validation, dtype=np.int64)
        for horizon in config.horizons_minutes:
            long_targets = data.long_targets[horizon]
            short_targets = data.short_targets[horizon]
            usable_training = _resolved(
                training_indices, long_targets, short_targets
            )
            usable_validation = _resolved(
                validation_indices, long_targets, short_targets
            )
            if usable_training.size < config.minimum_training_rows:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has insufficient training rows"
                )
            if usable_validation.size == 0:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has no resolved "
                    "validation labels"
                )
            candidate_id = selected[horizon]
            long_predictions, short_predictions = _fit_predict(
                config.candidate(candidate_id),
                config,
                data.features[usable_training],
                long_targets[usable_training],
                short_targets[usable_training],
                data.features[usable_validation],
            )
            for position, row_index in enumerate(usable_validation):
                predicted_long = float(long_predictions[position])
                predicted_short = float(short_predictions[position])
                action = _action(
                    predicted_long, predicted_short, config.action_threshold
                )
                entry_time = data.entry_times[row_index]
                exit_time = data.exit_times[horizon][row_index]
                decision_time = data.decision_times[row_index]
                if (
                    not isinstance(entry_time, datetime)
                    or not isinstance(exit_time, datetime)
                    or not decision_time <= entry_time < exit_time
                ):
                    raise RuntimeError("resolved label execution times are invalid")
                realized = (
                    float(long_targets[row_index])
                    if action == "long"
                    else float(short_targets[row_index])
                    if action == "short"
                    else 0.0
                )
                rows.append(
                    {
                        "model_set": config.id,
                        "validation_set": plan.id,
                        "fold_id": fold.id,
                        "symbol": data.symbol,
                        "decision_time": decision_time,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "horizon_minutes": horizon,
                        "predicted_long_return": predicted_long,
                        "predicted_short_return": predicted_short,
                        "selected_candidate": candidate_id,
                        "action": action,
                        "realized_return": realized,
                    }
                )
    return pa.Table.from_pylist(rows, schema=prediction_schema(config))
