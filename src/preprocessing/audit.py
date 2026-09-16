import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DIAGNOSTIC_EXACT_COLUMNS = {"DSM_IV_TR"}
DIAGNOSTIC_PREFIXES = ("ADI_R_", "ADOS_", "SRS_", "SCQ_", "AQ_")


def get_diagnostic_instrument_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that represent diagnosis-linked behavioral assessment instruments."""
    instrument_columns = []

    for col in df.columns:
        if col in DIAGNOSTIC_EXACT_COLUMNS:
            instrument_columns.append(col)
        elif any(col.startswith(prefix) for prefix in DIAGNOSTIC_PREFIXES):
            instrument_columns.append(col)

    return sorted(set(instrument_columns))


def log_data_leakage_audit(
    df: pd.DataFrame,
    final_feature_list: Iterable[str] | None = None,
    output_path: str | Path | None = None,
    dropped_columns: Iterable[str] | None = None,
) -> dict:
    """Create a formal leakage audit record and optionally save it as JSON."""
    diagnostic_columns = get_diagnostic_instrument_columns(df)
    dropped = list(dropped_columns) if dropped_columns is not None else diagnostic_columns

    audit_record = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "diagnostic_test_instruments": diagnostic_columns,
        "dropped_columns": dropped,
        "final_feature_list": list(final_feature_list) if final_feature_list is not None else [],
        "rationale": {
            "diagnostic_test_instruments": (
                "Explicitly removed because they are diagnosis-linked assessment instruments "
                "with structural missingness, post-diagnostic content, or direct measurement of ASD traits. "
                "Keeping them leaks the target label or creates outcome-dependent features."
            ),
            "structural_missingness": (
                "These instruments are not consistently administered across all participants; in the ABIDE I "
                "phenotype table they show large blocks of missing values in control rows or diagnostic-specific "
                "subsets, making them invalid for general predictive modeling."
            ),
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(audit_record, f, indent=2)

    return audit_record
