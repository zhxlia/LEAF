#!/usr/bin/env python3
"""Download and prepare the 16 paper benchmarks; targets are the last CSV column."""
from __future__ import annotations
import argparse
import importlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
FEATLLM_METADATA_BASE = 'https://raw.githubusercontent.com/Sungwon-Han/FeatLLM/main/data'
FEATLLM_METADATA_ALIAS = {'adult': 'adult', 'blood': 'blood', 'credit-g': 'credit-g', 'heart': 'heart'}

MANUAL_METADATA: dict[str, dict[str, str]] = {'wine': {'fixed acidity': 'fixed acidity in the wine', 'volatile acidity': 'volatile acidity in the wine', 'citric acid': 'citric acid concentration in the wine', 'residual sugar': 'residual sugar remaining in the wine', 'chlorides': 'chloride concentration in the wine', 'free sulfur dioxide': 'free sulfur dioxide concentration in the wine', 'total sulfur dioxide': 'total sulfur dioxide concentration in the wine', 'density': 'density of the wine', 'pH': 'pH value of the wine', 'sulphates': 'sulphate concentration in the wine', 'alcohol': 'alcohol percentage of the wine'}, 'insurance': {'age': 'age of the insurance beneficiary', 'sex': 'sex of the insurance beneficiary', 'bmi': 'body mass index of the insurance beneficiary', 'children': 'number of children covered by the insurance plan', 'smoker': 'whether the insurance beneficiary is a smoker', 'region': 'residential region of the insurance beneficiary'}, 'cdc_diabetes': {'HighBP': 'whether the person has high blood pressure', 'HighChol': 'whether the person has high cholesterol', 'CholCheck': 'whether the person had a cholesterol check in the past five years', 'BMI': 'body mass index', 'Smoker': 'whether the person has smoked at least 100 cigarettes in their lifetime', 'Stroke': 'whether the person has ever had a stroke', 'HeartDiseaseorAttack': 'whether the person has coronary heart disease or myocardial infarction history', 'PhysActivity': 'whether the person reported physical activity in the past 30 days', 'Fruits': 'whether the person consumes fruit at least once per day', 'Veggies': 'whether the person consumes vegetables at least once per day', 'HvyAlcoholConsump': 'whether the person reports heavy alcohol consumption', 'AnyHealthcare': 'whether the person has any healthcare coverage', 'NoDocbcCost': 'whether cost prevented the person from seeing a doctor', 'GenHlth': 'self-reported general health status', 'MentHlth': 'number of days of poor mental health in the past 30 days', 'PhysHlth': 'number of days of poor physical health in the past 30 days', 'DiffWalk': 'whether the person has serious difficulty walking or climbing stairs', 'Sex': 'sex of the person', 'Age': 'age category of the person', 'Education': 'education level of the person', 'Income': 'income category of the person'}}

def require_package(pkg: str, import_name: Optional[str]=None) -> None:
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError as e:
        raise SystemExit(f"Missing package '{pkg}'. Please install it first, e.g. pip install {pkg}") from e

def to_series(y: object, fallback_name: str='target') -> pd.Series:
    if isinstance(y, pd.Series):
        s = y.copy()
        s.name = s.name or fallback_name
        return s
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f'Expected exactly 1 target column, but got {list(y.columns)}. Use a dataset-specific handler instead.')
        s = y.iloc[:, 0].copy()
        s.name = y.columns[0] or fallback_name
        return s
    s = pd.Series(y, name=fallback_name)
    return s

def sanitize_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype('string').str.strip()
    return out

def coerce_numeric_like(s: pd.Series) -> pd.Series:
    """Best-effort conversion for numeric-looking string columns.

    Handles common locale cases such as decimal commas (e.g. '12,34') and
    thousands commas (e.g. '1,234.56'). Non-convertible values become NaN.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    x = s.astype('string').str.strip()
    x = x.str.replace('\xa0', '', regex=False)
    x = x.str.replace(' ', '', regex=False)
    comma_decimal = x.str.contains(',', na=False) & ~x.str.contains('\\.', na=False)
    x = x.where(~comma_decimal, x.str.replace(',', '.', regex=False))
    thousands_commas = x.str.contains(',', na=False) & x.str.contains('\\.', na=False)
    x = x.where(~thousands_commas, x.str.replace(',', '', regex=False))
    return pd.to_numeric(x, errors='coerce')

def maybe_parse_numeric_columns(df: pd.DataFrame, threshold: float=0.95) -> pd.DataFrame:
    """Convert object/string columns to numeric when almost all non-null values parse."""
    out = df.copy()
    for col in out.columns:
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue
        non_null = out[col].notna().sum()
        if non_null == 0:
            continue
        parsed = coerce_numeric_like(out[col])
        success_ratio = parsed.notna().sum() / non_null
        if success_ratio >= threshold:
            out[col] = parsed
    return out

def save_featllm_csv(df: pd.DataFrame, target_col: str, out_path: Path) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in columns: {list(df.columns)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[ordered_cols].copy()
    df = maybe_parse_numeric_columns(df)
    df = sanitize_strings(df)
    df.to_csv(out_path, index=False)
    return df

def maybe_skip(out_path: Path, force: bool) -> bool:
    if out_path.exists() and (not force):
        print(f'[skip] {out_path.name} already exists')
        return True
    return False

def metadata_path(name: str, out_dir: Path) -> Path:
    return out_dir / f'{name}-metadata.json'

def maybe_skip_metadata(meta_path: Path, force_metadata: bool) -> bool:
    if meta_path.exists() and (not force_metadata):
        print(f'[skip] {meta_path.name} already exists')
        return True
    return False

def _local_name(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def _col_letters_to_index(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        if not ch.isalpha():
            continue
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    name = 'xl/sharedStrings.xml'
    if name not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(name))
    values: list[str] = []
    for si in root.iter():
        if _local_name(si.tag) != 'si':
            continue
        parts: list[str] = []
        for node in si.iter():
            if _local_name(node.tag) == 't':
                parts.append(node.text or '')
        values.append(''.join(parts))
    return values

def _xlsx_first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read('xl/workbook.xml'))
    rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    first_rid: Optional[str] = None
    for elem in workbook.iter():
        if _local_name(elem.tag) == 'sheet':
            for (key, val) in elem.attrib.items():
                if key.endswith('id'):
                    first_rid = val
                    break
            if first_rid:
                break
    if not first_rid:
        raise RuntimeError('Could not locate first worksheet in workbook.xml')
    rel_target: Optional[str] = None
    for elem in rels.iter():
        if _local_name(elem.tag) != 'Relationship':
            continue
        if elem.attrib.get('Id') == first_rid:
            rel_target = elem.attrib.get('Target')
            break
    if not rel_target:
        raise RuntimeError(f"Could not resolve worksheet relationship '{first_rid}'")
    rel_target = rel_target.replace('\\', '/')
    if rel_target.startswith('/'):
        return rel_target.lstrip('/')
    if rel_target.startswith('xl/'):
        return rel_target
    return f'xl/{rel_target}'

def _xlsx_cell_value(c: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = c.attrib.get('t')
    inline_text: Optional[str] = None
    v_text: Optional[str] = None
    for node in c:
        lname = _local_name(node.tag)
        if lname == 'v':
            v_text = node.text
        elif lname == 'is':
            parts: list[str] = []
            for inner in node.iter():
                if _local_name(inner.tag) == 't':
                    parts.append(inner.text or '')
            inline_text = ''.join(parts)
    if cell_type == 'inlineStr':
        return inline_text
    if cell_type == 's':
        if v_text is None:
            return None
        try:
            return shared_strings[int(v_text)]
        except Exception:
            return v_text
    if cell_type == 'b':
        return v_text == '1'
    if cell_type == 'str':
        return v_text
    if v_text is None:
        return inline_text
    try:
        num = float(v_text)
        if num.is_integer():
            return int(num)
        return num
    except Exception:
        return v_text

def _read_xlsx_fallback(content: bytes) -> Optional[pd.DataFrame]:
    """Very small XLSX reader fallback for strict/unusual OOXML workbooks.

    It reads the first worksheet, assumes the first row is the header, and is
    intended only as a compatibility fallback when pandas/openpyxl refuses to
    parse an otherwise valid file.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except Exception:
        return None
    try:
        sheet_path = _xlsx_first_sheet_path(zf)
        shared_strings = _xlsx_shared_strings(zf)
        root = ET.fromstring(zf.read(sheet_path))
    except Exception:
        return None
    rows_data: list[list[Any]] = []
    max_cols = 0
    for row in root.iter():
        if _local_name(row.tag) != 'row':
            continue
        current: dict[int, Any] = {}
        for cell in row:
            if _local_name(cell.tag) != 'c':
                continue
            ref = cell.attrib.get('r', '')
            letters = ''.join((ch for ch in ref if ch.isalpha()))
            if not letters:
                continue
            col_idx = _col_letters_to_index(letters)
            current[col_idx] = _xlsx_cell_value(cell, shared_strings)
        if not current:
            continue
        row_list = [None] * (max(current.keys()) + 1)
        for (idx, value) in current.items():
            row_list[idx] = value
        rows_data.append(row_list)
        max_cols = max(max_cols, len(row_list))
    if not rows_data:
        return None
    rows_data = [r + [None] * (max_cols - len(r)) for r in rows_data]
    header = rows_data[0]
    body = rows_data[1:]
    normalized_header: list[str] = []
    used: dict[str, int] = {}
    for (i, h) in enumerate(header):
        name = str(h).strip() if h is not None and str(h).strip() else f'column_{i + 1}'
        if name in used:
            used[name] += 1
            name = f'{name}_{used[name]}'
        else:
            used[name] = 0
        normalized_header.append(name)
    return pd.DataFrame(body, columns=normalized_header)

def _try_read_table_bytes(name: str, content: bytes) -> Optional[pd.DataFrame]:
    """Best-effort reader for common tabular file types."""
    low = name.lower()
    if low.endswith('.csv'):
        for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                return pd.read_csv(io.BytesIO(content), encoding=enc)
            except Exception:
                pass
        return None
    if low.endswith('.tsv') or low.endswith('.txt') or low.endswith('.data'):
        attempts = [{'sep': None, 'engine': 'python'}, {'sep': '\t'}, {'sep': ';'}, {'sep': ','}, {'delim_whitespace': True}]
        for kwargs in attempts:
            try:
                return pd.read_csv(io.BytesIO(content), **kwargs)
            except Exception:
                pass
        return None
    if low.endswith('.xlsx') or low.endswith('.xls'):
        try:
            require_package('openpyxl')
        except SystemExit:
            pass
        try:
            return pd.read_excel(io.BytesIO(content))
        except Exception:
            return _read_xlsx_fallback(content)
    return None

def _read_zip_first_table_from_bytes(content: bytes, depth: int=0, max_depth: int=5) -> pd.DataFrame:
    if depth > max_depth:
        raise RuntimeError(f'Exceeded max nested-zip depth ({max_depth}).')
    zf = zipfile.ZipFile(io.BytesIO(content))
    names = zf.namelist()
    errors: list[str] = []
    table_exts = ('.csv', '.xlsx', '.xls', '.tsv', '.txt', '.data')
    for name in names:
        low = name.lower()
        if not low.endswith(table_exts):
            continue
        try:
            with zf.open(name) as f:
                raw = f.read()
            df = _try_read_table_bytes(name, raw)
            if df is not None and isinstance(df, pd.DataFrame) and (df.shape[1] > 0):
                return df
            errors.append(f'{name}: parsed as None/empty')
        except Exception as e:
            errors.append(f'{name}: {type(e).__name__}: {e}')
    for name in names:
        low = name.lower()
        if not low.endswith('.zip'):
            continue
        try:
            with zf.open(name) as f:
                inner = f.read()
            return _read_zip_first_table_from_bytes(inner, depth=depth + 1, max_depth=max_depth)
        except zipfile.BadZipFile as e:
            errors.append(f'{name}: BadZipFile: {e}')
        except Exception as e:
            errors.append(f'{name}: {type(e).__name__}: {e}')
    raise RuntimeError(f'No readable table found in zip. Archive members: {names}. Tried and skipped: {errors}')

def read_zip_first_table(url: str) -> pd.DataFrame:
    with urlopen(url, timeout=60) as resp:
        content = resp.read()
    return _read_zip_first_table_from_bytes(content)

def read_local_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    df = _try_read_table_bytes(path.name, raw)
    if df is None or df.shape[1] == 0:
        raise RuntimeError(f'Could not parse local table file: {path}')
    return df

def _normalize_description_text(text: Any) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, float) and pd.isna(text):
        return None
    s = str(text).strip()
    if not s or s.lower() in {'nan', 'none'}:
        return None
    s = re.sub('\\s+', ' ', s)
    return s

def humanize_column_name(col: str) -> str:
    text = str(col).strip()
    text = text.replace('_', ' ').replace('-', ' ').replace('/', ' ')
    text = re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    if not text:
        return 'feature value'
    return text.lower()

def fallback_description_for_column(col: str) -> str:
    phrase = humanize_column_name(col)
    if phrase.startswith('whether '):
        return phrase
    return f'value of {phrase}'

def normalize_metadata_dict(metadata: Optional[dict[str, Any]], feature_columns: list[str]) -> dict[str, str]:
    if not metadata:
        return {}
    out: dict[str, str] = {}
    for col in feature_columns:
        if col not in metadata:
            continue
        desc = _normalize_description_text(metadata[col])
        if desc:
            out[col] = desc
    return out

def infer_metadata_from_columns(df: pd.DataFrame, target_col: str) -> dict[str, str]:
    return {col: fallback_description_for_column(col) for col in df.columns if col != target_col}

def extract_uci_metadata(ds: Any, feature_columns: list[str]) -> dict[str, str]:
    variables = getattr(ds, 'variables', None)
    if variables is None or not isinstance(variables, pd.DataFrame) or variables.empty:
        return {}
    name_col = None
    desc_col = None
    for cand in ['name', 'Name']:
        if cand in variables.columns:
            name_col = cand
            break
    for cand in ['description', 'Description']:
        if cand in variables.columns:
            desc_col = cand
            break
    if name_col is None or desc_col is None:
        return {}
    out: dict[str, str] = {}
    feature_set = set(feature_columns)
    for (_, row) in variables.iterrows():
        name = row.get(name_col)
        desc = row.get(desc_col)
        if name in feature_set:
            cleaned = _normalize_description_text(desc)
            if cleaned:
                out[str(name)] = cleaned
    return out

def extract_openml_metadata(openml_ds: Any, feature_columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    feature_set = set(feature_columns)
    features = getattr(openml_ds, 'features', None)
    if not features:
        return out
    for feat in features.values():
        name = getattr(feat, 'name', None)
        if name not in feature_set:
            continue
        desc = None
        for attr in ['description', 'comment', 'data_type']:
            value = getattr(feat, attr, None)
            if _normalize_description_text(value):
                desc = _normalize_description_text(value)
                break
        if desc:
            out[str(name)] = str(desc)
    return out

def download_featllm_metadata(dataset_name: str) -> Optional[dict[str, str]]:
    alias = FEATLLM_METADATA_ALIAS.get(dataset_name)
    if not alias:
        return None
    url = f'{FEATLLM_METADATA_BASE}/{alias}-metadata.json'
    try:
        with urlopen(url, timeout=30) as resp:
            raw = resp.read().decode('utf-8')
    except (HTTPError, URLError, TimeoutError):
        return None
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): str(v) for (k, v) in data.items()}

def ensure_metadata_json(name: str, df: pd.DataFrame, target_col: str, out_dir: Path, *, force_metadata: bool=False, metadata_override: Optional[dict[str, str]]=None, uci_ds: Any=None, openml_ds: Any=None) -> None:
    meta_path = metadata_path(name, out_dir)
    if maybe_skip_metadata(meta_path, force_metadata=force_metadata):
        return
    feature_columns = [c for c in df.columns if c != target_col]
    meta = normalize_metadata_dict(metadata_override, feature_columns)
    if not meta:
        meta = normalize_metadata_dict(download_featllm_metadata(name), feature_columns)
    if not meta and uci_ds is not None:
        meta = normalize_metadata_dict(extract_uci_metadata(uci_ds, feature_columns), feature_columns)
    if not meta and openml_ds is not None:
        meta = normalize_metadata_dict(extract_openml_metadata(openml_ds, feature_columns), feature_columns)
    if not meta and name in MANUAL_METADATA:
        meta = normalize_metadata_dict(MANUAL_METADATA[name], feature_columns)
    if not meta:
        meta = infer_metadata_from_columns(df, target_col)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'[ok] {meta_path.name} | features={len(meta)}')

def fetch_openml_dataset(openml_id: int) -> tuple[pd.DataFrame, str, str, Any]:
    require_package('openml')
    import openml
    ds = openml.datasets.get_dataset(openml_id, download_data=True)
    target = ds.default_target_attribute
    if target is None:
        raise ValueError(f'OpenML dataset {openml_id} has no default_target_attribute')
    if isinstance(target, str) and ',' in target:
        target = target.split(',')[0].strip()
    (X, y, _, _) = ds.get_data(dataset_format='dataframe', target=target)
    y = to_series(y, fallback_name=target)
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    ignore_attr = getattr(ds, 'ignore_attribute', None)
    if isinstance(ignore_attr, str) and ignore_attr.strip():
        drop_cols = [c.strip() for c in ignore_attr.split(',') if c.strip()]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return (df, y.name, str(getattr(ds, 'name', openml_id)), ds)

def fetch_uci_dataset(uci_id: int):
    require_package('ucimlrepo')
    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo(id=uci_id)

def resolve_kaggle_cli() -> str:
    kaggle_bin = shutil.which('kaggle')
    if kaggle_bin:
        return kaggle_bin
    candidate = Path(sys.executable).with_name('kaggle')
    if candidate.exists():
        return str(candidate)
    raise SystemExit('Could not find Kaggle CLI executable. Install it with `pip install kaggle`, then ensure the `kaggle` command is on PATH or in the current environment bin/.')

def run_kaggle_download(dataset_slug: str, filename: Optional[str]=None) -> Path:
    kaggle_bin = resolve_kaggle_cli()
    tmpdir = Path(tempfile.mkdtemp(prefix='kaggle_dl_'))
    cmd = [kaggle_bin, 'datasets', 'download', '-d', dataset_slug, '-p', str(tmpdir), '--unzip']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Kaggle download failed for {dataset_slug}\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}')
    if filename is not None:
        candidate = tmpdir / filename
        if not candidate.exists():
            files = [p.name for p in tmpdir.iterdir() if p.is_file()]
            raise FileNotFoundError(f"Expected '{filename}' inside Kaggle download for {dataset_slug}, found: {files}")
        return candidate
    files = sorted((p for p in tmpdir.iterdir() if p.is_file()))
    if len(files) != 1:
        raise FileNotFoundError(f'Expected 1 file for {dataset_slug}, found {[p.name for p in files]}')
    return files[0]

def write_openml(name: str, openml_id: int, out_dir: Path, *, force: bool=False, force_metadata: bool=False) -> None:
    out_path = out_dir / f'{name}.csv'
    if maybe_skip(out_path, force):
        if not maybe_skip_metadata(metadata_path(name, out_dir), force_metadata):
            df_existing = pd.read_csv(out_path)
            ensure_metadata_json(name, df_existing, df_existing.columns[-1], out_dir, force_metadata=force_metadata)
        return
    (df, target, src_name, openml_ds) = fetch_openml_dataset(openml_id)
    df = save_featllm_csv(df, target, out_path)
    ensure_metadata_json(name, df, target, out_dir, force_metadata=force_metadata, openml_ds=openml_ds)
    print(f'[ok] {name}.csv <- OpenML {openml_id} ({src_name}) | shape={df.shape}')

def write_kaggle_csv(name: str, dataset_slug: str, csv_filename: str, target_col: str, out_dir: Path, *, force: bool=False, force_metadata: bool=False, drop_cols: Optional[list[str]]=None, metadata_override: Optional[dict[str, str]]=None) -> None:
    out_path = out_dir / f'{name}.csv'
    if maybe_skip(out_path, force):
        if not maybe_skip_metadata(metadata_path(name, out_dir), force_metadata):
            df_existing = pd.read_csv(out_path)
            ensure_metadata_json(name, df_existing, df_existing.columns[-1], out_dir, force_metadata=force_metadata, metadata_override=metadata_override)
        return
    csv_path = run_kaggle_download(dataset_slug, csv_filename)
    df = pd.read_csv(csv_path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = save_featllm_csv(df, target_col, out_path)
    ensure_metadata_json(name, df, target_col, out_dir, force_metadata=force_metadata, metadata_override=metadata_override)
    print(f'[ok] {name}.csv <- Kaggle {dataset_slug} | shape={df.shape}')

def write_cultivars_regression(out_dir: Path, *, force: bool=False, force_metadata: bool=False) -> None:
    out_path = out_dir / 'cultivars_regression.csv'
    if maybe_skip(out_path, force):
        if not maybe_skip_metadata(metadata_path('cultivars_regression', out_dir), force_metadata):
            df_existing = pd.read_csv(out_path)
            ensure_metadata_json('cultivars_regression', df_existing, df_existing.columns[-1], out_dir, force_metadata=force_metadata)
        return
    ds = fetch_uci_dataset(913)
    df = ds.data.features.copy()
    target_col = 'GY'
    if target_col not in df.columns:
        raise ValueError(f'Expected target column {target_col}, got columns: {list(df.columns)}')
    df[target_col] = coerce_numeric_like(df[target_col])
    df = df.dropna(subset=[target_col])
    df = save_featllm_csv(df, target_col, out_path)
    ensure_metadata_json('cultivars_regression', df, target_col, out_dir, force_metadata=force_metadata, uci_ds=ds)
    print(f'[ok] cultivars_regression.csv <- UCI 913 ({target_col}) | shape={df.shape}')

def write_gallstone_direct(out_dir: Path, *, force: bool=False, force_metadata: bool=False, local_excel: Optional[str]=None) -> None:
    out_path = out_dir / 'gallstone.csv'
    if maybe_skip(out_path, force):
        if not maybe_skip_metadata(metadata_path('gallstone', out_dir), force_metadata):
            df_existing = pd.read_csv(out_path)
            ensure_metadata_json('gallstone', df_existing, df_existing.columns[-1], out_dir, force_metadata=force_metadata)
        return
    if local_excel:
        df = read_local_table(local_excel)
        source_desc = f'local file {local_excel}'
    else:
        url = 'https://archive.ics.uci.edu/static/public/1150/gallstone-1.zip'
        df = read_zip_first_table(url)
        source_desc = 'UCI direct download'
    df.columns = [str(c).strip() for c in df.columns]
    target_candidates = [c for c in df.columns if 'gallstone' in c.lower() and 'status' in c.lower()]
    if not target_candidates:
        for cand in ['Gallstone Status', 'gallstone_status', 'label', 'class', 'target', 'y']:
            if cand in df.columns:
                target_candidates = [cand]
                break
    if not target_candidates:
        raise RuntimeError(f'Could not find Gallstone target column. Columns: {list(df.columns)}')
    target_col = target_candidates[0]
    df = save_featllm_csv(df, target_col, out_path)
    ensure_metadata_json('gallstone', df, target_col, out_dir, force_metadata=force_metadata)
    print(f'[ok] gallstone.csv <- {source_desc} | shape={df.shape}')

def write_bike_regression(out_dir: Path, *, force: bool=False, force_metadata: bool=False) -> None:
    out_path = out_dir / 'bike.csv'
    if maybe_skip(out_path, force):
        if not maybe_skip_metadata(metadata_path('bike', out_dir), force_metadata):
            df_existing = pd.read_csv(out_path)
            ensure_metadata_json('bike', df_existing, df_existing.columns[-1], out_dir, force_metadata=force_metadata)
        return
    ds = fetch_uci_dataset(275)
    raw = getattr(ds.data, 'original', None)
    if isinstance(raw, pd.DataFrame) and 'cnt' in raw.columns:
        df = raw.copy()
    else:
        X = ds.data.features.copy()
        y_df = ds.data.targets.copy() if isinstance(ds.data.targets, pd.DataFrame) else pd.DataFrame(ds.data.targets)
        if 'cnt' not in y_df.columns and y_df.shape[1] == 1:
            y_df.columns = ['cnt']
        df = pd.concat([X.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
    drop_cols = [c for c in ['instant', 'dteday', 'casual', 'registered'] if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    target_col = 'cnt' if 'cnt' in df.columns else df.columns[-1]
    df = save_featllm_csv(df, target_col, out_path)
    ensure_metadata_json('bike', df, target_col, out_dir, force_metadata=force_metadata, uci_ds=ds)
    print(f'[ok] bike.csv <- UCI 275 | shape={df.shape}')

def write_infrared_thermography_temperature(out_dir: Path, *, force: bool=False, force_metadata: bool=False, target: str='aveOralF') -> None:
    out_path = out_dir / 'infrared_thermography_temperature.csv'
    if maybe_skip(out_path, force):
        if not maybe_skip_metadata(metadata_path('infrared_thermography_temperature', out_dir), force_metadata):
            df_existing = pd.read_csv(out_path)
            ensure_metadata_json('infrared_thermography_temperature', df_existing, df_existing.columns[-1], out_dir, force_metadata=force_metadata)
        return
    ds = fetch_uci_dataset(925)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        if target not in y.columns:
            raise ValueError(f'target={target} is not in available targets: {list(y.columns)}')
        y = y[target]
    else:
        y = to_series(y)
        if y.name != target:
            print(f"[warn] ucimlrepo returned a single target '{y.name}', using it instead of requested '{target}'.")
            target = y.name
    drop_cols = [c for c in ['SubjectID'] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    y = to_series(y, fallback_name=target).rename(target)
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df = save_featllm_csv(df, target, out_path)
    ensure_metadata_json('infrared_thermography_temperature', df, target, out_dir, force_metadata=force_metadata, uci_ds=ds)
    print(f'[ok] infrared_thermography_temperature.csv <- UCI 925 ({target}) | shape={df.shape}')

def write_wine_regression(out_dir: Path, *, force: bool=False, force_metadata: bool=False) -> None:
    out_path = out_dir / 'wine.csv'
    if maybe_skip(out_path, force):
        if not maybe_skip_metadata(metadata_path('wine', out_dir), force_metadata):
            df_existing = pd.read_csv(out_path)
            ensure_metadata_json('wine', df_existing, df_existing.columns[-1], out_dir, force_metadata=force_metadata)
        return
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    df = pd.read_csv(url, sep=';')
    df = save_featllm_csv(df, 'quality', out_path)
    ensure_metadata_json('wine', df, 'quality', out_dir, force_metadata=force_metadata, metadata_override=MANUAL_METADATA.get('wine'))
    print(f'[ok] wine.csv <- UCI winequality-white.csv | shape={df.shape}')

PAPER_DATASETS = (
    "adult", "blood", "breast-w", "credit-g", "cdc_diabetes", "heart", "gallstone",
    "cpu_small", "diamonds", "forest-fires", "housing", "bike", "cultivars_regression",
    "infrared_thermography_temperature", "wine", "insurance",
)

def main():
    parser = argparse.ArgumentParser(description="Prepare only the 16 LEAF benchmark datasets.")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--datasets", nargs="+", choices=PAPER_DATASETS, default=list(PAPER_DATASETS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gallstone-file", default=None)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    openml_ids = {"adult": 1590, "blood": 1464, "breast-w": 15, "credit-g": 31,
                  "cpu_small": 562, "diamonds": 42225, "forest-fires": 42363, "housing": 43996}
    kaggle = {
        "heart": ("fedesoriano/heart-failure-prediction", "heart.csv", "HeartDisease"),
        "cdc_diabetes": ("alexteboul/diabetes-health-indicators-dataset",
                         "diabetes_012_health_indicators_BRFSS2015.csv", "Diabetes_012"),
        "insurance": ("teertha/ushealthinsurancedataset", "insurance.csv", "charges"),
    }
    special = {
        "bike": write_bike_regression,
        "cultivars_regression": write_cultivars_regression,
        "infrared_thermography_temperature": write_infrared_thermography_temperature,
        "wine": write_wine_regression,
    }
    for dataset in args.datasets:
        if dataset in openml_ids:
            write_openml(dataset, openml_ids[dataset], output, force=args.force)
        elif dataset in kaggle:
            slug, filename, target = kaggle[dataset]
            write_kaggle_csv(dataset, slug, filename, target, output, force=args.force,
                             metadata_override=MANUAL_METADATA.get(dataset))
        elif dataset == "gallstone":
            write_gallstone_direct(output, force=args.force, local_excel=args.gallstone_file)
        else:
            special[dataset](output, force=args.force)

if __name__ == "__main__":
    main()
