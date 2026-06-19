import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import warnings
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings(
    "ignore",
    message=r"Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector\.",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import TrainerCallback


DEFAULT_KAGGLE_DATASET_ROOT = "/kaggle/input"
DEFAULT_INPUT_XLSX = DEFAULT_KAGGLE_DATASET_ROOT
DEFAULT_MODEL_NAME = DEFAULT_KAGGLE_DATASET_ROOT
DEFAULT_OUTPUT_DIR = "/kaggle/working/nllb_k2c_ko2zh_lora_v2_1_nllb13b"
DEFAULT_SRC_LANG = "kor_Hang"
DEFAULT_TGT_LANG = "zho_Hans"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_text(text) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s


def resolve_input_xlsx_path(path_or_dir: str) -> str:
    path_or_dir = os.path.normpath(str(path_or_dir).strip().strip('"').strip("'"))
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        xlsx_files = []
        for root, _dirs, files in os.walk(path_or_dir):
            for name in files:
                if name.lower().endswith(".xlsx"):
                    xlsx_files.append(os.path.join(root, name))
        if not xlsx_files:
            raise RuntimeError(f"未在目录中找到 xlsx 语料文件: {path_or_dir}")
        xlsx_files = sorted(xlsx_files)
        preferred = [p for p in xlsx_files if "cleaned_new-corpus" in os.path.basename(p).lower()]
        return preferred[0] if preferred else xlsx_files[0]
    raise RuntimeError(f"语料路径不存在: {path_or_dir}")


def resolve_nllb_model_path(model_name_or_dir: str) -> str:
    model_name_or_dir = str(model_name_or_dir).strip().strip('"').strip("'")
    if not model_name_or_dir:
        raise RuntimeError("模型路径不能为空")
    if not os.path.exists(model_name_or_dir):
        if str(model_name_or_dir).startswith("/"):
            raise RuntimeError(
                f"模型路径不存在: {model_name_or_dir}\n"
                "请把 --model-name 指向你上传到 Kaggle Input 的 NLLB-200-1.3B 模型目录，"
                "或者直接传 /kaggle/input 让脚本自动在其中定位。"
            )
        raise RuntimeError(
            f"模型路径不存在: {model_name_or_dir}\n"
            "本脚本只支持从本地目录加载 NLLB-200-1.3B（Kaggle Input/Working）。"
        )

    model_dir = os.path.normpath(model_name_or_dir)

    def has_model_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return os.path.exists(os.path.join(path, "config.json")) and (
            os.path.exists(os.path.join(path, "pytorch_model.bin"))
            or os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
        )

    if has_model_files(model_dir):
        return model_dir

    refs_main = os.path.join(model_dir, "refs", "main")
    if os.path.exists(refs_main):
        try:
            with open(refs_main, "r", encoding="utf-8") as f:
                snapshot_id = f.read().strip()
            snapshot_dir = os.path.join(model_dir, "snapshots", snapshot_id)
            if snapshot_id and has_model_files(snapshot_dir):
                return snapshot_dir
        except Exception:
            pass

    snapshots_dir = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        candidates = []
        for name in os.listdir(snapshots_dir):
            cand = os.path.join(snapshots_dir, name)
            if has_model_files(cand):
                candidates.append(cand)
        if candidates:
            return sorted(candidates)[-1]

    recursive_candidates = []
    for root, dirs, _files in os.walk(model_dir):
        for d in dirs:
            cand = os.path.join(root, d)
            if has_model_files(cand):
                recursive_candidates.append(cand)
    if recursive_candidates:
        preferred = [p for p in recursive_candidates if "nllb-200-1.3b" in p.lower() or "nllb-200-13b" in p.lower()]
        if preferred:
            return sorted(preferred)[0]
        raise RuntimeError(
            "当前脚本只用于 NLLB-200-1.3B。请把 --model-name 指向 nllb-200-1.3B 的模型目录。"
        )

    raise RuntimeError(f"无法在目录中定位可加载的 NLLB-200-1.3B 模型: {model_dir}")


def _extract_checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if os.path.isdir(full_path) and re.fullmatch(r"checkpoint-\d+", name):
            candidates.append(full_path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (_extract_checkpoint_step(x), x))[-1]


def read_best_checkpoint_record(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None

    best_json = os.path.join(output_dir, "best_checkpoint.json")
    if os.path.isfile(best_json):
        try:
            with open(best_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            checkpoint_path = str(payload.get("best_model_checkpoint", "")).strip()
            if checkpoint_path and os.path.isdir(checkpoint_path):
                return checkpoint_path
        except Exception:
            pass

    best_txt = os.path.join(output_dir, "best_checkpoint.txt")
    if os.path.isfile(best_txt):
        try:
            with open(best_txt, "r", encoding="utf-8") as f:
                checkpoint_path = f.read().strip()
            if checkpoint_path and os.path.isdir(checkpoint_path):
                return checkpoint_path
        except Exception:
            pass

    return None


def resolve_resume_checkpoint_path(resume_from_checkpoint: str, output_dir: str) -> str | None:
    raw_value = str(resume_from_checkpoint or "").strip().strip('"').strip("'")
    if not raw_value:
        return None

    mode = raw_value.lower()
    if mode in {"true", "1", "yes", "y"}:
        mode = "best"
    elif mode in {"false", "0", "no", "n", "none", "null"}:
        return None

    if mode in {"best", "auto"}:
        best_checkpoint = read_best_checkpoint_record(output_dir)
        if best_checkpoint:
            return best_checkpoint
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint:
            return latest_checkpoint
        print(f"[续训] 未找到 best 或 checkpoint-*，改为从头开始训练: {output_dir}")
        return None

    if mode in {"latest", "last"}:
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint:
            return latest_checkpoint
        print(f"[续训] 未找到 checkpoint-*，改为从头开始训练: {output_dir}")
        return None

    checkpoint_path = os.path.normpath(raw_value)
    if os.path.isfile(checkpoint_path):
        if os.path.basename(checkpoint_path) == "best_checkpoint.json":
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            best_checkpoint = str(payload.get("best_model_checkpoint", "")).strip()
            if best_checkpoint and os.path.isdir(best_checkpoint):
                return best_checkpoint
        if os.path.basename(checkpoint_path) == "best_checkpoint.txt":
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                best_checkpoint = f.read().strip()
            if best_checkpoint and os.path.isdir(best_checkpoint):
                return best_checkpoint
        raise RuntimeError(f"续训失败：文件不是有效的 best checkpoint 记录: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        if re.fullmatch(r"checkpoint-\d+", os.path.basename(checkpoint_path)):
            return checkpoint_path
        best_checkpoint = read_best_checkpoint_record(checkpoint_path)
        if best_checkpoint:
            return best_checkpoint
        latest_checkpoint = find_latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            return latest_checkpoint
        raise RuntimeError(f"续训失败：目录中未找到可恢复的 checkpoint: {checkpoint_path}")

    raise RuntimeError(f"续训路径不存在: {raw_value}")


def make_zip_from_directory(source_dir: str, zip_path: str) -> str | None:
    source_dir = os.path.normpath(str(source_dir).strip())
    if not source_dir or not os.path.isdir(source_dir):
        return None

    zip_base = os.path.splitext(os.path.normpath(zip_path))[0]
    zip_file = f"{zip_base}.zip"
    if os.path.exists(zip_file):
        os.remove(zip_file)

    return shutil.make_archive(
        zip_base,
        "zip",
        root_dir=os.path.dirname(source_dir),
        base_dir=os.path.basename(source_dir),
    )


def persist_best_checkpoint_artifacts(
    output_dir: str,
    best_model_checkpoint: str | None,
    best_metric,
    metric_for_best_model: str,
    greater_is_better: bool,
    global_step: int,
) -> str | None:
    best_checkpoint_dir = str(best_model_checkpoint or "").strip()
    best_zip_path = None

    if best_checkpoint_dir and os.path.isdir(best_checkpoint_dir):
        best_zip_path = make_zip_from_directory(
            best_checkpoint_dir,
            os.path.join(output_dir, "best_checkpoint.zip"),
        )

    best_checkpoint_payload = {
        "best_model_checkpoint": best_checkpoint_dir,
        "best_metric": best_metric,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "global_step": global_step,
        "best_checkpoint_zip": best_zip_path,
    }
    with open(os.path.join(output_dir, "best_checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump(best_checkpoint_payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "best_checkpoint.txt"), "w", encoding="utf-8") as f:
        f.write(best_checkpoint_dir)

    return best_zip_path


def prune_non_best_checkpoints(output_dir: str, best_checkpoint: str | None) -> list[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return []

    best_checkpoint = os.path.normpath(str(best_checkpoint or "").strip()) if best_checkpoint else ""
    removed_dirs: list[str] = []
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if not (os.path.isdir(full_path) and re.fullmatch(r"checkpoint-\d+", name)):
            continue
        norm_full_path = os.path.normpath(full_path)
        if best_checkpoint and norm_full_path == best_checkpoint:
            continue
        shutil.rmtree(full_path, ignore_errors=True)
        removed_dirs.append(norm_full_path)
    return removed_dirs


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


@dataclass
class CorpusConfig:
    src_col: str
    tgt_col: str
    tag_col: str | None
    quality_col: str | None
    delete_mark_col: str | None


def detect_columns(df: pd.DataFrame) -> CorpusConfig:
    src_col = find_column(df, ["KO", "ko", "source", "src", "korean"])
    tgt_col = find_column(df, ["ZH修正", "zh修正", "修正", "ZH", "zh", "target", "tgt", "chinese"])
    if not src_col or not tgt_col:
        raise RuntimeError(f"未能识别 KO / ZH修正 列，当前列名: {list(df.columns)}")
    tag_col = find_column(df, ["tag", "标签", "category"])
    quality_col = find_column(df, ["quality_score", "score", "quality", "质量分"])
    delete_mark_col = find_column(df, ["delete_mark", "deleted", "drop", "删除标记"])
    return CorpusConfig(
        src_col=src_col,
        tgt_col=tgt_col,
        tag_col=tag_col,
        quality_col=quality_col,
        delete_mark_col=delete_mark_col,
    )


def parse_delete_mark(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) != 0.0
    s = str(value).strip().lower()
    return s in {"1", "1.0", "true", "yes", "y", "delete", "deleted", "drop", "是", "删", "删除"}


def parse_quality(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def detect_structural_features(src_text: str, tgt_text: str) -> dict[str, bool]:
    merged = f"{src_text}\n{tgt_text}"
    return {
        "digits_or_units": bool(re.search(r"\d", merged) or re.search(r"\b(?:kg|mm|cm|mb|gb|hz|ms|sec|min|km|px|%)\b", merged, flags=re.I)),
        "brackets_or_placeholders": bool(re.search(r"[\(\)\[\]\{\}<>]", merged) or re.search(r"\{[^{}]+\}|%s|%d|:\w+|\$\{[^{}]+\}", merged)),
        "path_or_symbols": bool(re.search(r"[\\/_:\-=+@#*&|~]", merged) or re.search(r"->|=>|::", merged)),
        "mixed_script": bool(
            re.search(r"[A-Za-z]{2,}", merged)
            and re.search(r"[\u4e00-\u9fff]", merged)
            or re.search(r"[A-Za-z]{2,}", merged)
            and re.search(r"[\uac00-\ud7af]", merged)
        ),
        "dense_punctuation": len(re.findall(r"[,.:;!?()\[\]{}<>/%\-+_=#@\\|]", merged)) >= 4,
    }


def compute_row_boost(src_text: str, tgt_text: str, tag_text: str, args) -> int:
    tag_lower = str(tag_text or "").lower()
    feature_flags = detect_structural_features(src_text, tgt_text)
    extra = 0

    if args.automatic_boost > 1 and re.search(r"automatic|automation|auto|系统|按钮|界面|程序|功能", tag_lower):
        extra += max(0, int(args.automatic_boost) - 1)

    if not args.disable_structural_boost:
        structural_extra = 0
        if feature_flags["digits_or_units"] and int(args.boost_digits_units) > 0:
            structural_extra = 1
        if feature_flags["brackets_or_placeholders"] and int(args.boost_brackets_placeholders) > 0:
            structural_extra = 1
        if feature_flags["path_or_symbols"] and int(args.boost_path_symbols) > 0:
            structural_extra = 1
        if feature_flags["mixed_script"] and int(args.boost_mixed_script) > 0:
            structural_extra = 1
        if feature_flags["dense_punctuation"] and int(args.boost_dense_punctuation) > 0:
            structural_extra = 1
        extra += structural_extra

    return max(0, min(int(args.max_extra_copies), extra))


def expand_training_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    expanded_rows: list[dict] = []
    for row in df.to_dict("records"):
        expanded_rows.append(row)
        extra = compute_row_boost(row["src_text"], row["tgt_text"], row.get("tag", ""), args)
        for _ in range(extra):
            expanded_rows.append(dict(row))
    return pd.DataFrame(expanded_rows)


def load_parallel_corpus(args) -> pd.DataFrame:
    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet_name or 0)
    cfg = detect_columns(df)

    total_rows = len(df)
    work = df.copy()
    work[cfg.src_col] = work[cfg.src_col].map(normalize_text)
    work[cfg.tgt_col] = work[cfg.tgt_col].map(normalize_text)
    before_non_empty = len(work)
    work = work[(work[cfg.src_col] != "") & (work[cfg.tgt_col] != "")]
    after_non_empty = len(work)
    deleted_by_mark = 0

    if cfg.delete_mark_col:
        delete_mask = work[cfg.delete_mark_col].map(parse_delete_mark)
        deleted_by_mark = int(delete_mask.sum())
        work = work[~delete_mask]

    after_delete_mark = len(work)

    if cfg.tag_col and args.keep_tags:
        keep_tags = [t.strip().lower() for t in args.keep_tags.split(",") if t.strip()]
        if keep_tags:
            tag_s = work[cfg.tag_col].fillna("").astype(str).str.lower()
            work = work[tag_s.apply(lambda x: any(t in x for t in keep_tags))]

    after_keep_tags = len(work)

    out = pd.DataFrame(
        {
            "src_text": work[cfg.src_col].astype(str),
            "tgt_text": work[cfg.tgt_col].astype(str),
            "tag": work[cfg.tag_col].astype(str) if cfg.tag_col else "",
            "quality_score": work[cfg.quality_col].map(parse_quality) if cfg.quality_col else np.nan,
        }
    )
    before_dedup = len(out)
    out = out.drop_duplicates(subset=["src_text", "tgt_text"]).reset_index(drop=True)
    base_size = len(out)
    out = expand_training_rows(out, args).reset_index(drop=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"原始总行数: {total_rows}")
    print(f"源/目标非空后: {after_non_empty} (去掉 {before_non_empty - after_non_empty} 行)")
    if cfg.delete_mark_col:
        print(f"delete_mark 删除: {deleted_by_mark} 行")
    print(f"delete_mark 后: {after_delete_mark}")
    if cfg.tag_col and args.keep_tags:
        print(f"keep_tags 过滤后: {after_keep_tags}")
    print(f"去重前样本数: {before_dedup}")
    print(f"基础去重后样本数: {base_size}")
    print(f"增强后样本数: {len(out)}")
    return out


def train_val_split(df: pd.DataFrame, valid_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 20:
        valid_size = max(1, min(4, len(df) // 5 or 1))
        valid_df = df.iloc[:valid_size].reset_index(drop=True)
        train_df = df.iloc[valid_size:].reset_index(drop=True)
        if train_df.empty:
            raise RuntimeError("训练集为空，请增加语料。")
        return train_df, valid_df

    try:
        from sklearn.model_selection import train_test_split

        if "tag" in df.columns and df["tag"].astype(str).nunique() > 1:
            train_df, valid_df = train_test_split(
                df,
                test_size=max(valid_ratio, 0.02),
                random_state=seed,
                shuffle=True,
                stratify=df["tag"].astype(str),
            )
        else:
            train_df, valid_df = train_test_split(
                df,
                test_size=max(valid_ratio, 0.02),
                random_state=seed,
                shuffle=True,
            )
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
    except Exception:
        valid_size = max(1, int(len(df) * valid_ratio))
        valid_df = df.iloc[:valid_size].reset_index(drop=True)
        train_df = df.iloc[valid_size:].reset_index(drop=True)
        if train_df.empty:
            raise RuntimeError("训练集为空，请降低 valid_ratio 或增加语料。")
        return train_df, valid_df


def build_datasets(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[Dataset, Dataset]:
    return Dataset.from_pandas(train_df, preserve_index=False), Dataset.from_pandas(valid_df, preserve_index=False)


def resolve_precision(prefer_bf16: bool):
    import torch

    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False, "precision_name": "fp32"}

    if prefer_bf16:
        try:
            if torch.cuda.is_bf16_supported():
                return {"fp16": False, "bf16": True, "precision_name": "bf16"}
        except Exception:
            pass

    return {"fp16": True, "bf16": False, "precision_name": "fp16"}


def _version_tuple(v: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", str(v or ""))
    return tuple(int(x) for x in nums[:4]) if nums else (0,)


def disable_incompatible_torchao_for_peft() -> None:
    try:
        import importlib.metadata as importlib_metadata

        torchao_version = importlib_metadata.version("torchao")
    except Exception:
        return

    if _version_tuple(torchao_version) >= _version_tuple("0.16.0"):
        return

    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_import_utils.is_torchao_available = lambda: False
        peft_lora_torchao.is_torchao_available = lambda: False
        print(f"[兼容修复] 已禁用不兼容的 torchao {torchao_version}，避免影响 PEFT LoRA 注入。")
    except Exception as e:
        print(f"[兼容修复] 禁用 torchao 失败: {e}")


def safe_import_metrics():
    try:
        import evaluate

        bleu_metric = evaluate.load("sacrebleu")
        chrf_metric = evaluate.load("chrf")
        return bleu_metric, chrf_metric
    except Exception as e:
        print(f"[警告] BLEU/chrF 指标未启用: {e}")
        return None, None


class PersistBestCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, metric_for_best_model: str, greater_is_better: bool):
        self.output_dir = output_dir
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self._last_best_checkpoint = ""

    def on_save(self, args, state, control, **kwargs):
        best_checkpoint = str(state.best_model_checkpoint or "").strip()
        if best_checkpoint and best_checkpoint != self._last_best_checkpoint and os.path.isdir(best_checkpoint):
            best_zip_path = persist_best_checkpoint_artifacts(
                output_dir=self.output_dir,
                best_model_checkpoint=best_checkpoint,
                best_metric=state.best_metric,
                metric_for_best_model=self.metric_for_best_model,
                greater_is_better=self.greater_is_better,
                global_step=state.global_step,
            )
            self._last_best_checkpoint = best_checkpoint
            if best_zip_path:
                print(f"[best] 已更新最佳 checkpoint 压缩包: {best_zip_path}")
        removed_dirs = prune_non_best_checkpoints(self.output_dir, best_checkpoint)
        if removed_dirs:
            print(f"[checkpoint] 已删除非最佳 checkpoint: {len(removed_dirs)} 个")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        best_checkpoint = str(state.best_model_checkpoint or "").strip()
        best_zip_path = persist_best_checkpoint_artifacts(
            output_dir=self.output_dir,
            best_model_checkpoint=best_checkpoint,
            best_metric=state.best_metric,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            global_step=state.global_step,
        )
        removed_dirs = prune_non_best_checkpoints(self.output_dir, best_checkpoint)
        if best_checkpoint and best_zip_path:
            print(f"[best] 训练结束，最佳 checkpoint 压缩包: {best_zip_path}")
        if removed_dirs:
            print(f"[checkpoint] 训练结束已清理非最佳 checkpoint: {len(removed_dirs)} 个")
        return control


def main():
    parser = argparse.ArgumentParser(description="Kaggle: 微调 NLLB-200-1.3B v2.1 (KO -> ZH修正)")
    parser.add_argument("--input-xlsx", default=DEFAULT_INPUT_XLSX)
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--src-lang", default=DEFAULT_SRC_LANG)
    parser.add_argument("--tgt-lang", default=DEFAULT_TGT_LANG)
    parser.add_argument("--keep-tags", default="")
    parser.add_argument("--automatic-boost", type=int, default=1)
    parser.add_argument("--disable-structural-boost", action="store_true")
    parser.add_argument("--boost-digits-units", type=int, default=1)
    parser.add_argument("--boost-brackets-placeholders", type=int, default=1)
    parser.add_argument("--boost-path-symbols", type=int, default=1)
    parser.add_argument("--boost-mixed-script", type=int, default=1)
    parser.add_argument("--boost-dense-punctuation", type=int, default=1)
    parser.add_argument("--max-extra-copies", type=int, default=1)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--max-source-length", type=int, default=196)
    parser.add_argument("--max-target-length", type=int, default=98)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--generation-num-beams", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit-checkpoints", type=int, default=2)
    parser.add_argument("--merge-and-save", action="store_true", default=True)
    parser.add_argument("--resume-from-checkpoint", default="best")
    parser.add_argument("--prefer-bf16", action="store_true")
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=20)
    args, _ = parser.parse_known_args()

    if args.no_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args.input_xlsx = resolve_input_xlsx_path(args.input_xlsx)
    args.model_name = resolve_nllb_model_path(args.model_name)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    disable_incompatible_torchao_for_peft()
    from transformers import (
        AutoConfig,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    precision_cfg = resolve_precision(prefer_bf16=bool(args.prefer_bf16))
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = bool(torch.cuda.is_available() and gpu_count >= 2 and not args.no_multi_gpu)
    active_gpu_count = gpu_count if use_multi_gpu else min(1, gpu_count)
    if active_gpu_count <= 0:
        active_gpu_count = 1

    df = load_parallel_corpus(args)
    train_df, valid_df = train_val_split(df, args.valid_ratio, args.seed)

    train_df.to_csv(os.path.join(args.output_dir, "train_preview.csv"), index=False, encoding="utf-8-sig")
    valid_df.to_csv(os.path.join(args.output_dir, "valid_preview.csv"), index=False, encoding="utf-8-sig")

    effective_global_batch = (
        int(args.per_device_train_batch_size)
        * int(active_gpu_count)
        * int(args.gradient_accumulation_steps)
    )
    resume_checkpoint = resolve_resume_checkpoint_path(args.resume_from_checkpoint, args.output_dir)
    print(f"总语料: {len(df)}")
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(valid_df)}")
    print(f"基础模型: {args.model_name}")
    print(f"语料路径: {args.input_xlsx}")
    print("数据清洗策略: 仅过滤 delete_mark 与源/目标为空的样本；quality_score 不做最低分筛选。")
    print(f"GPU 数量: {gpu_count} | 启用多卡: {use_multi_gpu} | 精度: {precision_cfg['precision_name']}")
    print(
        f"有效总 Batch = per_device_train_batch_size({args.per_device_train_batch_size}) x GPU({active_gpu_count}) x grad_accum({args.gradient_accumulation_steps}) = {effective_global_batch}"
    )
    if resume_checkpoint:
        print(f"续训 checkpoint: {resume_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.src_lang = args.src_lang
    forced_bos_token_id = int(tokenizer.convert_tokens_to_ids(args.tgt_lang))

    train_ds, valid_ds = build_datasets(train_df, valid_df)

    def preprocess(batch):
        tokenizer.src_lang = args.src_lang
        model_inputs = tokenizer(
            batch["src_text"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["tgt_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    valid_ds = valid_ds.map(preprocess, batched=True, remove_columns=valid_ds.column_names)

    if precision_cfg["bf16"]:
        load_dtype = torch.bfloat16
    elif precision_cfg["fp16"]:
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.tie_word_embeddings = False
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        config=model_config,
        torch_dtype=load_dtype,
    )
    print(f"[兼容修复] 基础模型按 {str(load_dtype).replace('torch.', '')} 精度加载，降低初始化显存占用。")
    try:
        model.config.tie_word_embeddings = True
        model.tie_weights()
        print("[兼容修复] 已强制重新绑定 NLLB 共享词嵌入，避免 embedding 未共享导致显存翻倍。")
    except Exception as exc:
        print(f"[兼容修复] 重新绑定共享词嵌入失败: {exc}")
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_bos_token_id = forced_bos_token_id
        model.generation_config.max_length = int(args.generation_max_length)
        model.generation_config.num_beams = int(args.generation_num_beams)

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    bleu_metric, chrf_metric = safe_import_metrics()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def compute_metrics(eval_preds):
        if bleu_metric is None or chrf_metric is None:
            return {}

        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(labels != -100, labels, pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [normalize_text(x) for x in decoded_preds]
        decoded_labels = [normalize_text(x) for x in decoded_labels]

        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[x] for x in decoded_labels],
        )
        chrf = chrf_metric.compute(
            predictions=decoded_preds,
            references=[[x] for x in decoded_labels],
        )

        exact_match = 0.0
        if decoded_preds:
            exact_match = sum(int(p == r) for p, r in zip(decoded_preds, decoded_labels)) / len(decoded_preds)

        pred_lens = [max(1, len(x)) for x in decoded_preds]
        ref_lens = [max(1, len(x)) for x in decoded_labels]
        len_ratio = float(sum(pred_lens) / max(1, sum(ref_lens)))

        return {
            "bleu": float(bleu.get("score", 0.0)),
            "chrf": float(chrf.get("score", 0.0)),
            "exact_match": float(exact_match),
            "len_ratio": float(len_ratio),
        }

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if precision_cfg["fp16"] or precision_cfg["bf16"] else None,
    )

    steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, effective_global_batch)))
    total_train_steps = max(1, math.ceil(steps_per_epoch * float(args.num_train_epochs)))
    warmup_steps = max(0, int(total_train_steps * float(args.warmup_ratio)))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        generation_max_length=int(args.generation_max_length),
        generation_num_beams=int(args.generation_num_beams),
        eval_strategy="steps",
        eval_steps=int(args.save_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        fp16=precision_cfg["fp16"],
        bf16=precision_cfg["bf16"],
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrf" if bleu_metric is not None else "eval_loss",
        greater_is_better=True if bleu_metric is not None else False,
        seed=args.seed,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )
    best_checkpoint_callback = PersistBestCheckpointCallback(
        output_dir=args.output_dir,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=bool(training_args.greater_is_better),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics if bleu_metric is not None else None,
        callbacks=[best_checkpoint_callback],
    )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate(
        metric_key_prefix="eval",
        max_length=args.generation_max_length,
        num_beams=args.generation_num_beams,
    )
    print("验证结果:", json.dumps(eval_metrics, ensure_ascii=False, indent=2))

    best_checkpoint_zip = persist_best_checkpoint_artifacts(
        output_dir=args.output_dir,
        best_model_checkpoint=trainer.state.best_model_checkpoint,
        best_metric=trainer.state.best_metric,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=bool(training_args.greater_is_better),
        global_step=trainer.state.global_step,
    )

    metrics_out = {
        "train_result": dict(train_result.metrics),
        "eval_result": dict(eval_metrics),
        "gpu_count": gpu_count,
        "use_multi_gpu": use_multi_gpu,
        "effective_global_batch": effective_global_batch,
        "precision": precision_cfg["precision_name"],
        "resumed_from_checkpoint": resume_checkpoint,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "best_checkpoint_zip": best_checkpoint_zip,
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if args.merge_and_save:
        merged_dir = os.path.join(args.output_dir, "merged_model")
        merged_model = trainer.model.merge_and_unload()
        if getattr(merged_model, "generation_config", None) is not None:
            merged_model.generation_config.forced_bos_token_id = forced_bos_token_id
            merged_model.generation_config.max_length = int(args.generation_max_length)
            merged_model.generation_config.num_beams = int(args.generation_num_beams)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"已导出合并后的完整模型: {merged_dir}")

    sample_df = valid_df.head(30).copy()
    if not sample_df.empty:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        infer_model = trainer.model.to(device)
        infer_model.eval()
        preds = []
        for text in sample_df["src_text"].tolist():
            tokenizer.src_lang = args.src_lang
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_source_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = infer_model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=args.generation_max_length,
                    num_beams=args.generation_num_beams,
                )
            preds.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())
        sample_df["pred"] = preds
        sample_df.to_csv(os.path.join(args.output_dir, "valid_samples_with_pred.csv"), index=False, encoding="utf-8-sig")

    print("训练完成。请下载以下目录中的文件到本地：")
    print(f"1) Adapter: {adapter_dir}")
    print(f"2) Metrics: {os.path.join(args.output_dir, 'metrics_summary.json')}")
    if best_checkpoint_zip:
        print(f"3) Best checkpoint zip: {best_checkpoint_zip}")
    if args.merge_and_save:
        print(f"4) Merged model: {os.path.join(args.output_dir, 'merged_model')}")


if __name__ == "__main__":
    main()
