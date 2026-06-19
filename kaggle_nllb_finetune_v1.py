import argparse
import json
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset


DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_INPUT_XLSX = "/kaggle/input/k2c-corpus/Cleaned_NEW-CORPUS(125650ea)-merged-20260609 -taged_20260612_123200.xlsx"
DEFAULT_OUTPUT_DIR = "/kaggle/working/nllb_k2c_ko2zh_lora"
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
    return CorpusConfig(src_col=src_col, tgt_col=tgt_col, tag_col=tag_col, quality_col=quality_col, delete_mark_col=delete_mark_col)


def parse_delete_mark(value) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "delete", "deleted", "drop"}


def parse_quality(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def boost_automatic_rows(df: pd.DataFrame, tag_col: str | None, factor: int) -> pd.DataFrame:
    if factor <= 1 or not tag_col or tag_col not in df.columns:
        return df
    tag_s = df[tag_col].fillna("").astype(str).str.lower()
    mask = tag_s.str.contains(r"automatic|automation|auto|系统|按钮|界面|程序|功能", regex=True)
    boosted = df.loc[mask].copy()
    if boosted.empty:
        return df
    extras = [boosted.copy() for _ in range(factor - 1)]
    return pd.concat([df] + extras, ignore_index=True)


def load_parallel_corpus(args) -> pd.DataFrame:
    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet_name or 0)
    cfg = detect_columns(df)

    work = df.copy()
    work[cfg.src_col] = work[cfg.src_col].map(normalize_text)
    work[cfg.tgt_col] = work[cfg.tgt_col].map(normalize_text)
    work = work[(work[cfg.src_col] != "") & (work[cfg.tgt_col] != "")]

    if cfg.delete_mark_col:
        work = work[~work[cfg.delete_mark_col].map(parse_delete_mark)]

    if cfg.quality_col:
        # 按当前训练策略，quality_score 仅作为分析字段保留，不参与最低分过滤。
        # 也就是说：除 delete_mark 外，其余非空平行样本都保留给模型训练。
        pass

    if cfg.tag_col and args.keep_tags:
        keep_tags = [t.strip().lower() for t in args.keep_tags.split(",") if t.strip()]
        if keep_tags:
            tag_s = work[cfg.tag_col].fillna("").astype(str).str.lower()
            work = work[tag_s.apply(lambda x: any(t in x for t in keep_tags))]

    work = work.drop_duplicates(subset=[cfg.src_col, cfg.tgt_col]).reset_index(drop=True)
    work = boost_automatic_rows(work, cfg.tag_col, args.automatic_boost)

    out = pd.DataFrame(
        {
            "src_text": work[cfg.src_col].astype(str),
            "tgt_text": work[cfg.tgt_col].astype(str),
            "tag": work[cfg.tag_col].astype(str) if cfg.tag_col else "",
        }
    )
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    return out


def train_val_split(df: pd.DataFrame, valid_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_size = max(1, int(len(df) * valid_ratio))
    if len(df) <= valid_size:
        valid_size = max(1, len(df) // 10)
    valid_df = df.iloc[:valid_size].reset_index(drop=True)
    train_df = df.iloc[valid_size:].reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError("训练集为空，请降低 valid_ratio 或增加语料。")
    return train_df, valid_df


def build_datasets(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[Dataset, Dataset]:
    return Dataset.from_pandas(train_df, preserve_index=False), Dataset.from_pandas(valid_df, preserve_index=False)


def main():
    parser = argparse.ArgumentParser(description="Kaggle: 微调 NLLB-200-distilled-600M (KO -> ZH修正)")
    parser.add_argument("--input-xlsx", default=DEFAULT_INPUT_XLSX)
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--src-lang", default=DEFAULT_SRC_LANG)
    parser.add_argument("--tgt-lang", default=DEFAULT_TGT_LANG)
    parser.add_argument("--min-quality-score", type=float, default=80.0)
    parser.add_argument("--keep-tags", default="")
    parser.add_argument("--automatic-boost", type=int, default=2)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--merge-and-save", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_parallel_corpus(args)
    train_df, valid_df = train_val_split(df, args.valid_ratio, args.seed)

    train_df.to_csv(os.path.join(args.output_dir, "train_preview.csv"), index=False, encoding="utf-8-sig")
    valid_df.to_csv(os.path.join(args.output_dir, "valid_preview.csv"), index=False, encoding="utf-8-sig")

    print(f"总语料: {len(df)}")
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(valid_df)}")
    print(f"基础模型: {args.model_name}")
    print("数据清洗策略: 仅过滤 delete_mark 与源/目标为空的样本；quality_score 不做最低分筛选。")
    if args.min_quality_score != 80.0:
        print(f"提示: 当前脚本已忽略 --min-quality-score={args.min_quality_score}，保留所有未被 delete_mark 删除的非空样本。")

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.src_lang = args.src_lang

    train_ds, valid_ds = build_datasets(train_df, valid_df)

    def preprocess(batch):
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

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
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

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        fp16=True,
        bf16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(max_length=args.max_target_length)
    print("验证结果:", json.dumps(eval_metrics, ensure_ascii=False, indent=2))

    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if args.merge_and_save:
        merged_dir = os.path.join(args.output_dir, "merged_model")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"已导出合并后的完整模型: {merged_dir}")

    sample_df = valid_df.head(20).copy()
    if not sample_df.empty:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        infer_model = trainer.model.to(device)
        infer_model.eval()
        forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang]
        preds = []
        for text in sample_df["src_text"].tolist():
            tokenizer.src_lang = args.src_lang
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_source_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = infer_model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=args.max_target_length,
                )
            preds.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())
        sample_df["pred"] = preds
        sample_df.to_csv(os.path.join(args.output_dir, "valid_samples_with_pred.csv"), index=False, encoding="utf-8-sig")

    print("训练完成。请下载以下目录中的文件到本地：")
    print(f"1) Adapter: {adapter_dir}")
    if args.merge_and_save:
        print(f"2) Merged model: {os.path.join(args.output_dir, 'merged_model')}")


if __name__ == "__main__":
    main()
