import argparse
import json
import os
import shutil
import tempfile
import zipfile

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少依赖 torch。\n"
        "请先执行:\n"
        '  C:/Users/zhengchunji/AppData/Local/Programs/Python/Python313/python.exe -m pip install torch'
    ) from exc

try:
    from peft import PeftModel
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少依赖 peft。\n"
        "请先执行:\n"
        '  C:/Users/zhengchunji/AppData/Local/Programs/Python/Python313/python.exe -m pip install peft transformers accelerate sentencepiece safetensors'
    ) from exc

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少依赖 transformers。\n"
        "请先执行:\n"
        '  C:/Users/zhengchunji/AppData/Local/Programs/Python/Python313/python.exe -m pip install transformers accelerate sentencepiece safetensors'
    ) from exc


DEFAULT_BASE_MODEL_DIR = r"D:\K2C_Translator_Plus\models\models--facebook--nllb-200-3.3B"
DEFAULT_SRC_LANG = "kor_Hang"
DEFAULT_TGT_LANG = "zho_Hans"
DEFAULT_WORK_DIR = r"D:\K2C_Translator_Plus\translated models\NLLB200-33B"
DEFAULT_CHECKPOINT_PATH = os.path.join(DEFAULT_WORK_DIR, "checkpoint-4000")
DEFAULT_MERGED_OUTPUT_DIR = os.path.join(DEFAULT_WORK_DIR, "merged_model")


def has_model_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    try:
        names = set(os.listdir(path))
    except Exception:
        return False
    has_config = "config.json" in names
    has_tokenizer = (
        "tokenizer.json" in names
        or "tokenizer_config.json" in names
        or "sentencepiece.bpe.model" in names
    )
    has_weights = (
        "pytorch_model.bin" in names
        or "pytorch_model.bin.index.json" in names
        or any(name.startswith("pytorch_model-") and name.endswith(".bin") for name in names)
        or "model.safetensors" in names
        or "model.safetensors.index.json" in names
        or any(name.startswith("model-") and name.endswith(".safetensors") for name in names)
    )
    return has_config and has_tokenizer and has_weights


def resolve_base_model_dir(model_name_or_dir: str) -> str:
    model_dir = os.path.normpath(model_name_or_dir)
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

    for root, dirs, _files in os.walk(model_dir):
        for d in dirs:
            cand = os.path.join(root, d)
            if has_model_files(cand):
                return cand

    raise RuntimeError(f"无法定位 NLLB-200-3.3B 基础模型目录: {model_name_or_dir}")


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def resolve_default_checkpoint_path() -> str:
    preferred_paths = [
        os.path.join(DEFAULT_WORK_DIR, "checkpoint-4000"),
        os.path.join(DEFAULT_WORK_DIR, "best_checkpoint.json"),
        os.path.join(DEFAULT_WORK_DIR, "best_checkpoint.txt"),
        os.path.join(DEFAULT_WORK_DIR, "best_checkpoint.zip"),
    ]
    for path in preferred_paths:
        if os.path.exists(path):
            return path
    return preferred_paths[0]


def resolve_checkpoint_source(path_or_record: str) -> tuple[str, str | None]:
    raw = os.path.normpath(path_or_record)
    cleanup_dir = None

    if os.path.isdir(raw):
        return raw, cleanup_dir

    if not os.path.exists(raw):
        raise FileNotFoundError(f"找不到 checkpoint 路径: {raw}")

    lower_name = os.path.basename(raw).lower()
    if lower_name == "best_checkpoint.json":
        payload = json.loads(_read_text_file(raw))
        best_dir = str(payload.get("best_model_checkpoint", "")).strip()
        if best_dir and os.path.isdir(best_dir):
            return os.path.normpath(best_dir), cleanup_dir
        raise RuntimeError(f"best_checkpoint.json 中没有有效的 best_model_checkpoint: {raw}")

    if lower_name == "best_checkpoint.txt":
        best_dir = _read_text_file(raw)
        if best_dir and os.path.isdir(best_dir):
            return os.path.normpath(best_dir), cleanup_dir
        raise RuntimeError(f"best_checkpoint.txt 指向的目录不存在: {best_dir}")

    if lower_name.endswith(".zip"):
        cleanup_dir = tempfile.mkdtemp(prefix="nllb33b_best_ckpt_")
        with zipfile.ZipFile(raw, "r") as zf:
            zf.extractall(cleanup_dir)
        checkpoint_dirs = []
        for root, dirs, files in os.walk(cleanup_dir):
            if "adapter_config.json" in files or "adapter_model.safetensors" in files or "adapter_model.bin" in files:
                checkpoint_dirs.append(root)
            for d in dirs:
                if d.startswith("checkpoint-"):
                    checkpoint_dirs.append(os.path.join(root, d))
        if checkpoint_dirs:
            checkpoint_dirs = sorted(set(os.path.normpath(x) for x in checkpoint_dirs), key=len)
            return checkpoint_dirs[0], cleanup_dir
        raise RuntimeError(f"zip 中没有找到可用的 checkpoint/adpater 目录: {raw}")

    raise RuntimeError(f"不支持的 checkpoint 输入: {raw}")


def merge_checkpoint_to_model(
    base_model_dir: str,
    checkpoint_dir: str,
    merged_output_dir: str,
    src_lang: str,
    tgt_lang: str,
    device: str,
) -> str:
    os.makedirs(merged_output_dir, exist_ok=True)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"[1/4] 加载基础模型: {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.src_lang = src_lang
    forced_bos_token_id = int(tokenizer.convert_tokens_to_ids(tgt_lang))

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if device.startswith("cuda"):
        model = model.to(device)

    print(f"[2/4] 加载 LoRA/QLoRA checkpoint: {checkpoint_dir}")
    model = PeftModel.from_pretrained(model, checkpoint_dir)

    print("[3/4] 合并 adapter 到基础模型 ...")
    merged_model = model.merge_and_unload()
    if getattr(merged_model, "generation_config", None) is not None:
        merged_model.generation_config.forced_bos_token_id = forced_bos_token_id
        merged_model.generation_config.max_length = 96
        merged_model.generation_config.num_beams = 1

    print(f"[4/4] 保存 merged_model: {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    return merged_output_dir


def build_parser():
    parser = argparse.ArgumentParser(description="本地合成 NLLB-200-3.3B FT merged_model")
    parser.add_argument("--base-model-dir", default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument(
        "--checkpoint-path",
        default=resolve_default_checkpoint_path(),
        help="可直接使用 checkpoint-4000 目录；也支持 best_checkpoint.json、best_checkpoint.txt、best_checkpoint.zip 作为定位入口",
    )
    parser.add_argument("--merged-output-dir", default=DEFAULT_MERGED_OUTPUT_DIR)
    parser.add_argument("--src-lang", default=DEFAULT_SRC_LANG)
    parser.add_argument("--tgt-lang", default=DEFAULT_TGT_LANG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cleanup_dir = None
    try:
        base_model_dir = resolve_base_model_dir(args.base_model_dir)
        checkpoint_dir, cleanup_dir = resolve_checkpoint_source(args.checkpoint_path)
        merged_model_dir = merge_checkpoint_to_model(
            base_model_dir=base_model_dir,
            checkpoint_dir=checkpoint_dir,
            merged_output_dir=args.merged_output_dir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            device=args.device,
        )
        print(f"\n[完成] merged_model 已输出到: {merged_model_dir}")
    finally:
        if cleanup_dir and os.path.isdir(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
