from datetime import datetime
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "kor_Hang"
TARGET_LANG = "zho_Hans"
CACHE_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_BASE_DIR = Path(__file__).resolve().parent / "translated_files"
SUPPORTED_FILE_TYPES = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
    ".srt",
    ".yaml",
    ".yml",
    ".ini",
}

tokenizer: Any = None
model: Any = None


def load_model() -> tuple[Any, Any]:
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(CACHE_DIR),
            src_lang=SOURCE_LANG,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            cache_dir=str(CACHE_DIR),
        )
    return tokenizer, model


def translate_texts(texts: list[str]) -> list[str]:
    current_tokenizer, current_model = load_model()
    inputs = current_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    generated_tokens = current_model.generate(
        **inputs,
        forced_bos_token_id=current_tokenizer.convert_tokens_to_ids(TARGET_LANG),
        max_length=256,
    )
    return current_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def translate_ko_to_zh(text: str) -> str:
    return translate_texts([text])[0]


def split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1]
    return line, ""


def preserve_padding(text: str, translated: str) -> str:
    leading_size = len(text) - len(text.lstrip())
    trailing_size = len(text) - len(text.rstrip())
    leading = text[:leading_size]
    trailing = text[len(text) - trailing_size :] if trailing_size else ""
    return f"{leading}{translated}{trailing}"


def chunk_list(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def translate_text_preserving_layout(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [text]

    parsed_lines: list[dict[str, str | bool]] = []
    contents_to_translate: list[str] = []
    for line in lines:
        body, ending = split_line_ending(line)
        if body.strip():
            contents_to_translate.append(body.strip())
            parsed_lines.append(
                {
                    "body": body,
                    "ending": ending,
                    "needs_translation": True,
                }
            )
        else:
            parsed_lines.append(
                {
                    "body": body,
                    "ending": ending,
                    "needs_translation": False,
                }
            )

    translated_contents: list[str] = []
    for chunk in chunk_list(contents_to_translate, 8):
        translated_contents.extend(translate_texts(chunk))

    translated_index = 0
    translated_lines: list[str] = []
    for parsed_line in parsed_lines:
        body = str(parsed_line["body"])
        ending = str(parsed_line["ending"])
        if bool(parsed_line["needs_translation"]):
            translated_body = translated_contents[translated_index]
            translated_index += 1
            translated_lines.append(preserve_padding(body, translated_body) + ending)
        else:
            translated_lines.append(body + ending)
    return "".join(translated_lines)


def read_text_file(file_path: Path) -> tuple[str, str]:
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return file_path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件编码：{file_path}")


class TranslatorUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("K2C Translator")
        self.root.geometry("980x760")
        self.pending_input_after_id: str | None = None
        self.is_translating = False
        self.batch_running = False
        self.last_translated_text = ""
        self.last_output_dir: Path | None = None

        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        top_row = ttk.Frame(container)
        top_row.pack(fill="x", pady=(0, 8))

        ttk.Label(top_row, text="韩文输入").pack(side="left")
        self.auto_translate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_row,
            text="输入后自动翻译",
            variable=self.auto_translate_var,
        ).pack(side="right")

        self.input_text = tk.Text(container, height=10, wrap="word")
        self.input_text.pack(fill="both", expand=True, pady=(6, 12))
        self.bind_text_widget(self.input_text, allow_paste=True)
        self.input_text.bind("<KeyRelease>", self.on_input_changed)

        button_row = ttk.Frame(container)
        button_row.pack(fill="x", pady=(0, 12))

        self.translate_button = ttk.Button(
            button_row,
            text="翻译为中文",
            command=self.start_translation,
        )
        self.translate_button.pack(side="left")

        ttk.Button(button_row, text="清空", command=self.clear_text).pack(
            side="left",
            padx=(8, 0),
        )
        ttk.Button(button_row, text="粘贴到输入框", command=self.paste_to_input).pack(
            side="left",
            padx=(8, 0),
        )
        ttk.Button(button_row, text="复制中文结果", command=self.copy_output).pack(
            side="left",
            padx=(8, 0),
        )
        ttk.Button(
            button_row,
            text="选择文件批量翻译",
            command=self.choose_files_for_batch_translation,
        ).pack(side="left", padx=(20, 0))
        ttk.Button(
            button_row,
            text="选择文件夹批量翻译",
            command=self.choose_folder_for_batch_translation,
        ).pack(side="left", padx=(8, 0))

        ttk.Label(container, text="中文输出").pack(anchor="w")
        self.output_text = tk.Text(container, height=10, wrap="word")
        self.output_text.pack(fill="both", expand=True, pady=(6, 12))
        self.bind_text_widget(self.output_text, allow_paste=False)

        self.status_var = tk.StringVar(
            value="准备就绪。首次翻译会下载模型，请耐心等待。"
        )
        ttk.Label(container, textvariable=self.status_var).pack(anchor="w")
        self.output_dir_var = tk.StringVar(value="批量输出目录：尚未生成")
        ttk.Label(container, textvariable=self.output_dir_var).pack(anchor="w", pady=(6, 0))

    def bind_text_widget(self, widget: tk.Text, allow_paste: bool) -> None:
        widget.bind("<Command-a>", lambda _event: self.select_all(widget))
        widget.bind("<Control-a>", lambda _event: self.select_all(widget))
        widget.bind("<Command-c>", lambda _event: self.copy_selected_text(widget))
        widget.bind("<Control-c>", lambda _event: self.copy_selected_text(widget))
        widget.bind("<Command-x>", lambda _event: self.cut_selected_text(widget))
        widget.bind("<Control-x>", lambda _event: self.cut_selected_text(widget))
        if allow_paste:
            widget.bind("<Command-v>", lambda _event: self.paste_text(widget))
            widget.bind("<Control-v>", lambda _event: self.paste_text(widget))
        widget.bind("<Button-2>", lambda event: self.show_context_menu(event, widget, allow_paste))
        widget.bind("<Button-3>", lambda event: self.show_context_menu(event, widget, allow_paste))

    def show_context_menu(
        self,
        event: tk.Event,
        widget: tk.Text,
        allow_paste: bool,
    ) -> str:
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="全选", command=lambda: self.select_all(widget))
        menu.add_command(label="复制", command=lambda: self.copy_selected_text(widget))
        if allow_paste:
            menu.add_command(label="粘贴", command=lambda: self.paste_text(widget))
            menu.add_command(label="剪切", command=lambda: self.cut_selected_text(widget))
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def select_all(self, widget: tk.Text) -> str:
        widget.tag_add(tk.SEL, "1.0", "end-1c")
        widget.mark_set(tk.INSERT, "1.0")
        widget.see(tk.INSERT)
        return "break"

    def copy_selected_text(self, widget: tk.Text) -> str:
        try:
            selected_text = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return "break"
        self.root.clipboard_clear()
        self.root.clipboard_append(selected_text)
        return "break"

    def cut_selected_text(self, widget: tk.Text) -> str:
        result = self.copy_selected_text(widget)
        if result == "break":
            try:
                widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                pass
        return "break"

    def paste_text(self, widget: tk.Text) -> str:
        try:
            clipboard_text = self.root.clipboard_get()
        except tk.TclError:
            return "break"
        widget.insert(tk.INSERT, clipboard_text)
        if widget is self.input_text:
            self.schedule_auto_translation()
        return "break"

    def paste_to_input(self) -> None:
        self.input_text.focus_set()
        self.paste_text(self.input_text)

    def copy_output(self) -> None:
        content = self.output_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("提示", "当前还没有可复制的中文结果。")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("中文结果已复制到剪贴板。")

    def clear_text(self) -> None:
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.last_translated_text = ""
        self.status_var.set("已清空输入与输出。")

    def on_input_changed(self, *_args: object) -> None:
        self.schedule_auto_translation()

    def schedule_auto_translation(self) -> None:
        if not self.auto_translate_var.get():
            return
        if self.pending_input_after_id is not None:
            self.root.after_cancel(self.pending_input_after_id)
        self.pending_input_after_id = self.root.after(700, self.start_translation)

    def start_translation(self) -> None:
        self.pending_input_after_id = None
        source_text = self.input_text.get("1.0", tk.END).strip()
        if not source_text:
            self.output_text.delete("1.0", tk.END)
            self.last_translated_text = ""
            self.status_var.set("请输入要翻译的韩文内容。")
            return
        if self.batch_running:
            messagebox.showinfo("提示", "当前正在进行批量翻译，请稍后再试。")
            return
        if self.is_translating:
            return
        if source_text == self.last_translated_text:
            return

        self.is_translating = True
        self.translate_button.config(state="disabled")
        self.status_var.set("正在翻译中，首次运行可能需要下载模型...")
        worker = threading.Thread(
            target=self._run_translation,
            args=(source_text,),
            daemon=True,
        )
        worker.start()

    def _run_translation(self, source_text: str) -> None:
        try:
            translated_text = translate_ko_to_zh(source_text)
            self.root.after(0, self._show_result, source_text, translated_text)
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))

    def _show_result(self, source_text: str, translated_text: str) -> None:
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", translated_text)
        self.last_translated_text = source_text
        self.is_translating = False
        self.translate_button.config(state="normal")
        self.status_var.set("翻译完成。")
        latest_input = self.input_text.get("1.0", tk.END).strip()
        if self.auto_translate_var.get() and latest_input and latest_input != source_text:
            self.schedule_auto_translation()

    def _show_error(self, error_message: str) -> None:
        self.is_translating = False
        self.translate_button.config(state="normal")
        self.status_var.set("翻译失败。")
        messagebox.showerror("翻译失败", error_message)

    def choose_files_for_batch_translation(self) -> None:
        if self.is_translating or self.batch_running:
            messagebox.showinfo("提示", "当前已有翻译任务在运行，请稍后再试。")
            return
        selected_files = filedialog.askopenfilenames(
            title="选择要翻译的文本文件",
            filetypes=(
                ("文本文件", "*.txt *.md *.csv *.json *.log *.srt *.yaml *.yml *.ini"),
                ("所有文件", "*.*"),
            ),
        )
        if not selected_files:
            return
        self.start_batch_translation([Path(file_path) for file_path in selected_files])

    def choose_folder_for_batch_translation(self) -> None:
        if self.is_translating or self.batch_running:
            messagebox.showinfo("提示", "当前已有翻译任务在运行，请稍后再试。")
            return
        selected_folder = filedialog.askdirectory(title="选择包含待翻译文件的文件夹")
        if not selected_folder:
            return
        file_paths = self.collect_supported_files(Path(selected_folder))
        if not file_paths:
            messagebox.showwarning("提示", "该文件夹中没有找到支持的文本文件。")
            return
        self.start_batch_translation(file_paths)

    def collect_supported_files(self, folder_path: Path) -> list[Path]:
        return sorted(
            [
                file_path
                for file_path in folder_path.rglob("*")
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_TYPES
            ]
        )

    def start_batch_translation(self, file_paths: list[Path]) -> None:
        self.batch_running = True
        self.translate_button.config(state="disabled")
        self.status_var.set(f"准备批量翻译，共 {len(file_paths)} 个文件...")
        worker = threading.Thread(
            target=self._run_batch_translation,
            args=(file_paths,),
            daemon=True,
        )
        worker.start()

    def _run_batch_translation(self, file_paths: list[Path]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_BASE_DIR / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            total_files = len(file_paths)
            translated_count = 0
            for index, file_path in enumerate(file_paths, start=1):
                self.root.after(
                    0,
                    self.status_var.set,
                    f"正在批量翻译 {index}/{total_files}: {file_path.name}",
                )
                original_text, encoding = read_text_file(file_path)
                translated_text = translate_text_preserving_layout(original_text)
                output_name = f"translated_{timestamp}_{file_path.name}"
                output_path = output_dir / output_name
                output_path.write_text(translated_text, encoding=encoding)
                translated_count += 1
            self.root.after(0, self._show_batch_result, output_dir, translated_count)
        except Exception as exc:
            self.root.after(0, self._show_batch_error, str(exc))

    def _show_batch_result(self, output_dir: Path, translated_count: int) -> None:
        self.batch_running = False
        self.translate_button.config(state="normal")
        self.last_output_dir = output_dir
        self.output_dir_var.set(f"批量输出目录：{output_dir}")
        self.status_var.set(f"批量翻译完成，共生成 {translated_count} 个文件。")
        messagebox.showinfo(
            "批量翻译完成",
            f"已生成 {translated_count} 个翻译文件。\n输出目录：\n{output_dir}",
        )

    def _show_batch_error(self, error_message: str) -> None:
        self.batch_running = False
        self.translate_button.config(state="normal")
        self.status_var.set("批量翻译失败。")
        messagebox.showerror("批量翻译失败", error_message)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TranslatorUI().run()
