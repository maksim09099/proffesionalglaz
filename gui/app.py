from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from src.iris_project.paths import DEFAULT_OUTPUT_DIR
from src.iris_project.pipeline import run_prediction


class IrisGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Iris Analyzer")
        self.geometry("900x600")
        self.selected_path: Path | None = None
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=12)
        self.result_var = tk.StringVar(value="Выберите изображение глаза")
        tk.Label(self, textvariable=self.result_var).pack()
        tk.Button(self, text="Выбрать изображение", command=self.select_image).pack(pady=4)
        tk.Button(self, text="Запустить анализ", command=self.predict).pack(pady=4)

    def select_image(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not selected:
            return
        self.selected_path = Path(selected)
        self._show_image(self.selected_path)

    def predict(self) -> None:
        if self.selected_path is None:
            messagebox.showwarning("Внимание", "Сначала выберите изображение")
            return
        result = run_prediction(self.selected_path, DEFAULT_OUTPUT_DIR)
        self.result_var.set(
            f"Center=({result['center']['x']}, {result['center']['y']}), "
            f"radius={result['radius']}, confidence={result['confidence']:.2f}"
        )
        self._show_image(Path(result["annotated_image"]))

    def _show_image(self, path: Path) -> None:
        image = Image.open(path).resize((420, 320))
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo


if __name__ == "__main__":
    IrisGui().mainloop()
