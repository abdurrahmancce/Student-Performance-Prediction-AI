import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import joblib
import pandas as pd

try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
except ImportError:
    print("This app needs ttkbootstrap. Install it with:\n"
          "    pip install ttkbootstrap")
    sys.exit(1)

from PIL import Image, ImageTk

# Paths (resolved relative to this file, so it works from any working dir)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"

FEATURES = [
    "attendance", "ct_marks", "mid_marks",
    "previous_semester_gpa", "study_hours_per_week",
]

FIELD_SPECS = [
    # key,                     label,                         min, max, step, decimals
    ("attendance",             "Attendance",                  0,   10,  0.5, 1),
    ("ct_marks",                "CT Marks",                    0,   10,  0.5, 1),
    ("mid_marks",                "Mid Marks",                   0,   30,  1.0, 1),
    ("previous_semester_gpa",  "Previous Semester GPA",       0,   4.0, 0.05, 2),
    ("study_hours_per_week",   "Study Hours / Week",          0,   40,  1.0, 1),
]

LIGHT_THEME = "flatly"
DARK_THEME = "darkly"


# Model wrapper

class ModelBundle:
    """Loads (and can reload) the trained model, scaler, and metadata."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.meta = None
        self.reload()

    @property
    def is_ready(self):
        return self.model is not None and self.scaler is not None

    def reload(self):
        try:
            self.model = joblib.load(MODELS_DIR / "best_model.joblib")
            self.scaler = joblib.load(MODELS_DIR / "scaler.joblib")
            with open(MODELS_DIR / "model_meta.json") as f:
                self.meta = json.load(f)
        except FileNotFoundError:
            self.model = None
            self.scaler = None
            self.meta = None

    def predict(self, values: dict):
        X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=FEATURES)
        pred = self.model.predict(X_scaled)[0]
        proba = (
            self.model.predict_proba(X_scaled)[0][1]
            if hasattr(self.model, "predict_proba") else float(pred)
        )
        return int(pred), float(proba)


# Reusable UI bits

class StatCard(tb.Frame):
    """Small metric card: big number + caption."""

    def __init__(self, master, title, value, bootstyle=SECONDARY, **kw):
        super().__init__(master, bootstyle=bootstyle, padding=16, **kw)
        self.value_lbl = tb.Label(
            self, text=value, font=("Segoe UI", 22, "bold"), bootstyle=(INVERSE, bootstyle)
        )
        self.value_lbl.pack(anchor="w")
        tb.Label(
            self, text=title, font=("Segoe UI", 10), bootstyle=(INVERSE, bootstyle)
        ).pack(anchor="w")

    def set_value(self, value):
        self.value_lbl.configure(text=value)


class LabeledSlider(tb.Frame):
    """A labeled slider + synced spinbox for precise numeric entry."""

    def __init__(self, master, label, lo, hi, step, decimals, default=None, **kw):
        super().__init__(master, **kw)
        self.decimals = decimals
        self.var = tb.DoubleVar(value=default if default is not None else (lo + hi) / 2)

        top = tb.Frame(self)
        top.pack(fill="x")
        tb.Label(top, text=label, font=("Segoe UI", 10, "bold")).pack(side="left")
        self.value_lbl = tb.Label(top, text=self._fmt(self.var.get()),
                                    font=("Segoe UI", 10), bootstyle=PRIMARY)
        self.value_lbl.pack(side="right")

        row = tb.Frame(self)
        row.pack(fill="x", pady=(4, 0))

        self.scale = tb.Scale(
            row, from_=lo, to=hi, orient="horizontal",
            variable=self.var, bootstyle=PRIMARY,
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.spin = tb.Spinbox(
            row, from_=lo, to=hi, increment=step, textvariable=self.var,
            width=7, bootstyle=PRIMARY,
        )
        self.spin.pack(side="right")

        # Keep the value label in sync no matter how the value changes
        # (scale drag, spinbox arrows, or typed entry).
        self.var.trace_add("write", lambda *args: self._refresh_label())

    def _fmt(self, val):
        return f"{float(val):.{self.decimals}f}"

    def _refresh_label(self):
        try:
            self.value_lbl.configure(text=self._fmt(self.var.get()))
        except tb.tk.TclError:
            pass  # ignore transient empty/invalid states while typing

    def get(self):
        return round(float(self.var.get()), self.decimals)


# Pages

class PredictPage(tb.Frame):
    def __init__(self, master, bundle: ModelBundle):
        super().__init__(master, padding=24)
        self.bundle = bundle
        self.sliders = {}

        tb.Label(self, text="Predict Student Outcome",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tb.Label(self, text="Enter the student's stats to get a Pass/Fail prediction.",
                  bootstyle=SECONDARY).pack(anchor="w", pady=(0, 16))

        body = tb.Frame(self)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        # Left: form card
        form_card = tb.Labelframe(body, text="  Student Data  ", padding=20, bootstyle=PRIMARY)
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        for key, label, lo, hi, step, decimals in FIELD_SPECS:
            slider = LabeledSlider(form_card, label, lo, hi, step, decimals)
            slider.pack(fill="x", pady=10)
            self.sliders[key] = slider

        self.predict_btn = tb.Button(
            form_card, text="  Predict Result  ", bootstyle=SUCCESS,
            command=self.on_predict,
        )
        self.predict_btn.pack(fill="x", pady=(18, 0), ipady=8)

        # Right: result card
        self.result_card = tb.Frame(body, bootstyle=SECONDARY, padding=20)
        self.result_card.grid(row=0, column=1, sticky="nsew")

        self.result_icon = tb.Label(
            self.result_card, text="🎓", font=("Segoe UI", 40), bootstyle=(INVERSE, SECONDARY)
        )
        self.result_icon.pack(pady=(10, 6))

        self.result_title = tb.Label(
            self.result_card, text="Awaiting input", font=("Segoe UI", 16, "bold"),
            bootstyle=(INVERSE, SECONDARY),
        )
        self.result_title.pack()

        self.result_sub = tb.Label(
            self.result_card, text="Fill the form and press Predict.",
            bootstyle=(INVERSE, SECONDARY), wraplength=220, justify="center"
        )
        self.result_sub.pack(pady=(4, 16))

        self.confidence_caption = tb.Label(
            self.result_card, text="Confidence", font=("Segoe UI", 9, "bold"),
            bootstyle=(INVERSE, SECONDARY),
        )
        self.confidence_caption.pack(anchor="w")

        # A light trough behind the bar keeps the fill visible against any
        # card color (a same-color fill on a same-color card would be
        # invisible, e.g. green-on-green).
        bar_track = tb.Frame(self.result_card, bootstyle=LIGHT, padding=2)
        bar_track.pack(fill="x", pady=(2, 4))
        self.confidence_bar = tb.Progressbar(
            bar_track, mode="determinate", bootstyle=SECONDARY, maximum=100
        )
        self.confidence_bar.pack(fill="x")

        self.confidence_lbl = tb.Label(
            self.result_card, text="—", bootstyle=(INVERSE, SECONDARY)
        )
        self.confidence_lbl.pack(anchor="e")

        self.model_note = tb.Label(
            self.result_card, text="", bootstyle=(INVERSE, SECONDARY), font=("Segoe UI", 8),
            wraplength=220, justify="center"
        )
        self.model_note.pack(side="bottom", pady=(16, 0))

        self.refresh_ready_state()

    def _style_card(self, style):
        """Recolor the result card and every child label consistently, so
        text always renders inverse (white) on top of the card's own solid
        background color instead of a mismatched default background."""
        self.result_card.configure(bootstyle=style)
        for widget in (self.result_icon, self.result_title, self.result_sub,
                        self.confidence_caption, self.confidence_lbl, self.model_note):
            widget.configure(bootstyle=(INVERSE, style))
        self.confidence_bar.configure(bootstyle=style)

    def refresh_ready_state(self):
        if self.bundle.is_ready:
            model_name = self.bundle.meta.get("model_name", "model")
            acc = self.bundle.meta["metrics"].get("test_accuracy")
            note = f"Using: {model_name}"
            if acc is not None:
                note += f"  (test accuracy {acc*100:.1f}%)"
            self.model_note.configure(text=note)
            self.predict_btn.configure(state="normal")
        else:
            self.model_note.configure(
                text="No trained model found. Go to 'Data & Training' and train one first."
            )
            self.predict_btn.configure(state="disabled")

    def on_predict(self):
        if not self.bundle.is_ready:
            self.refresh_ready_state()
            return

        values = {key: slider.get() for key, slider in self.sliders.items()}
        pred, proba = self.bundle.predict(values)

        confidence = proba if pred == 1 else (1 - proba)
        self.confidence_bar.configure(value=confidence * 100)
        self.confidence_lbl.configure(text=f"{confidence*100:.1f}%")

        if pred == 1:
            self.result_icon.configure(text="✅")
            self._style_card(SUCCESS)
            self.result_title.configure(text="PASS")
            self.result_sub.configure(text="This student is predicted to pass.")
        else:
            self.result_icon.configure(text="⚠️")
            self._style_card(DANGER)
            self.result_title.configure(text="FAIL")
            self.result_sub.configure(text="This student is predicted to fail. Consider early support.")


class InsightsPage(tb.Frame):
    def __init__(self, master, bundle: ModelBundle):
        super().__init__(master, padding=24)
        self.bundle = bundle
        self._img_refs = []  # keep PhotoImage refs alive

        header = tb.Frame(self)
        header.pack(fill="x")
        tb.Label(header, text="Model Insights", font=("Segoe UI", 18, "bold")).pack(side="left")
        tb.Button(header, text="⟳ Refresh", bootstyle=(SECONDARY, "outline"),
                   command=self.refresh).pack(side="right")

        self.cards_row = tb.Frame(self)
        self.cards_row.pack(fill="x", pady=(16, 20))

        self.stat_cards = {}
        specs = [
            ("accuracy", "Accuracy", SUCCESS),
            ("precision", "Precision", INFO),
            ("recall", "Recall", INFO),
            ("f1", "F1 Score", PRIMARY),
            ("roc_auc", "ROC-AUC", WARNING),
        ]
        for i, (key, label, style) in enumerate(specs):
            card = StatCard(self.cards_row, label, "—", bootstyle=style)
            card.grid(row=0, column=i, sticky="nsew", padx=6)
            self.cards_row.columnconfigure(i, weight=1)
            self.stat_cards[key] = card

        # scrollable image gallery
        gallery_label = tb.Label(self, text="Charts", font=("Segoe UI", 13, "bold"))
        gallery_label.pack(anchor="w")

        canvas_wrap = tb.Frame(self)
        canvas_wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.canvas = tb.Canvas(canvas_wrap, highlightthickness=0)
        scrollbar = tb.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=scrollbar.set)
        self.canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="bottom", fill="x")

        self.gallery_frame = tb.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        self.gallery_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.empty_lbl = tb.Label(
            self, text="", bootstyle=SECONDARY, font=("Segoe UI", 10)
        )
        self.empty_lbl.pack(pady=20)

        self.refresh()

    def refresh(self):
        self.bundle.reload()

        if self.bundle.is_ready:
            m = self.bundle.meta["metrics"]
            self.stat_cards["accuracy"].set_value(f"{m.get('test_accuracy', 0)*100:.1f}%")
            self.stat_cards["precision"].set_value(f"{m.get('test_precision', 0)*100:.1f}%")
            self.stat_cards["recall"].set_value(f"{m.get('test_recall', 0)*100:.1f}%")
            self.stat_cards["f1"].set_value(f"{m.get('test_f1', 0)*100:.1f}%")
            roc = m.get("test_roc_auc")
            self.stat_cards["roc_auc"].set_value(f"{roc*100:.1f}%" if roc is not None else "—")
            self.empty_lbl.configure(text="")
        else:
            for card in self.stat_cards.values():
                card.set_value("—")
            self.empty_lbl.configure(text="No trained model yet — train one from 'Data & Training'.")

        self._load_gallery()

    def _load_gallery(self):
        for w in self.gallery_frame.winfo_children():
            w.destroy()
        self._img_refs.clear()

        chart_files = [
            ("Confusion Matrix", REPORTS_DIR / "confusion_matrix.png"),
            ("Model Comparison", REPORTS_DIR / "model_comparison.png"),
            ("Feature Importance", REPORTS_DIR / "feature_importance.png"),
        ]

        any_found = False
        for i, (title, path) in enumerate(chart_files):
            if not path.exists():
                continue
            any_found = True
            card = tb.Frame(self.gallery_frame, bootstyle=SECONDARY, padding=10)
            card.grid(row=0, column=i, padx=8, pady=4, sticky="n")
            tb.Label(card, text=title, font=("Segoe UI", 10, "bold")).pack(pady=(0, 6))

            img = Image.open(path)
            img.thumbnail((340, 280))
            photo = ImageTk.PhotoImage(img)
            self._img_refs.append(photo)
            tb.Label(card, image=photo).pack()

        if not any_found:
            tb.Label(self.gallery_frame, text="No charts yet — train the model to generate them.",
                      bootstyle=SECONDARY).grid(row=0, column=0, padx=8, pady=8)


class TrainingPage(tb.Frame):
    def __init__(self, master, bundle: ModelBundle, on_trained):
        super().__init__(master, padding=24)
        self.bundle = bundle
        self.on_trained = on_trained
        self.msg_queue = queue.Queue()
        self.worker = None

        tb.Label(self, text="Data & Training", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tb.Label(
            self, text="Regenerate the synthetic dataset and/or retrain the model.",
            bootstyle=SECONDARY,
        ).pack(anchor="w", pady=(0, 16))

        btn_row = tb.Frame(self)
        btn_row.pack(fill="x")

        self.gen_btn = tb.Button(
            btn_row, text="🗂  Generate Dataset", bootstyle=INFO,
            command=self.run_generate,
        )
        self.gen_btn.pack(side="left", padx=(0, 10), ipady=6)

        self.train_btn = tb.Button(
            btn_row, text="🚀  Train Model", bootstyle=SUCCESS,
            command=self.run_train,
        )
        self.train_btn.pack(side="left", ipady=6)

        self.progress = tb.Progressbar(self, mode="indeterminate", bootstyle=(SUCCESS, STRIPED))
        self.progress.pack(fill="x", pady=(16, 6))

        self.status_lbl = tb.Label(self, text="Idle.", bootstyle=SECONDARY)
        self.status_lbl.pack(anchor="w")

        log_frame = tb.Labelframe(self, text="  Log  ", padding=10, bootstyle=SECONDARY)
        log_frame.pack(fill="both", expand=True, pady=(16, 0))

        self.log_text = tb.Text(log_frame, height=14, wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, side="left")
        log_scroll = tb.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set, state="disabled")

        self.after(150, self._poll_queue)

    def _log(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy, status=""):
        self.gen_btn.configure(state="disabled" if busy else "normal")
        self.train_btn.configure(state="disabled" if busy else "normal")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status:
            self.status_lbl.configure(text=status)

    def run_generate(self):
        self._run_script(SRC_DIR / "generate_dataset.py", "Generating dataset...")

    def run_train(self):
        self._run_script(SRC_DIR / "train_model.py", "Training model... this may take a moment.")

    def _run_script(self, script_path, status_text):
        if self.worker and self.worker.is_alive():
            return
        self._set_busy(True, status_text)
        self._log(f"$ python {script_path.name}")
        self.worker = threading.Thread(
            target=self._worker_run, args=(script_path,), daemon=True
        )
        self.worker.start()

    def _worker_run(self, script_path):
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                self.msg_queue.put(("log", line.rstrip()))
            proc.wait()
            if proc.returncode == 0:
                self.msg_queue.put(("done", script_path.name))
            else:
                self.msg_queue.put(("error", f"{script_path.name} exited with code {proc.returncode}"))
        except Exception as exc:
            self.msg_queue.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._log(f"✔ {payload} finished successfully.\n")
                    self._set_busy(False, "Idle.")
                    self.bundle.reload()
                    self.on_trained()
                elif kind == "error":
                    self._log(f"✘ Error: {payload}\n")
                    self._set_busy(False, "Failed — see log above.")
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)


# Main app

class App(tb.Window):
    def __init__(self):
        super().__init__(title="Student Performance Prediction AI", themename=LIGHT_THEME)
        self.geometry("1080x720")
        self.minsize(920, 620)
        self.is_dark = False

        self.bundle = ModelBundle()

        self._build_layout()

    def _build_layout(self):
        root = tb.Frame(self)
        root.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tb.Frame(root, bootstyle=DARK, padding=(16, 20))
        sidebar.pack(side="left", fill="y")

        tb.Label(
            sidebar, text="🎓 StudentAI", font=("Segoe UI", 15, "bold"),
            bootstyle=(INVERSE, DARK),
        ).pack(anchor="w", pady=(0, 30))

        self.nav_buttons = {}
        nav_items = [
            ("predict", "🎯  Predict"),
            ("insights", "📊  Model Insights"),
            ("training", "⚙️  Data & Training"),
        ]
        for key, label in nav_items:
            btn = tb.Button(
                sidebar, text=label, bootstyle=(LIGHT, "outline-toolbutton"),
                width=20, command=lambda k=key: self.show_page(k),
            )
            btn.pack(fill="x", pady=4)
            self.nav_buttons[key] = btn

        tb.Separator(sidebar, bootstyle=LIGHT).pack(fill="x", pady=16)

        self.theme_btn = tb.Button(
            sidebar, text="🌙  Dark Mode", bootstyle=(LIGHT, "outline-toolbutton"),
            width=20, command=self.toggle_theme,
        )
        self.theme_btn.pack(fill="x", side="bottom")

        # Content area (stacked pages)
        content = tb.Frame(root)
        content.pack(side="left", fill="both", expand=True)

        self.pages = {
            "predict": PredictPage(content, self.bundle),
            "insights": InsightsPage(content, self.bundle),
            "training": TrainingPage(content, self.bundle, on_trained=self._on_trained),
        }
        for page in self.pages.values():
            page.place(x=0, y=0, relwidth=1, relheight=1)

        self.show_page("predict")

    def show_page(self, key):
        self.pages[key].tkraise()
        for k, btn in self.nav_buttons.items():
            btn.configure(bootstyle=(SUCCESS if k == key else LIGHT, "outline-toolbutton"))

    def _on_trained(self):
        self.pages["predict"].refresh_ready_state()
        self.pages["insights"].refresh()

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.style.theme_use(DARK_THEME if self.is_dark else LIGHT_THEME)
        self.theme_btn.configure(text="☀️  Light Mode" if self.is_dark else "🌙  Dark Mode")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
