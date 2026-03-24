import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from filters import (
    adaptive_threshold,
    adaptive_threshold_inv,
    black_hat_transform_filter,
    bilateral_filter_threshold,
    binary_threshold,
    canny_edge_detection,
    clahe_filter,
    contour_geometry_cleanup_filter,
    dilation_filter,
    erosion_dilation_filter,
    erosion_filter,
    gaussian_blur_filter,
    gamma_blackhat_otsu_filter,
    gamma_darken_filter,
    gamma_correction,
    high_contrast_transition_filter,
    hsv_color_filter,
    inverse_filter,
    lcd_digit_pipeline_filter,
    morphological_operations,
    otsu_binarization_filter,
    straight_diagonal_lines_filter,
)


FILTER_CHOICES = [
    "Binary Threshold",
    "Adaptive Threshold (Inverted)",
    "Canny Edge Detection",
    "Gamma Darken",
    "Black Hat Transform",
    "Otsu Binarization",
    "Gamma + Black Hat + Otsu",
    "High-Contrast Transitions (B/W)",
    "Contour Geometry Cleanup",
    "Morphological Operations",
    "HSV Color Filter",
    "Bilateral Filter + Threshold",
    "Gaussian Blur",
    "CLAHE",
    "Inverse",
    "Straight/Diagonal Lines",
    "LCD Digit Pipeline (5-Step)",
    "Erosion",
    "Dilation",
]


class PreprocessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scoreboard Preprocessing GUI")
        self.root.geometry("1400x800")

        self.original_image = None
        self.current_image_path = None
        self.filter_pipeline = []
        self.dataset_images = []

        self._build_ui()
        self._load_initial_image()

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")

        ttk.Button(controls, text="Open Image", command=self.open_image).grid(row=0, column=0, padx=5)
        ttk.Button(controls, text="New Random Image", command=self.load_random_image).grid(row=0, column=1, padx=5)

        self.filter_var = tk.StringVar(value=FILTER_CHOICES[0])
        ttk.Combobox(
            controls,
            textvariable=self.filter_var,
            values=FILTER_CHOICES,
            state="readonly",
            width=30,
        ).grid(row=0, column=2, padx=5)

        ttk.Label(controls, text="Kernel").grid(row=0, column=3, padx=(10, 3))
        self.kernel_var = tk.StringVar(value="3")
        ttk.Entry(controls, textvariable=self.kernel_var, width=6).grid(row=0, column=4, padx=3)

        ttk.Label(controls, text="Iterations").grid(row=0, column=5, padx=(10, 3))
        self.iter_var = tk.StringVar(value="1")
        ttk.Entry(controls, textvariable=self.iter_var, width=6).grid(row=0, column=6, padx=3)

        ttk.Label(controls, text="Dilate Iter").grid(row=0, column=7, padx=(10, 3))
        self.dilate_iter_var = tk.StringVar(value="1")
        ttk.Entry(controls, textvariable=self.dilate_iter_var, width=6).grid(row=0, column=8, padx=3)

        ttk.Button(controls, text="Add Filter", command=self.add_filter).grid(row=0, column=9, padx=8)
        ttk.Button(controls, text="Remove Selected", command=self.remove_selected).grid(row=0, column=10, padx=5)
        ttk.Button(controls, text="Clear Filters", command=self.clear_filters).grid(row=0, column=11, padx=5)

        pipeline_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        pipeline_frame.pack(fill="x")
        ttk.Label(pipeline_frame, text="Filter Pipeline (applied top to bottom):").pack(anchor="w")
        self.pipeline_listbox = tk.Listbox(pipeline_frame, height=6)
        self.pipeline_listbox.pack(fill="x", pady=5)

        images_frame = ttk.Frame(self.root, padding=10)
        images_frame.pack(fill="both", expand=True)

        left_frame = ttk.LabelFrame(images_frame, text="Original")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        right_frame = ttk.LabelFrame(images_frame, text="Processed")
        right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.original_label = ttk.Label(left_frame)
        self.original_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.processed_label = ttk.Label(right_frame)
        self.processed_label.pack(fill="both", expand=True, padx=10, pady=10)

    def _collect_dataset_images(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs = [
            os.path.join(script_dir, "..", "data", "data"),
            os.path.join(script_dir, "..", "data"),
        ]
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        for candidate in candidate_dirs:
            candidate = os.path.normpath(candidate)
            if not os.path.isdir(candidate):
                continue

            collected = []
            for root, _, files in os.walk(candidate):
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in image_extensions:
                        collected.append(os.path.join(root, file_name))

            if collected:
                return collected

        return []

    def _set_image_from_path(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return False

        self.original_image = image
        self.current_image_path = image_path
        self.update_views()
        return True

    def _load_initial_image(self):
        self.dataset_images = self._collect_dataset_images()
        if self.dataset_images:
            self.load_random_image()
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(script_dir, "scoreboard.png")
        if os.path.exists(default_path):
            self._set_image_from_path(default_path)

    def load_random_image(self):
        if not self.dataset_images:
            self.dataset_images = self._collect_dataset_images()

        if not self.dataset_images:
            messagebox.showwarning(
                "Data Folder Not Found",
                "No images were found in the data folder.",
            )
            return

        selected_path = random.choice(self.dataset_images)
        if not self._set_image_from_path(selected_path):
            messagebox.showerror("Error", "Could not load the selected random image.")

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")],
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not load selected image.")
            return

        self.original_image = image
        self.current_image_path = file_path
        self.filter_pipeline = []
        self.pipeline_listbox.delete(0, tk.END)
        self.update_views()

    def _safe_positive_int(self, text, default_value):
        try:
            value = int(text)
            return value if value > 0 else default_value
        except ValueError:
            return default_value

    def add_filter(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Load an image before adding filters.")
            return

        filter_name = self.filter_var.get()
        kernel = self._safe_positive_int(self.kernel_var.get(), 3)
        iterations = self._safe_positive_int(self.iter_var.get(), 1)
        dilate_iterations = self._safe_positive_int(self.dilate_iter_var.get(), 1)

        config = {
            "name": filter_name,
            "kernel": kernel,
            "iterations": iterations,
            "dilate_iterations": dilate_iterations,
        }
        self.filter_pipeline.append(config)
        self._refresh_pipeline_list()
        self.update_views()

    def remove_selected(self):
        selected = self.pipeline_listbox.curselection()
        if not selected:
            return

        del self.filter_pipeline[selected[0]]
        self._refresh_pipeline_list()
        self.update_views()

    def clear_filters(self):
        self.filter_pipeline = []
        self._refresh_pipeline_list()
        self.update_views()

    def _refresh_pipeline_list(self):
        self.pipeline_listbox.delete(0, tk.END)
        for i, item in enumerate(self.filter_pipeline, start=1):
            if item["name"] == "Erosion + Dilation (Combined)":
                text = (
                    f"{i}. {item['name']} "
                    f"(k={item['kernel']}, e={item['iterations']}, d={item['dilate_iterations']})"
                )
            elif item["name"] == "Gaussian Blur":
                text = f"{i}. {item['name']} (k={item['kernel']})"
            elif item["name"] == "Black Hat Transform":
                text = f"{i}. {item['name']} (k={item['kernel']})"
            elif item["name"] in {"Erosion", "Dilation"}:
                text = f"{i}. {item['name']} (k={item['kernel']}, it={item['iterations']})"
            else:
                text = f"{i}. {item['name']}"
            self.pipeline_listbox.insert(tk.END, text)

    def _apply_single_filter(self, image, filter_config):
        name = filter_config["name"]
        kernel = filter_config["kernel"]
        iterations = filter_config["iterations"]
        dilate_iterations = filter_config["dilate_iterations"]

        if name == "Binary Threshold":
            return binary_threshold(image)
        if name == "Adaptive Threshold":
            return adaptive_threshold(image)
        if name == "Adaptive Threshold (Inverted)":
            return adaptive_threshold_inv(image)
        if name == "Canny Edge Detection":
            return canny_edge_detection(image)
        if name == "Gamma Darken":
            return gamma_darken_filter(image, gamma=1.6)
        if name == "Black Hat Transform":
            odd_kernel = kernel if kernel % 2 == 1 else kernel + 1
            odd_kernel = max(5, odd_kernel)
            return black_hat_transform_filter(image, kernel_size=(odd_kernel, odd_kernel))
        if name == "Otsu Binarization":
            return otsu_binarization_filter(image)
        if name == "Gamma + Black Hat + Otsu":
            odd_kernel = kernel if kernel % 2 == 1 else kernel + 1
            odd_kernel = max(5, odd_kernel)
            return gamma_blackhat_otsu_filter(image, gamma=1.6, blackhat_kernel=(odd_kernel, odd_kernel))
        if name == "High-Contrast Transitions (B/W)":
            return high_contrast_transition_filter(image)
        if name == "Contour Geometry Cleanup":
            return contour_geometry_cleanup_filter(image)
        if name == "Morphological Operations":
            return morphological_operations(image)
        if name == "HSV Color Filter":
            return hsv_color_filter(image)
        if name == "Bilateral Filter + Threshold":
            return bilateral_filter_threshold(image)
        if name == "Gaussian Blur":
            # Gaussian blur requires odd kernel dimensions.
            odd_kernel = kernel if kernel % 2 == 1 else kernel + 1
            odd_kernel = max(3, odd_kernel)
            return gaussian_blur_filter(image, (odd_kernel, odd_kernel))
        if name == "CLAHE":
            return clahe_filter(image)
        if name == "Gamma Correction":
            return gamma_correction(image)
        if name == "Inverse":
            return inverse_filter(image)
        if name == "Straight/Diagonal Lines":
            return straight_diagonal_lines_filter(image)
        if name == "LCD Digit Pipeline (5-Step)":
            return lcd_digit_pipeline_filter(image)
        if name == "Erosion":
            return erosion_filter(image, (kernel, kernel), iterations)
        if name == "Dilation":
            return dilation_filter(image, (kernel, kernel), iterations)
        if name == "Erosion + Dilation (Combined)":
            return erosion_dilation_filter(image, (kernel, kernel), iterations, dilate_iterations)
        return image

    def _display_image(self, image, target_label, max_size=(640, 640)):
        if image is None:
            target_label.configure(image="")
            target_label.image = None
            return

        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = rgb_image.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h, 1.0)
        resized = cv2.resize(rgb_image, (int(w * scale), int(h * scale)))

        pil_image = Image.fromarray(resized)
        tk_image = ImageTk.PhotoImage(pil_image)
        target_label.configure(image=tk_image)
        target_label.image = tk_image

    def update_views(self):
        if self.original_image is None:
            return

        processed = self.original_image.copy()
        for filter_config in self.filter_pipeline:
            processed = self._apply_single_filter(processed, filter_config)

        self._display_image(self.original_image, self.original_label)
        self._display_image(processed, self.processed_label)


def main():
    root = tk.Tk()
    app = PreprocessingGUI(root)
    if app.original_image is None:
        messagebox.showinfo(
            "Info",
            "No images found in the data folder and no default image found. Use 'Open Image' to load one.",
        )
    root.mainloop()


if __name__ == "__main__":
    main()
