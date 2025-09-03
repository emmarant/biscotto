import numpy as np
import time
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interactive, interactive_output
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from google.colab import data_table
from midap.midap_jupyter.segmentation_jupyter import SegmentationJupyter
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import os, subprocess
import tensorflow as tf



def select_seg_models(self,df):
    """
    Display searchable table with available models. 
    Table contents are read from a CSV file and passed into this function as a Pandas dataframe.
    A widget with a (string-searchable) list of model names allows the user to choose 2 or more models.
    'Apply' will select the desired models with no further action - for later use
    'Apply and run' will select desired models and do the segmentation on the images selected in earlier step
    INPUT: a dataframe containing the list of available models and their specifications
    OUTPUT: arrays of segmentation intances - masks - of each chosen model and for each selected image.
    """
    self.get_segmentation_models()
    display(data_table.DataTable(df, include_index=False, num_rows_per_page=10))

    all_names = df["Model Name"].astype(str).tolist()

    search = widgets.Text(placeholder="filter models with ... (substring match)", layout=widgets.Layout(width="40%"))
    sel    = widgets.SelectMultiple(options=sorted(all_names), rows=12, description="Select")
    btn_all   = widgets.Button(description="Select all (filtered)", tootip='Select all models matching filter keywords', icon="check-square")
    btn_clear  = widgets.Button(description="Clear", tooltip='Clear selection',icon="trash")
    btn_apply = widgets.Button(description="Apply selection",tooltip='Select models for later use',icon="tasks")
    btn_applyrun   = widgets.Button(description="Apply & run", tooltip='Select models and run segmentation',icon="play-circle",button_style="primary")
    out = widgets.Output()

    def refresh_options(_=None):
        q = search.value.lower().strip()
        opts = [n for n in all_names if q in n.lower()] if q else sorted(all_names)
        current = set(sel.value)
        sel.options = opts
        sel.value = tuple([o for o in opts if o in current])

    search.observe(refresh_options, names="value")
    refresh_options()

    def on_all_clicked(_):
        sel.value = tuple(sel.options)

    def on_none_clicked(_):
        sel.value = ()

    btn_all.on_click(on_all_clicked)
    btn_clear.on_click(on_none_clicked)


    def _apply_selection(run_now=False):
        selected = set(sel.value)

        self.model_checkboxes = {
            name: widgets.Checkbox(value=(name in selected), indent=False, layout=widgets.Layout(width="1px", height="1px"))
            for name in all_names
        }

        with out:
            clear_output()
            print(f"Selected {len(selected)} model(s):")
            for n in sorted(selected):
                print("  •", n)

        if run_now:
            self.select_segmentation_models()
            self.run_all_chosen_models_timing()

    def on_apply_clicked(_):
        _apply_selection(run_now=False)

    def on_applyrun_clicked(_):
        _apply_selection(run_now=True)

    btn_apply.on_click(on_apply_clicked)
    btn_applyrun.on_click(on_applyrun_clicked)

    sel_ui = widgets.VBox([
        widgets.HBox([search, btn_all, btn_clear, btn_apply, btn_applyrun]),
        sel,
        out
    ])

    display(sel_ui)
    


def draw_seg_inst_outlines(self,ax, inst_labels, color="yellow", lw=1.5):
        inst = np.asarray(inst_labels)
        if inst.ndim == 3 and inst.shape[-1] == 2:  
            inst = inst[..., 0]
        labels = np.unique(inst)
        labels = labels[labels != 0]  

        for lab in labels:
            ax.contour(inst == lab, levels=[0.5], colors=[color], linewidths=lw)



def compare_and_plot_segmentations(self):
        """
        Modification of MIDAP's sj.compare_segmentations() method: includes contour and overlay plots.
        Also minor changes in plot organization.
        Visualises:
          1. raw image
          2. composite image: segmentation overlap map
          3. instance segmentation of model-1
          4. instance segmentation of model-2
          5. overlay raw image + outlines of instance segmentation of model-1
          6. overlay raw image + outlines of instance segmentation of model-2
          7. bar-plot of the per-model mean semantic disagreement scores
             with standard deviation as error bars.
        """

        # ----------------------------------------------------------------
        # prepare bar-plot data (only once)
        # ----------------------------------------------------------------
        if not hasattr(self, "model_diff_scores"):
            self.model_diff_scores = self.compute_model_diff_scores()


        def f(a, b, c):
            fig = plt.figure(figsize=(12, 16))
            fig.canvas.header_visible = False

            gs = fig.add_gridspec(
                nrows=4, ncols=2,
                height_ratios=[1.0, 1.0, 1.0, 0.8],
                hspace=0.16, wspace=0.08
            )

        # ---- Row 1 (raw and composite) ----
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1],sharex=ax0, sharey=ax0)

        # ---- Row 2 (two segmentations) ----
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
            ax3 = fig.add_subplot(gs[1, 1], sharex=ax0, sharey=ax0)

        # ---- Row 3 (two segmentations) ----
            ax4 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)
            ax5 = fig.add_subplot(gs[2, 1], sharex=ax0, sharey=ax0)

        # ---- Row 4 (bar plot) ----
            ax6 = fig.add_subplot(gs[3, :])

        # ---- raw image ----
            raw = self.imgs_cut[int(c)]
            ax0.imshow(raw, cmap="gray")
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_title("Raw image",fontsize=15)


        # ---- composite segmentation image ----
            inst_a = self.dict_all_models_label[a][int(c)]
            inst_a = np.asarray(inst_a)
            if inst_a.ndim == 3 and inst_a.shape[-1] == 2:
               inst_a = inst_a[..., 0]
            inst_a_binary_mask = (inst_a!=0).astype(np.uint8)

            inst_b = self.dict_all_models_label[b][int(c)]
            inst_b = np.asarray(inst_b)
            if inst_b.ndim == 3 and inst_b.shape[-1] == 2:
                inst_b = inst_b[..., 0]
            inst_b_binary_mask = (inst_b!=0).astype(np.uint8)

            comb = inst_a_binary_mask + 2*inst_b_binary_mask
            colors = ["black", "#E69F00", "#56B4E9", "white"]
            cmap = ListedColormap(colors)

            im1 = ax1.imshow(comb, cmap=cmap)
            handles = [
                Line2D([0], [0], marker="o", color="black", label="Background",
                    markerfacecolor="black", markeredgecolor="black", markersize=10, linestyle="None"),
                Line2D([0], [0], marker="o", color="white", label="Overlap",
                    markerfacecolor="white", markeredgecolor="black", markersize=10, linestyle="None"),
                Line2D([0], [0], marker="o", color="#E69F00", label="Model 1",
                    markerfacecolor="#E69F00", markeredgecolor="black", markersize=10, linestyle="None"),
                Line2D([0], [0], marker="o", color="#56B4E9", label="Model 2",
                    markerfacecolor="#56B4E9", markeredgecolor="black", markersize=10, linestyle="None")
            ]
            ax1.legend(handles=handles,loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False,fontsize=11)


        # ---- instance seg – model 1 ----
            inst_a = np.ma.masked_where(inst_a == 0, inst_a)
            ax2.imshow(inst_a, cmap="tab20")
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title("Model 1 (instance)",fontsize=15)


        # ---- instance seg – model 2 ----
            inst_b = np.ma.masked_where(inst_b == 0, inst_b)
            ax3.imshow(inst_b, cmap="tab20")
            ax3.set_xticks([]); ax3.set_yticks([])
            ax3.set_title("Model 2 (instance)",fontsize=15)


        # ---- raw + seg overlay – model 1 ----
            ax4.imshow(raw, cmap="gray")
            self.draw_seg_inst_outlines(ax4, inst_a)
            ax4.set_xticks([]); ax4.set_yticks([])
            ax4.set_title("Raw + Model 1 (outlines)",fontsize=15)


        # ---- raw + seg overlay – model 2 ----
            ax5.imshow(raw, cmap="gray")
            self.draw_seg_inst_outlines(ax5, inst_b, color="cyan", lw=1.5)
            ax5.set_xticks([]); ax5.set_yticks([])
            ax5.set_title("Raw + Model 2 (outlines)",fontsize=15)

        # ---- bar plot: mean disagreements + std dev ----
            mdl_ids = list(self.model_diff_scores.keys())
            scores, std_devs = zip(*[self.model_diff_scores[m] for m in mdl_ids])
            short_mdl_ids = [f"{m[:5]}...{m.split('_')[-1]}" for m in mdl_ids]

            ax6.bar(range(len(mdl_ids)), scores, yerr=std_devs, width=0.35, capsize=8)
            ax6.set_xticks(range(len(mdl_ids)))
            ax6.set_xticklabels(short_mdl_ids, rotation=45, ha="right")
            ax6.set_ylabel("Mean semantic difference")
            ax6.set_title("Per-model disagreement",fontsize=15)

            plt.show()


        controls = widgets.VBox([
          widgets.Dropdown(
              options=self.dict_all_models.keys(),
              description="Model 1", layout=widgets.Layout(width="50%")
          ),
          widgets.Dropdown(
              options=self.dict_all_models.keys(),
              description="Model 2", layout=widgets.Layout(width="50%")
          ),
          widgets.IntSlider(
              min=0,
              max=len(next(iter(self.dict_all_models.values()))) - 1,
              description="Image ID"
          ),
      ])

        self.output_seg_comp = interactive_output(
            f,
            {"a": controls.children[0], "b": controls.children[1], "c": controls.children[2]},
        )

        display(controls, self.output_seg_comp)

def run_all_chosen_models_timing(self):
    """
    Runs all pretrained models of chosen model types and records inference times.
    - self.model_inference_times: seconds per image (avg over imgs_cut)
    - self.model_inference_times_total: total wall time per model (seconds)
    Returns simple table with inference times and hardware info (CPU, GPU, etc)
    """
    self.dict_all_models = {}
    self.dict_all_models_label = {}
    self.model_inference_times = {}
    self.model_inference_times_total = {}

    try:
        n_imgs = len(self.imgs_cut)
    except Exception:
        n_imgs = 1

    rows = []

    for nnt, models in self.all_chosen_seg_models.items():
        self.select_segmentator(nnt)
        for model in models:
            model_name = "_".join((model).split("_")[2:])

            key = f"{model}"
            t0 = time.perf_counter()

            self.pred.run_image_stack_jupyter(
                self.imgs_cut, model_name, clean_border=False
            )
            elapsed = time.perf_counter() - t0

            self.dict_all_models[key] = self.pred.seg_bin
            self.dict_all_models_label[key] = self.pred.seg_label

            self.model_inference_times_total[key] = elapsed
            self.model_inference_times[key] = elapsed / max(1, n_imgs)

            # ---- time inference summary table ---------
            rows.append({
                "Model": key,
                "Images": n_imgs,
                "Total time (s)": elapsed,
                "Images / s": (n_imgs / elapsed) if elapsed > 0 else float("inf"),
            })

            # ------------------------------------------------------
            # Free GPU memory that might still be held by the just
            # finished predictor.  This is crucial when executing
            # multiple models sequentially in the same notebook /
            # Colab runtime to avoid out-of-memory crashes.
            # ------------------------------------------------------
            if hasattr(self.pred, "cleanup"):
                self.pred.cleanup()
    


    print("\n\n\n\n\n==== Inference Time Summary ====\n")
    if rows:
        _print_runtime_env()
        df = pd.DataFrame(rows)
        display(df)


def _print_runtime_env():
    """Print a one-line summary of the compute device (TPU/GPU/CPU)."""
    # TPU
    if os.environ.get("COLAB_TPU_ADDR"):
        print("____ Running on TPU ____")
        return

    # GPU
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        if gpu_name:
            print(f"____ Running on GPU: {gpu_name.splitlines()[0]} ____")
            return
    except Exception:
        pass

    # CPU 
    try:
        cpu = subprocess.check_output(
            "lscpu | grep 'Model name' | sed 's/.*: //'",
            shell=True, text=True
        ).strip()
        if cpu:
            print(f"____ Running on CPU: {cpu} ____")
            return
    except Exception:
        pass

    print("____ Running on CPU ____")


# --- PATCHERS ---------------------------------------------------------------------

def patch_SJ_class():
    """Attach extension methods to the SegmentationJupyter class."""
    SegmentationJupyter.select_seg_models = select_seg_models
    SegmentationJupyter.compare_and_plot_segmentations = compare_and_plot_segmentations
    SegmentationJupyter.draw_seg_inst_outlines = draw_seg_inst_outlines
    SegmentationJupyter.run_all_chosen_models_timing = run_all_chosen_models_timing

# --- AUTO-PATCH ON IMPORT ---------------------------------------------------------

try:
    patch_SJ_class()  
    print("[midap_ext] Patched SegmentationJupyter with extra methods.")
except Exception as e:
    print(f"[midap_ext] Warning: could not patch class on import: {e}")
    pass


__all__ = [
    "select_seg_models",
    "draw_seg_inst_outlines",
    "compare_and_plot_segmentations",
    "patch_SJ_class",
    "run_all_chose_models_timing"
]
