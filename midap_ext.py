,import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from google.colab import data_table
from midap.midap_jupyter.segmentation_jupyter import SegmentationJupyter


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

    search = widgets.Text(placeholder="filter models with ... (substring match)", layout=Layout(width="40%"))
    sel    = widgets.SelectMultiple(options=sorted(all_names), rows=12, description="Select")
    btn_all   = widgets.Button(description="Select all (filtered)", tootip='Select all models matching filter keywords')
    btn_clear  = widgets.Button(description="Clear", tooltip='Clear selection')
    btn_apply = widgets.Button(description="Apply selection",tooltip='Apply selected models')
    btn_applyrun   = widgets.Button(description="Apply & run", tooltip='Apply selected models and run segmentation',button_style="primary")
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
            name: widgets.Checkbox(value=(name in selected), indent=False, layout=Layout(width="1px", height="1px"))
            for name in all_names
        }

        with out:
            clear_output()
            print(f"Selected {len(selected)} model(s):")
            for n in sorted(selected):
                print("  •", n)

        if run_now:
            self.select_segmentation_models()
            self.run_all_chosen_models()

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
          2. instance segmentation of model-1
          3. instance segmentation of model-2
          4. overlay raw image + outlines of instance segmentation of model-1
          5. overlay raw image + outlines of instance segmentation of model-2
          6. bar-plot of the per-model mean semantic disagreement scores
             with standard deviation as error bars.
        """

        # ----------------------------------------------------------------
        # prepare bar-plot data (only once)
        # ----------------------------------------------------------------
        if not hasattr(self, "model_diff_scores"):
            self.model_diff_scores = self.compute_model_diff_scores()


        def f(a, b, c):
            fig = plt.figure(figsize=(20, 22))
            gs = fig.add_gridspec(
                nrows=4, ncols=2,
                height_ratios=[1.0, 1.0, 1.0, 0.8],   
                hspace=0.13, wspace=0.08
            )

        # ---- Row 1 (raw) ----
            ax0 = fig.add_subplot(gs[0, :])

        # ---- Row 2 (two segmentations) — side by side ----
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
            ax2 = fig.add_subplot(gs[1, 1], sharex=ax0, sharey=ax0)

        # Row 3 (two segmentations) — side by side
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)
            ax4 = fig.add_subplot(gs[2, 1], sharex=ax0, sharey=ax0)

        # Row 4 (bar plot) — span both columns
            ax5 = fig.add_subplot(gs[3, :])

        # ---- raw image ----
            raw = self.imgs_cut[int(c)]
            ax0.imshow(raw, cmap="gray")
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_title("Raw image")

        # ---- instance seg – model 1 ----
            inst_a = self.dict_all_models_label[a][int(c)]
            inst_a = np.asarray(inst_a)
            if inst_a.ndim == 3 and inst_a.shape[-1] == 2:
               inst_a = inst_a[..., 0]
            inst_a = np.ma.masked_where(inst_a == 0, inst_a)
            ax1.imshow(inst_a, cmap="tab20")
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_title("Model 1 (instance)")


        # ---- instance seg – model 2 ----
            inst_b = self.dict_all_models_label[b][int(c)]
            inst_b = np.asarray(inst_b)
            if inst_b.ndim == 3 and inst_b.shape[-1] == 2:
                inst_b = inst_b[..., 0]
            inst_b = np.ma.masked_where(inst_b == 0, inst_b)
            ax2.imshow(inst_b, cmap="tab20")
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title("Model 2 (instance)")


        # ---- raw + seg overlay – model 1 ----
            inst_a = self.dict_all_models_label[a][int(c)]
            ax3.imshow(raw, cmap="gray")
            self.draw_seg_inst_outlines(ax3, inst_a)
            ax3.set_xticks([]); ax3.set_yticks([])
            ax3.set_title("Raw + Model 1 (outlines)")


        # ---- raw + seg overlay – model 2 ----
            inst_b = self.dict_all_models_label[b][int(c)]
            ax4.imshow(raw, cmap="gray")
            self.draw_seg_inst_outlines(ax4, inst_b, color="cyan", lw=1.5)
            ax4.set_xticks([]); ax4.set_yticks([])
            ax4.set_title("Raw + Model 2 (outlines)")

        # ---- bar plot: mean disagreements + std dev ----
            mdl_ids = list(self.model_diff_scores.keys())
            scores, std_devs = zip(*[self.model_diff_scores[m] for m in mdl_ids])
            short_mdl_ids = [f"{m[:5]}...{m.split('_')[-1]}" for m in mdl_ids]

            ax5.bar(range(len(mdl_ids)), scores, yerr=std_devs, capsize=5)
            ax5.set_xticks(range(len(mdl_ids)))
            ax5.set_xticklabels(short_mdl_ids, rotation=90)
            ax5.set_ylabel("Mean semantic difference")
            ax5.set_title("Per-model disagreement")

            plt.show()
            plt.close(fig) # stop figure count increasing with every run
  

        self.output_seg_comp = interactive(
            f,
            a=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                description="Model 1", layout=widgets.Layout(width="50%")
            ),
            b=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                description="Model 2", layout=widgets.Layout(width="50%")
            ),
            c=widgets.IntSlider(
                min=0,
                max=len(next(iter(self.dict_all_models.values()))) - 1,
                description="Image ID"
            ),
        )
        display(self.output_seg_comp)



# --- PATCHERS ---------------------------------------------------------------------

def patch_SJ_class():
    """Attach extension methods to the SegmentationJupyter class (affects all instances)."""
    SegmentationJupyter.select_seg_models = select_seg_models
    SegmentationJupyter.compare_and_plot_segmentations = compare_and_plot_segmentations




# --- AUTO-PATCH ON IMPORT ---------------------------------------------------------

try:
    patch_SJ_class()  # make methods available as sj.draw_instance_outlines(), sj.compare_and_plot_segmentations()
    print("[midap_ext] Patched SegmentationJupyter with extra methods.")
except Exception as e:
    print(f"[midap_ext] Warning: could not patch class on import: {e}")
    pass


__all__ = [
    "select_seg_models",
    "compare_and_plot_segmentations",
    "patch_SJ_class",
]
