import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive
import matplotlib.pyplot as plt
from midap.midap_jupyter.segmentation_jupyter import SegmentationJupyter

def draw_seg_inst_outlines(ax, inst_labels, color="yellow", lw=1.5):
        inst = np.asarray(inst_labels)
        if inst.ndim == 3 and inst.shape[-1] == 2:  
            inst = inst[..., 0]
        labels = np.unique(inst)
        labels = labels[labels != 0]  

        for lab in labels:
            ax.contour(inst == lab, levels=[0.5], colors=[color], linewidths=lw)


def compare_and_plot_segmentations(sj):
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
        if not hasattr(sj, "model_diff_scores"):
            sj.model_diff_scores = sj.compute_model_diff_scores()


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
            raw = sj.imgs_cut[int(c)]
            ax0.imshow(raw, cmap="gray")
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_title("Raw image")

        # ---- instance seg – model 1 ----
            inst_a = sj.dict_all_models_label[a][int(c)]
            inst_a = np.asarray(inst_a)
            if inst_a.ndim == 3 and inst_a.shape[-1] == 2:
               inst_a = inst_a[..., 0]
            inst_a = np.ma.masked_where(inst_a == 0, inst_a)
            ax1.imshow(inst_a, cmap="tab20")
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_title("Model 1 (instance)")


        # ---- instance seg – model 2 ----
            inst_b = sj.dict_all_models_label[b][int(c)]
            inst_b = np.asarray(inst_b)
            if inst_b.ndim == 3 and inst_b.shape[-1] == 2:
                inst_b = inst_b[..., 0]
            inst_b = np.ma.masked_where(inst_b == 0, inst_b)
            ax2.imshow(inst_b, cmap="tab20")
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title("Model 2 (instance)")


        # ---- raw + seg overlay – model 1 ----
            inst_a = sj.dict_all_models_label[a][int(c)]
            ax3.imshow(raw, cmap="gray")
            draw_seg_inst_outlines(ax3, inst_a)
            ax3.set_xticks([]); ax3.set_yticks([])
            ax3.set_title("Raw + Model 1 (outlines)")


        # ---- raw + seg overlay – model 2 ----
            inst_b = sj.dict_all_models_label[b][int(c)]
            ax4.imshow(raw, cmap="gray")
            draw_seg_inst_outlines(ax4, inst_b, color="cyan", lw=1.5)
            ax4.set_xticks([]); ax4.set_yticks([])
            ax4.set_title("Raw + Model 2 (outlines)")

        # ---- bar plot: mean disagreements + std dev ----
            mdl_ids = list(sj.model_diff_scores.keys())
            scores, std_devs = zip(*[sj.model_diff_scores[m] for m in mdl_ids])
            short_mdl_ids = [f"{m[:5]}...{m.split('_')[-1]}" for m in mdl_ids]

            ax5.bar(range(len(mdl_ids)), scores, yerr=std_devs, capsize=5)
            ax5.set_xticks(range(len(mdl_ids)))
            ax5.set_xticklabels(short_mdl_ids, rotation=90)
            ax5.set_ylabel("Mean semantic difference")
            ax5.set_title("Per-model disagreement")

            plt.show()
            plt.close(fig) # stop figure count increasing with every run
  

        sj.output_seg_comp = interactive(
            f,
            a=widgets.Dropdown(
                options=sj.dict_all_models.keys(),
                description="Model 1", layout=widgets.Layout(width="50%")
            ),
            b=widgets.Dropdown(
                options=sj.dict_all_models.keys(),
                description="Model 2", layout=widgets.Layout(width="50%")
            ),
            c=widgets.IntSlider(
                min=0,
                max=len(next(iter(sj.dict_all_models.values()))) - 1,
                description="Image ID"
            ),
        )
        display(sj.output_seg_comp)

def patch_SJ_class():
    SegmentationJupyter.compare_and_plot_segmentations = compare_and_plot_segmentations


# --- PATCHERS ---------------------------------------------------------------------

def patch_SJ_class():
    """Attach extension methods to the SegmentationJupyter class (affects all instances)."""
    SegmentationJupyter.compare_and_plot_segmentations = compare_and_plot_segmentations




# --- AUTO-PATCH ON IMPORT ---------------------------------------------------------

try:
    patch_SJ_class()  # make methods available as sj.draw_instance_outlines(), sj.compare_and_plot_segmentations()
    print("[midap_ext] Patched SegmentationJupyter with extra methods.")
except Exception as e:
    print(f"[midap_ext] Warning: could not patch class on import: {e}")
    pass


__all__ = [
    "compare_and_plot_segmentations",
    "patch_SJ_class",
]
