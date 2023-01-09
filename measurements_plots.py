#%%
import json
from pathlib import Path
import matplotlib.pyplot as plt

#%%
models = ["gpt2-xl", "EleutherAI/gpt-j-6B"]
all_methods = {"": "CDE", "-inlp": "probe", "-rlace": "RLACE", "-she-he": "logit lens"}

figure_folder = Path("figures/measurement_plots")
figure_folder.mkdir(exist_ok=True, parents=True)


def load_results(model: str, result_type: str):
    results = {}
    for method, method_name in all_methods.items():
        path = Path(f"measurements/v3-{model}{method}/{result_type}.json")
        if path.exists():
            results[method_name] = json.load(path.open("r"))
    return results


# %%


def plot_perplexity(model):
    results = load_results(model, "perplexity")
    for method, values in results.items():
        layers = [v["layer"] for v in values if v["layer"] != -2]
        ps = [v["p"] for v in values if v["layer"] != -2]
        plt.plot(layers, ps, label=method)
    ref = values[-1]
    assert ref["layer"] == -2
    ref_val = ref["p"]
    plt.axhline(ref_val, label="without ablation", color="black", linestyle="--")
    plt.ylim(ref_val * 0.9, ref_val * 1.1)

    plt.title(model)
    plt.xlabel("Layer")
    plt.ylabel("Perplexity")
    plt.legend()
    folder = figure_folder / model
    folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{figure_folder}/{model}/perplexity.png", bbox_inches="tight")


for model in models:
    plot_perplexity(model)
    plt.show()
# %%
def plot_stereotypes(model):
    results = load_results(model, "stereotype")
    for method, values in results.items():
        layers = [v["layer"] for v in values if v["layer"] != -2]
        # 0 anti stereotype, 1 stereotype, 2 irrelevant
        # compute 1 / (1 + 0)
        ps = [v["p"][1] / (v["p"][1] + v["p"][0]) for v in values if v["layer"] != -2]
        plt.plot(layers, ps, label=method)
    ref = values[-1]
    assert ref["layer"] == -2
    ref_val = ref["p"][1] / (ref["p"][1] + ref["p"][0])
    plt.axhline(ref_val, label="without ablation", color="black", linestyle="--")
    plt.axhline(0.5, label="as much stereotypes as anti stereotypes", color="gray", linestyle=":")
    plt.ylim(0.48, ref_val * 1.1)

    plt.title(model)
    plt.xlabel("Layer")
    plt.ylabel("Ratio of stereotype sentences")
    plt.legend()
    folder = figure_folder / model
    folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{figure_folder}/{model}/stereotype.png", bbox_inches="tight")


for model in models:
    plot_stereotypes(model)
    plt.show()
# %%
def plot_professions(model):
    results = load_results(model, "profession")
    for method, values in results.items():
        layers = [v["layer"] for v in values if v["layer"] != -2]
        # 0 anti stereotype, 1 stereotype, 2 irrelevant
        # compute 1 / (1 + 0)
        ps = [v["p"][1] for v in values if v["layer"] != -2]
        plt.plot(layers, ps, label=method)
    ref = values[-1]
    assert ref["layer"] == -2
    ref_val = ref["p"][1]
    plt.axhline(ref_val, label="without ablation", color="black", linestyle="--")
    plt.axhline(0, color="gray", linestyle=":")
    plt.axhline(1, color="gray", linestyle=":")
    plt.ylim(-0.1, 1.1)

    plt.title(model)
    plt.xlabel("Layer")
    plt.ylabel("Average equity score")
    plt.legend()
    folder = figure_folder / model
    folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{figure_folder}/{model}/professions.png", bbox_inches="tight")


for model in models:
    plot_professions(model)
    plt.show()
# %%
