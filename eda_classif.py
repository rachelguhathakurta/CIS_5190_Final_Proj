
from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans

PATH_COLS = ["image_path", "filepath", "image", "path", "file_name"]
LAT_COLS = ["Latitude", "latitude", "lat"]
LON_COLS = ["Longitude", "longitude", "lon"]

# palette (for plots, not for images)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9992"
INK, INK2, GRID, SURFACE =  "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
M_PER_DEG_LAT = 111_132.0

# helpers

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} in {list(df.columns)}")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


class Frame:
    """CSV + local metric coordinates (x = metres east, y = metres north of train centroid)."""

    def __init__(self, csv, ref=None, img_root=None):
        self.csv = csv
        self.df = pd.read_csv(csv)
        self.path_col = find_col(self.df, PATH_COLS)
        self.lat_col = find_col(self.df, LAT_COLS)
        self.lon_col = find_col(self.df, LON_COLS)
        self.root = img_root or os.path.dirname(os.path.abspath(csv))
        self.lat = self.df[self.lat_col].to_numpy(float)
        self.lon = self.df[self.lon_col].to_numpy(float)
        self.paths = [os.path.join(self.root, str(p)) for p in self.df[self.path_col]]
        self.ref = ref  # (lat0, lon0)

    def set_ref(self, ref):
        self.ref = ref
        lat0, lon0 = ref
        self.x = (self.lon - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
        self.y = (self.lat - lat0) * M_PER_DEG_LAT
        self.xy = np.c_[self.x, self.y]
        # photo "spots" (same definition as make_split.py: DBSCAN eps=4 m, min_samples=1)
        if "spot_id" in self.df.columns:
            self.spot = self.df["spot_id"].to_numpy()
        else:
            self.spot = DBSCAN(eps=4.0, min_samples=1).fit_predict(self.xy)


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, out, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p)


def ecdf(a):
    a = np.sort(a)
    return a, np.arange(1, len(a) + 1) / len(a)


# class definitions
def kmeans_classes(train, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(train.xy)
    return km.cluster_centers_, km.labels_


def grid_classes(train, cell_m):
    ix = np.floor(train.x / cell_m).astype(int)
    iy = np.floor(train.y / cell_m).astype(int)
    keys = pd.Series(list(zip(ix, iy)))
    codes, uniques = pd.factorize(keys)
    cents = np.array([train.xy[codes == c].mean(0) for c in range(len(uniques))])
    return cents, codes


def floor_error_m(frame, centroids_xy):
    """Best case: every sample is assigned to its nearest centroid, prediction = that centroid."""
    d, idx = cKDTree(centroids_xy).query(frame.xy)
    return float(d.mean()), idx


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", default=None)
    ap.add_argument("--img-root", default=None)
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--k", type=int, default=12, help="number of k-means classes for the detailed plots")
    ap.add_argument("--sweep", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32, 48, 64])
    ap.add_argument("--grid-cells", type=float, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--n-img", type=int, default=300, help="images sampled for image statistics")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--bounds", type=float, nargs=4, default=[39.950, 39.955, -75.195, -75.188],
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="the collection rectangle used in the HEIC conversion script")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    tr = Frame(args.train_csv, img_root=args.img_root)
    ref = (float(tr.lat.mean()), float(tr.lon.mean()))
    tr.set_ref(ref)
    va = None
    if args.val_csv:
        va = Frame(args.val_csv, img_root=args.img_root)
        va.set_ref(ref)

    # ---- 1. basic audit ---------------------------------------------------------------
    say("== Basic audit")
    for name, f in [("train", tr), ("val", va)]:
        if f is None:
            continue
        n = len(f.df)
        say(f"{name}: {n} rows | NaN coords: {int(np.isnan(f.lat).sum() + np.isnan(f.lon).sum())}"
            f" | duplicate rows: {int(f.df.duplicated().sum())}"
            f" | unique coordinate pairs: {len(set(zip(f.lat.round(7), f.lon.round(7))))}")
        if not args.no_images:
            missing = sum(not os.path.exists(p) for p in f.paths)
            say(f"{name}: missing image files: {missing}")
    sx, sy = tr.x.std(), tr.y.std()
    say(f"train centre (lat, lon): {ref[0]:.6f}, {ref[1]:.6f}")
    say(f"train spread: std E-W {sx:.1f} m, N-S {sy:.1f} m | extent E-W {np.ptp(tr.x):.0f} m, N-S {np.ptp(tr.y):.0f} m")
    d_nn, _ = cKDTree(tr.xy).query(tr.xy, k=2)
    nn = d_nn[:, 1]
    say(f"train nearest-neighbour distance (m): median {np.median(nn):.2f}, mean {nn.mean():.2f}, "
        f"90th pct {np.percentile(nn, 90):.1f}, max {nn.max():.1f}")
    # baselines
    base_mean = haversine_m(tr.lat, tr.lon, ref[0], ref[1]).mean()
    say(f"baseline 'predict train mean' on train: mean error {base_mean:.1f} m")
    if va is not None:
        bv = haversine_m(va.lat, va.lon, ref[0], ref[1]).mean()
        say(f"baseline 'predict train mean' on val:   mean error {bv:.1f} m   <- regression model must beat this")

    # ---- fig 1: where are the points -------------------------------------------------------   
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    ax[0].scatter(tr.x, tr.y, s=6, color=BLUE, alpha=0.45, linewidths=0, label=f"train (n={len(tr.x)})")
    if va is not None:
        ax[0].scatter(va.x, va.y, s=8, color=ORANGE, alpha=0.6, linewidths=0, label=f"val (n={len(va.x)})")
    ax[0].set_aspect("equal")
    style(ax[0], "Image locations (metres from train centre)", "east (m)", "north (m)")
    ax[0].legend(frameon=False, fontsize=9, markerscale=2)
    hb = ax[1].hexbin(tr.x, tr.y, gridsize=30, cmap="Blues", mincnt=1)
    ax[1].set_aspect("equal")
    style(ax[1], "Train density (images per hexagon)", "east (m)", "north (m)")
    fig.colorbar(hb, ax=ax[1], shrink=0.8)
    save(fig, args.out, "01_location_map.png")

    # ---- fig 2: marginals --------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(tr.x, bins=40, color=BLUE)
    style(ax[0], "East-west position", "metres from centre", "images")
    ax[1].hist(tr.y, bins=40, color=BLUE)
    style(ax[1], "North-south position", "metres from centre", "images")
    save(fig, args.out, "02_position_histograms.png")

    # ---- fig 2b: coverage of the collection rectangle ------------------------------------------
    la0, la1, lo0, lo1 = args.bounds
    rx0 = (lo0 - ref[1]) * M_PER_DEG_LAT * math.cos(math.radians(ref[0]))
    rx1 = (lo1 - ref[1]) * M_PER_DEG_LAT * math.cos(math.radians(ref[0]))
    ry0, ry1 = (la0 - ref[0]) * M_PER_DEG_LAT, (la1 - ref[0]) * M_PER_DEG_LAT
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.add_patch(plt.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, ec=GRAY, lw=1.5, ls="--"))
    ax.scatter(tr.x, tr.y, s=6, color=BLUE, alpha=0.5, linewidths=0, label="train")
    if va is not None:
        ax.scatter(va.x, va.y, s=8, color=ORANGE, alpha=0.7, linewidths=0, label="val")
    ax.set_xlim(min(rx0, tr.x.min()) - 20, max(rx1, tr.x.max()) + 20)
    ax.set_ylim(min(ry0, tr.y.min()) - 20, max(ry1, tr.y.max()) + 20)
    ax.set_aspect("equal")
    style(ax, "Data vs collection rectangle (dashed)", "east (m)", "north (m)")
    ax.legend(frameon=False, fontsize=9, markerscale=2)
    save(fig, args.out, "02b_coverage_of_rectangle.png")
    inside = (tr.x >= rx0) & (tr.x <= rx1) & (tr.y >= ry0) & (tr.y <= ry1)
    say(f"rectangle: {rx1 - rx0:.0f} m E-W x {ry1 - ry0:.0f} m N-S | train points inside: {inside.mean():.1%} | "
        f"area covered by data (bbox of 1-99 pct): "
        f"{(np.percentile(tr.x, 99) - np.percentile(tr.x, 1)):.0f} x {(np.percentile(tr.y, 99) - np.percentile(tr.y, 1)):.0f} m")

    # ---- fig 2c: spots --------------------------------------------------------------------------
    spots = pd.Series(tr.spot)
    per_spot = spots.value_counts()
    ext = []
    for sid, grp in pd.DataFrame({"s": tr.spot, "x": tr.x, "y": tr.y}).groupby("s"):
        ext.append(math.hypot(np.ptp(grp.x), np.ptp(grp.y)))
    ext = np.asarray(ext)
    say(f"\n== Spots (train): {len(per_spot)} spots | photos/spot min {per_spot.min()}, median {int(per_spot.median())}, "
        f"max {per_spot.max()} | spots with 1 photo: {(per_spot == 1).sum()}")
    say(f"spot extent (bbox diagonal, m): median {np.median(ext):.1f}, 90th pct {np.percentile(ext, 90):.1f}, "
        f"max {ext.max():.1f}  <- much larger than 4 m means DBSCAN chained photos along a path")
    if va is not None:
        say(f"val spots: {len(set(va.spot))} (ids may not be comparable if spot_id column is absent)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(per_spot.values, bins=min(30, max(5, per_spot.max())), color=BLUE)
    style(ax[0], "Photos per spot (train)", "photos in spot", "spots")
    ax[1].hist(ext, bins=30, color=BLUE)
    style(ax[1], "Spot extent (chaining check)", "bbox diagonal (m)", "spots")
    save(fig, args.out, "02c_spots.png")

    # ---- fig 3: how many classes can we afford? ------------------------------------------------
    say("\n== Class design sweep (oracle floor = error if classification were 100% accurate)")
    rows = []
    for k in [k for k in args.sweep if k < len(tr.x)]:
        c, lab = kmeans_classes(tr, k)
        e_tr, _ = floor_error_m(tr, c)
        e_va = floor_error_m(va, c)[0] if va is not None else np.nan
        sizes = np.bincount(lab, minlength=k)
        rows.append(("kmeans", k, k, e_tr, e_va, np.median(sizes), sizes.min()))
    for s in args.grid_cells:
        c, lab = grid_classes(tr, s)
        e_tr, _ = floor_error_m(tr, c)
        e_va = floor_error_m(va, c)[0] if va is not None else np.nan
        sizes = np.bincount(lab)
        rows.append((f"grid {s:g} m", s, len(c), e_tr, e_va, np.median(sizes), sizes.min()))
    sweep = pd.DataFrame(rows, columns=["method", "param", "n_classes", "floor_train_m", "floor_val_m",
                                        "median_per_class", "min_per_class"])
    say(sweep.round(2).to_string(index=False))
    sweep.to_csv(os.path.join(args.out, "class_sweep.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for method, color, ls_name in [("kmeans", BLUE, "k-means"), ("grid", ORANGE, "square grid")]:
        d = sweep[sweep.method.str.startswith(method)].sort_values("n_classes")
        ax[0].plot(d.n_classes, d.floor_train_m, "-o", color=color, lw=2, ms=5, label=f"{ls_name} (train)")
        if va is not None:
            ax[0].plot(d.n_classes, d.floor_val_m, "--o", color=color, lw=2, ms=5, mfc=SURFACE,
                       label=f"{ls_name} (val)")
        ax[1].plot(d.n_classes, d.median_per_class, "-o", color=color, lw=2, ms=5, label=ls_name)
    ax[0].axhline(base_mean, color=GRAY, ls=":", lw=1.5)
    ax[0].text(ax[0].get_xlim()[1], base_mean, " predict-the-mean", color=INK2, fontsize=8, va="bottom", ha="right")
    ax[0].set_xscale("log")
    style(ax[0], "Error floor vs number of classes", "number of classes (log)", "mean error if every prediction is a correct class (m)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].axhline(20, color=GRAY, ls=":", lw=1.5)
    ax[1].text(ax[1].get_xlim()[0], 20, " ~20 images/class", color=INK2, fontsize=8, va="bottom")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    style(ax[1], "Median training images per class", "number of classes (log)", "images per class (log)")
    ax[1].legend(frameon=False, fontsize=8)
    save(fig, args.out, "03_class_count_tradeoff.png")

    # ---- chosen K: balance + map ------------------------------------------------------------
    K = min(args.k, len(tr.x) - 1)
    cent, lab = kmeans_classes(tr, K)
    counts = np.bincount(lab, minlength=K)
    order = np.argsort(-counts)
    say(f"\n== Chosen K={K} (k-means in metres)")
    say(f"images per class: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()} "
        f"| majority-class share {counts.max() / counts.sum():.1%} (accuracy of always guessing the biggest class)")
    e_tr, _ = floor_error_m(tr, cent)
    spots_per_class = np.array([len(set(tr.spot[lab == c])) for c in range(K)])
    say(f"distinct spots per class: min {spots_per_class.min()}, median {int(np.median(spots_per_class))}, "
        f"max {spots_per_class.max()}  <- real diversity; photos from one spot are near-duplicates")
    say(f"oracle floor, train: {e_tr:.1f} m" + (f" | val: {floor_error_m(va, cent)[0]:.1f} m" if va is not None else ""))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax[0].bar(np.arange(K), counts[order], color=BLUE, width=0.7)
    ax[0].axhline(counts.mean(), color=GRAY, ls=":", lw=1.5)
    style(ax[0], f"Class balance (K={K}, sorted)", "class rank", "training images")
    ax[0].set_xticks(np.arange(K))
    ax[0].set_xticklabels([str(c) for c in order], fontsize=8)
    ax[0].set_xlabel("class id (sorted by size)", color=INK2, fontsize=9)
    ax[1].scatter(tr.x, tr.y, s=4, color=GRAY, alpha=0.35, linewidths=0)
    sc = ax[1].scatter(cent[:, 0], cent[:, 1], c=counts, cmap="Blues", s=90, edgecolors=INK2, linewidths=0.8,
                       vmin=0)
    for c in range(K):
        ax[1].annotate(str(c), cent[c], fontsize=7, color=INK, ha="center", va="center",
                       xytext=(0, 9), textcoords="offset points")
    ax[1].set_aspect("equal")
    style(ax[1], "Class centroids (shade = images in class)", "east (m)", "north (m)")
    fig.colorbar(sc, ax=ax[1], shrink=0.8)
    save(fig, args.out, f"04_classes_K{K}.png")

    # ---- fig 5: does val look like train? ----------------------------------------------------------
    if va is not None:
        d_vt, _ = cKDTree(tr.xy).query(va.xy)
        say("\n== Train/val overlap")
        say(f"distance from each val image to the closest train image (m): median {np.median(d_vt):.2f}, "
            f"mean {d_vt.mean():.2f}, 90th pct {np.percentile(d_vt, 90):.1f}, max {d_vt.max():.1f}")
        say("If these are small, val locations were seen in training (random split); "
            "if large, val is a spatial hold-out and classification will cap out well below 100%.")
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        for arr, color, lbl, ls in [(nn, BLUE, "train -> other train (leave-one-out)", "-"),
                                    (d_vt, ORANGE, "val -> nearest train", "-")]:
            xs, ys = ecdf(arr)
            ax.plot(xs, ys, color=color, lw=2, ls=ls, label=lbl)
        style(ax, "How close is the nearest labelled neighbour?", "distance (m)", "fraction of images")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        save(fig, args.out, "05_val_vs_train_distance.png")

    # ---- saving class tables for the modelling code ---------------------------------------------------
    cdf = pd.DataFrame({"class": range(K), "n_train": counts, "n_spots": spots_per_class})
    cdf["lat"] = [tr.lat[lab == c].mean() for c in range(K)]
    cdf["lon"] = [tr.lon[lab == c].mean() for c in range(K)]
    cdf.to_csv(os.path.join(args.out, f"centroids_K{K}.csv"), index=False)
    out_tr = tr.df[[tr.path_col, tr.lat_col, tr.lon_col]].copy()
    out_tr["class"] = lab
    out_tr.to_csv(os.path.join(args.out, f"class_assignments_train_K{K}.csv"), index=False)
    if va is not None:
        _, vlab = cKDTree(cent).query(va.xy)
        out_va = va.df[[va.path_col, va.lat_col, va.lon_col]].copy()
        out_va["class"] = vlab
        out_va.to_csv(os.path.join(args.out, f"class_assignments_val_K{K}.csv"), index=False)
        vc = np.bincount(vlab, minlength=K)
        say(f"val images per class: min {vc.min()}, median {int(np.median(vc))}, max {vc.max()} "
            f"(classes with 0 val images: {(vc == 0).sum()})")

    # ---- images --------------------------------------------------------------------------------------
    if not args.no_images:
        say("\n== Image statistics")
        idx = rng.choice(len(tr.paths), size=min(args.n_img, len(tr.paths)), replace=False)
        w, h, bright, cls = [], [], [], []
        for i in idx:
            try:
                with Image.open(tr.paths[i]) as im:
                    im = ImageOps.exif_transpose(im)
                    w.append(im.width)
                    h.append(im.height)
                    small = im.convert("L").resize((64, 64))
                    bright.append(np.asarray(small).mean())
                    cls.append(lab[i])
            except Exception as e:  # unreadable file
                say(f"could not read {tr.paths[i]}: {e}")
        w, h, bright, cls = map(np.asarray, (w, h, bright, cls))
        say(f"sampled {len(w)} images | width {w.min()}-{w.max()} (median {int(np.median(w))}), "
            f"height {h.min()}-{h.max()} (median {int(np.median(h))})")
        say(f"aspect ratio (w/h): {np.unique((w / h).round(2))[:6]} ... | after Resize((224,224)) "
            f"non-square images are squashed, not cropped")
        say(f"mean brightness (0-255): {bright.mean():.0f} +/- {bright.std():.0f}")

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        ax[0].scatter(w, h, s=10, color=BLUE, alpha=0.5, linewidths=0)
        style(ax[0], "Image size (after EXIF rotation)", "width (px)", "height (px)")
        ax[1].hist(bright, bins=30, color=BLUE)
        style(ax[1], "Mean brightness", "0 = black, 255 = white", "images")
        groups = [bright[cls == c] for c in order if (cls == c).sum() >= 3]
        names = [str(c) for c in order if (cls == c).sum() >= 3]
        if groups:
            bp = ax[2].boxplot(groups, patch_artist=True, widths=0.6, medianprops=dict(color=INK))
            for b in bp["boxes"]:
                b.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
            ax[2].set_xticklabels(names, fontsize=7)
        style(ax[2], "Brightness by class (lighting confound?)", "class id", "mean brightness")
        save(fig, args.out, "06_image_stats.png")

        # sample grid: 6 largest classes x 5 images
        show = order[: min(6, K)]
        ncol = 5
        fig, axes = plt.subplots(len(show), ncol, figsize=(ncol * 2.2, len(show) * 2.0), squeeze=False)
        for r, c in enumerate(show):
            members = np.where(lab == c)[0]
            pick = rng.choice(members, size=min(ncol, len(members)), replace=False)
            for j in range(ncol):
                a = axes[r, j]
                a.axis("off")
                if j < len(pick):
                    try:
                        with Image.open(tr.paths[pick[j]]) as im:
                            a.imshow(ImageOps.exif_transpose(im).convert("RGB").resize((160, 160)))
                    except Exception:
                        pass
                if j == 0:
                    a.set_title(f"class {c}  (n={counts[c]})", fontsize=9, loc="left", color=INK)
        save(fig, args.out, f"07_samples_per_class_K{K}.png")

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(log))
    print(f"\nDone. Everything is in {args.out}/")


if __name__ == "__main__":
    main()
