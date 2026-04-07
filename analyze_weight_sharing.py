#!/usr/bin/env python3
"""
Weight Sharing Viability Analysis for A4 Architecture
Loads final_model.pt and determines if layers share enough structure
to support a shared base + diagonal scaling + low-rank correction approach.
"""

import torch
import sys
import os

MATRIX_TYPES = {
    "c_q":       "blocks.{}.attn.c_q.weight",
    "c_k":       "blocks.{}.attn.c_k.weight",
    "c_v":       "blocks.{}.attn.c_v.weight",
    "attn_proj":  "blocks.{}.attn.proj.weight",
    "mlp_fc":    "blocks.{}.mlp.fc.weight",
    "mlp_proj":  "blocks.{}.mlp.proj.weight",
}

RANKS_TO_TEST = [4, 8, 16, 32, 48, 64]


def load_weights(checkpoint_path, num_layers):
    sd = torch.load(checkpoint_path, map_location="cpu")
    weights = {}  # {matrix_type: [W_0, W_1, ..., W_{n-1}]}
    for name, key_pattern in MATRIX_TYPES.items():
        layers = []
        for i in range(num_layers):
            key = key_pattern.format(i)
            if key not in sd:
                print(f"WARNING: {key} not found in checkpoint")
                break
            layers.append(sd[key].float())
        if len(layers) == num_layers:
            weights[name] = layers
    return weights


def svd_spectrum_analysis(weights, num_layers, enc_layers, dec_layers):
    print("\n" + "=" * 60)
    print("  1. SVD SPECTRUM SIMILARITY")
    print("=" * 60)

    all_cosines = {}
    for name, layer_ws in weights.items():
        # Compute singular values for each layer
        svals = []
        for w in layer_ws:
            _, s, _ = torch.linalg.svd(w, full_matrices=False)
            svals.append(s)

        print(f"\n--- {name} {tuple(layer_ws[0].shape)} ---")
        print(f"{'Layer':>6} | {'Top-5 singular values':>50}")
        for i, s in enumerate(svals):
            top5 = ", ".join(f"{v:.2f}" for v in s[:5].tolist())
            print(f"  {i:>4} | {top5}")

        # Cosine similarity of full singular value vectors
        n = num_layers
        cos_matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                cos_matrix[i, j] = torch.dot(svals[i], svals[j]) / (svals[i].norm() * svals[j].norm())

        enc_cos = [cos_matrix[i, j].item() for i in enc_layers for j in enc_layers if i < j]
        dec_cos = [cos_matrix[i, j].item() for i in dec_layers for j in dec_layers if i < j]
        cross_cos = [cos_matrix[i, j].item() for i in enc_layers for j in dec_layers]

        mean_enc = sum(enc_cos) / len(enc_cos) if enc_cos else 0
        mean_dec = sum(dec_cos) / len(dec_cos) if dec_cos else 0
        mean_cross = sum(cross_cos) / len(cross_cos) if cross_cos else 0

        print(f"  Cosine sim: encoder={mean_enc:.4f}  decoder={mean_dec:.4f}  cross={mean_cross:.4f}")
        all_cosines[name] = {"enc": mean_enc, "dec": mean_dec, "cross": mean_cross}

    return all_cosines


def pairwise_distance_analysis(weights, num_layers, enc_layers, dec_layers):
    print("\n" + "=" * 60)
    print("  2. PAIRWISE NORMALIZED DISTANCE")
    print("=" * 60)

    all_distances = {}
    for name, layer_ws in weights.items():
        norms = [w.norm().item() for w in layer_ws]
        n = num_layers

        dist_matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = (layer_ws[i] - layer_ws[j]).norm().item()
                    avg_norm = (norms[i] + norms[j]) / 2
                    dist_matrix[i, j] = diff / avg_norm

        print(f"\n--- {name} ---")
        header = "      " + "".join(f"  L{j:d}   " for j in range(n))
        print(header)
        for i in range(n):
            row = f"  L{i} |"
            for j in range(n):
                if i == j:
                    row += "   --  "
                else:
                    row += f" {dist_matrix[i,j]:.3f} "
            print(row)

        enc_dists = [dist_matrix[i, j].item() for i in enc_layers for j in enc_layers if i < j]
        dec_dists = [dist_matrix[i, j].item() for i in dec_layers for j in dec_layers if i < j]
        cross_dists = [dist_matrix[i, j].item() for i in enc_layers for j in dec_layers]

        mean_enc = sum(enc_dists) / len(enc_dists) if enc_dists else 0
        mean_dec = sum(dec_dists) / len(dec_dists) if dec_dists else 0
        mean_cross = sum(cross_dists) / len(cross_dists) if cross_dists else 0

        print(f"  Mean dist: encoder={mean_enc:.4f}  decoder={mean_dec:.4f}  cross={mean_cross:.4f}")
        all_distances[name] = {"enc": mean_enc, "dec": mean_dec, "cross": mean_cross}

    return all_distances


def residual_rank_analysis(weights, num_layers, enc_layers, dec_layers):
    print("\n" + "=" * 60)
    print("  3. MEAN RESIDUAL + RANK ANALYSIS (core test)")
    print("=" * 60)

    groupings = {
        "GLOBAL": list(range(num_layers)),
        "ENCODER": enc_layers,
        "DECODER": dec_layers,
    }

    results = {}  # {(name, grouping): mean_rank_90}

    for name, layer_ws in weights.items():
        print(f"\n--- {name} {tuple(layer_ws[0].shape)} ---")

        for group_name, group_indices in groupings.items():
            if len(group_indices) < 2:
                continue

            group_ws = [layer_ws[i] for i in group_indices]
            w_mean = torch.stack(group_ws).mean(dim=0)

            print(f"\n  Grouping: {group_name} (layers {group_indices})")
            header = f"  {'Layer':>5} |"
            for r in RANKS_TO_TEST:
                header += f" R-{r:>2}  "
            header += "| Min-90%"
            print(header)
            print("  " + "-" * (len(header) - 2))

            rank_90s = []
            for idx, li in enumerate(group_indices):
                residual = layer_ws[li] - w_mean
                _, s, _ = torch.linalg.svd(residual, full_matrices=False)
                total_energy = (s ** 2).sum().item()

                if total_energy < 1e-10:
                    print(f"  {li:>5} | {'(zero residual)':>40} | 0")
                    rank_90s.append(0)
                    continue

                row = f"  {li:>5} |"
                min_rank_90 = s.shape[0]
                cumulative = torch.cumsum(s ** 2, dim=0)
                for r in RANKS_TO_TEST:
                    r_clamped = min(r, s.shape[0])
                    frac = cumulative[r_clamped - 1].item() / total_energy * 100
                    row += f" {frac:5.1f}%"
                    if frac >= 90.0 and r_clamped < min_rank_90:
                        min_rank_90 = r_clamped

                # Find exact minimum rank for 90%
                fracs = cumulative / total_energy
                above_90 = (fracs >= 0.9).nonzero(as_tuple=True)[0]
                if len(above_90) > 0:
                    min_rank_90 = above_90[0].item() + 1
                else:
                    min_rank_90 = s.shape[0]

                row += f" | {min_rank_90}"
                print(row)
                rank_90s.append(min_rank_90)

            mean_r90 = sum(rank_90s) / len(rank_90s) if rank_90s else 0
            print(f"  Mean rank for 90%: {mean_r90:.1f}")

            # Also report the rank-32 energy capture averaged across layers
            avg_r32_energy = 0
            for li in group_indices:
                residual = layer_ws[li] - w_mean
                _, s, _ = torch.linalg.svd(residual, full_matrices=False)
                total = (s ** 2).sum().item()
                if total < 1e-10:
                    avg_r32_energy += 100.0
                else:
                    r32 = min(32, s.shape[0])
                    avg_r32_energy += (s[:r32] ** 2).sum().item() / total * 100
            avg_r32_energy /= len(group_indices)
            print(f"  Avg rank-32 energy capture: {avg_r32_energy:.1f}%")

            results[(name, group_name)] = {
                "mean_rank_90": mean_r90,
                "avg_r32_energy": avg_r32_energy,
            }

    return results


def diagonal_scaling_analysis(weights, num_layers, enc_layers, dec_layers):
    print("\n" + "=" * 60)
    print("  4. DIAGONAL SCALING BENEFIT")
    print("=" * 60)

    results = {}
    for name, layer_ws in weights.items():
        w_mean = torch.stack(layer_ws).mean(dim=0)

        print(f"\n--- {name} {tuple(layer_ws[0].shape)} ---")
        print(f"  {'Layer':>5} | {'No diag':>10} | {'With diag':>10} | {'Reduction':>10}")
        print("  " + "-" * 50)

        reductions = []
        for i, w in enumerate(layer_ws):
            residual_plain = (w - w_mean).norm().item()

            # Alternating least squares for optimal d, e
            # W_i ≈ diag(d) @ W_mean @ diag(e)
            d = torch.ones(w.shape[0])
            e = torch.ones(w.shape[1])

            for _als_iter in range(3):
                # Fix d, solve for e
                # min_e ||W_i - diag(d) @ W_mean @ diag(e)||^2
                # For each column k: e_k = sum_j(W_i[j,k] * d[j] * W_mean[j,k]) / sum_j((d[j]*W_mean[j,k])^2)
                dW = d.unsqueeze(1) * w_mean  # (out, in)
                num_e = (w * dW).sum(dim=0)
                den_e = (dW ** 2).sum(dim=0) + 1e-8
                e = num_e / den_e

                # Fix e, solve for d
                We = w_mean * e.unsqueeze(0)  # (out, in)
                num_d = (w * We).sum(dim=1)
                den_d = (We ** 2).sum(dim=1) + 1e-8
                d = num_d / den_d

            approx = d.unsqueeze(1) * w_mean * e.unsqueeze(0)
            residual_diag = (w - approx).norm().item()
            reduction = (1 - residual_diag / residual_plain) * 100 if residual_plain > 1e-8 else 0

            print(f"  {i:>5} | {residual_plain:>10.2f} | {residual_diag:>10.2f} | {reduction:>9.1f}%")
            reductions.append(reduction)

        mean_red = sum(reductions) / len(reductions)
        print(f"  Mean reduction: {mean_red:.1f}%")
        results[name] = mean_red

    return results


def print_verdict(all_distances, residual_results, diag_results):
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    # Check GO criteria
    # 1. Rank-32 >= 85% for >= 4 of 6 matrix types (global)
    r32_pass = 0
    r32_fail_severe = 0
    for name in MATRIX_TYPES:
        key = (name, "GLOBAL")
        if key in residual_results:
            e = residual_results[key]["avg_r32_energy"]
            if e >= 85:
                r32_pass += 1
            if e < 70:
                r32_fail_severe += 1

    # 2. Mean pairwise distance < 1.0 for >= 4 of 6
    dist_pass = 0
    dist_fail_severe = 0
    for name in MATRIX_TYPES:
        if name in all_distances:
            # Use cross-group (worst case) distance
            d = max(all_distances[name]["enc"], all_distances[name]["dec"], all_distances[name]["cross"])
            if d < 1.0:
                dist_pass += 1
            if d > 1.5:
                dist_fail_severe += 1

    # 3. Check if split sharing helps
    split_r32_pass = 0
    for name in MATRIX_TYPES:
        enc_key = (name, "ENCODER")
        dec_key = (name, "DECODER")
        if enc_key in residual_results and dec_key in residual_results:
            avg = (residual_results[enc_key]["avg_r32_energy"] + residual_results[dec_key]["avg_r32_energy"]) / 2
            if avg >= 90:
                split_r32_pass += 1

    # 4. Diagonal scaling benefit
    diag_useful = sum(1 for v in diag_results.values() if v >= 10)

    print(f"\n  Rank-32 >= 85% energy (global): {r32_pass}/6 matrix types pass")
    print(f"  Rank-32 < 70% energy (global):  {r32_fail_severe}/6 matrix types FAIL")
    print(f"  Pairwise dist < 1.0:            {dist_pass}/6 matrix types pass")
    print(f"  Pairwise dist > 1.5:            {dist_fail_severe}/6 matrix types FAIL")
    print(f"  Split sharing rank-32 >= 90%:   {split_r32_pass}/6 matrix types pass")
    print(f"  Diagonal scaling >= 10% help:   {diag_useful}/6 matrix types")

    # Decision
    if r32_fail_severe >= 3 or dist_fail_severe >= 1:
        verdict = "NO-GO"
        reason = "Layers are too different for weight sharing."
        if r32_fail_severe >= 3:
            reason += f" Rank-32 captures <70% energy for {r32_fail_severe} matrix types."
        if dist_fail_severe >= 1:
            reason += f" Pairwise distance >1.5 for {dist_fail_severe} matrix types."
    elif r32_pass >= 4 and dist_pass >= 4:
        verdict = "GO"
        reason = "Layers share strong structural similarity. Shared base + rank-32 correction is viable."
    elif split_r32_pass >= 4:
        verdict = "CONDITIONAL GO (split sharing)"
        reason = "Global sharing insufficient, but encoder/decoder split bases work."
    else:
        verdict = "CONDITIONAL GO (needs investigation)"
        reason = f"Mixed results: {r32_pass}/6 rank-32 pass, {dist_pass}/6 distance pass."

    print(f"\n  >>> {verdict} <<<")
    print(f"  {reason}")

    # Recommendations
    if "GO" in verdict:
        # Find optimal rank
        best_ranks = []
        for name in MATRIX_TYPES:
            key = (name, "GLOBAL")
            if key in residual_results:
                best_ranks.append(residual_results[key]["mean_rank_90"])
        if best_ranks:
            opt_rank = int(max(best_ranks))
            # Round up to nearest power of 2 or nice number
            for r in [8, 16, 24, 32, 48, 64]:
                if r >= opt_rank:
                    opt_rank = r
                    break
            print(f"\n  Recommended rank: {opt_rank}")

        use_diag = sum(diag_results.values()) / len(diag_results) >= 10
        print(f"  Use diagonal scaling: {'YES' if use_diag else 'NO (< 10% avg benefit)'}")

        sharing = "global" if r32_pass >= 4 else "split (encoder + decoder)"
        print(f"  Sharing strategy: {sharing}")

    print()


def main():
    checkpoint = "final_model.pt"
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]

    if not os.path.exists(checkpoint):
        print(f"ERROR: {checkpoint} not found. Train the baseline first.")
        sys.exit(1)

    print("=" * 60)
    print("  WEIGHT SHARING VIABILITY ANALYSIS")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Size: {os.path.getsize(checkpoint) / 1e6:.1f} MB")

    # Auto-detect num_layers
    sd = torch.load(checkpoint, map_location="cpu")
    num_layers = 0
    while f"blocks.{num_layers}.attn.c_q.weight" in sd:
        num_layers += 1
    del sd

    enc_layers = list(range(num_layers // 2))
    dec_layers = list(range(num_layers // 2, num_layers))

    print(f"  Layers: {num_layers} ({len(enc_layers)} encoder + {len(dec_layers)} decoder)")
    print(f"  Encoder: {enc_layers}, Decoder: {dec_layers}")

    weights = load_weights(checkpoint, num_layers)
    if not weights:
        print("ERROR: No weight matrices found in checkpoint.")
        sys.exit(1)

    cosines = svd_spectrum_analysis(weights, num_layers, enc_layers, dec_layers)
    distances = pairwise_distance_analysis(weights, num_layers, enc_layers, dec_layers)
    residuals = residual_rank_analysis(weights, num_layers, enc_layers, dec_layers)
    diag = diagonal_scaling_analysis(weights, num_layers, enc_layers, dec_layers)
    print_verdict(distances, residuals, diag)


if __name__ == "__main__":
    main()
