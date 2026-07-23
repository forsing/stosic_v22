from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v22_2csv.py — 7-node krug (K=7 / prilagodjenje 7/39) — Global optimality / local excess coexistence (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/global_optimality.pvs
  — local phys > opt (DAY→DUSK excess) AND global phys ≈ opt (DAY→NIGHT)
  — lem_coexistence: lokalni višak ne pobija globalnu geodeziju
  — ax_triangle: opt_AC ≤ opt_AB + opt_BC

Mapiranje na 7/39:
  c=||x_i−x_j||²; W² = matching cost/7
  phys_AB = prosek c na svim 7×7 ivicama (neoptimalan coupling)
  opt_AB, opt_BC, opt_AC = W² matching
  local_excess := phys_AB > opt_AB
  global_ok := (opt_AB+opt_BC − opt_AC) ≤ medijan istog jaza
  ako coexistence: skor += brojevi kola t+2 (puni luk A→C)
  next = top 7; bez randoma; stop ako uzastopni/AP
"""

from typing import List

import numpy as np

from stosic_v1_2csv import CSV_LOTO, CSV_PLUS, EPS, MAX_NUM, N_PICK, load_draws
from stosic_v2_2csv import top7_from_freq
from stosic_v8_2csv import cooccurrence_features, cost_matrix
from stosic_v9_2csv import optimal_matching_support
from stosic_v10_2csv import is_degenerate
from stosic_v12_2csv import w2_sq_draws


def phys_avg_cost(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    src = [int(n) - 1 for n in a]
    tgt = [int(n) - 1 for n in b]
    s = 0.0
    for i in src:
        for j in tgt:
            s += float(C[i, j])
    return s / float(N_PICK * N_PICK)


def predict_next(draws: np.ndarray) -> List[int]:
    if len(draws) < 3:
        raise SystemExit("CSV prekratak")
    C = cost_matrix(cooccurrence_features(draws))
    gaps = []
    triples = []
    for t in range(len(draws) - 2):
        opt_ab = w2_sq_draws(draws[t], draws[t + 1], C)
        opt_bc = w2_sq_draws(draws[t + 1], draws[t + 2], C)
        opt_ac = w2_sq_draws(draws[t], draws[t + 2], C)
        phys_ab = phys_avg_cost(draws[t], draws[t + 1], C)
        add_gap = (opt_ab + opt_bc) - opt_ac  # ≥0 (trougao)
        local_excess = phys_ab > opt_ab + EPS
        gaps.append(add_gap)
        triples.append((t, local_excess, add_gap))
    med = float(np.median(gaps)) if gaps else 0.0
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    for t, local_excess, add_gap in triples:
        # lem_coexistence: lokalni višak + globalno blizu geodezije
        if local_excess and add_gap <= med + EPS:
            for n in draws[t + 2]:
                skor[int(n) - 1] += 1.0
    if float(skor.sum()) <= 0:
        for d in draws:
            for n in d:
                skor[int(n) - 1] += 1.0
    combo = top7_from_freq(skor)
    if is_degenerate(combo):
        nu = np.zeros(MAX_NUM, dtype=np.float64)
        for d in draws:
            for n in d:
                nu[int(n) - 1] += 1.0
        combo = top7_from_freq(nu)
    return combo


def main():
    next_loto = predict_next(load_draws(CSV_LOTO))
    next_loto_plus = predict_next(load_draws(CSV_PLUS))
    if is_degenerate(next_loto):
        raise SystemExit("degenerisan next_loto (uzastopni/AP) — zaustavljen pre ispisa")
    if is_degenerate(next_loto_plus):
        raise SystemExit("degenerisan next_loto_plus (uzastopni/AP) — zaustavljen pre ispisa")
    print("next_loto:      ", next_loto)
    print("next_loto_plus: ", next_loto_plus)


if __name__ == "__main__":
    main()



"""
next_loto:       [8, x, 24, y, 32, z, 38]
next_loto_plus:  [7, x, 9, y, 23, z, 34]
"""



"""
v22: global_optimality — skor kad lokalni excess + globalni luk blizu geodezije.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
