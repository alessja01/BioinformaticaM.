from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# -----------------------------
# Data structures
# -----------------------------

Resid = Tuple[int, int]                 # residui(seq_id, pos0) 0-based
LibKey = Tuple[int, int, int, int]      # coppia di residui allineati(i, a, j, b) with i<j
Library = Dict[LibKey, float]           #dizionario: chiave= coppia di residui , valore= peso/confidenzaa


#evita duplicati 
def norm_key(i: int, a: int, j: int, b: int) -> LibKey: 
    """Ensure i<j ordering for library keys."""
    if i < j:
        return (i, a, j, b)
    return (j, b, i, a)


#righe di un MSA
@dataclass
class AlnRow:
    seq_id: int 
    aligned: str                        #sequenza allineata con gap
    col_to_pos: List[Optional[int]]     # aligned column -> original posizione (None if gap)

Profile = List[AlnRow] #lista di righe

# -----------------------------
# Needleman–Wunsch (global, gap lineare) + traceback
# -----------------------------
#allinea due sequenze globalmenre con penalità lineare
def needleman_wunsch_global(
    x: str,
    y: str,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> Tuple[str, str]:
    m, n = len(x), len(y)
    S = [[0] * (n + 1) for _ in range(m + 1)]
    TB = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        S[i][0] = S[i - 1][0] + gap
        TB[i][0] = "U"
    for j in range(1, n + 1):
        S[0][j] = S[0][j - 1] + gap
        TB[0][j] = "L"

    for i in range(1, m + 1):
        xi = x[i - 1]
        for j in range(1, n + 1):
            yj = y[j - 1]
            s_sub = match if xi == yj else mismatch

            diag = S[i - 1][j - 1] + s_sub
            up   = S[i - 1][j] + gap
            left = S[i][j - 1] + gap

            best = max(diag, up, left)
            S[i][j] = best

            if best == diag:
                TB[i][j] = "D"
            elif best == up:
                TB[i][j] = "U"
            else:
                TB[i][j] = "L"

    ai: List[str] = []
    aj: List[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            step = "L"
        elif j == 0:
            step = "U"
        else:
            step = TB[i][j]

        if step == "D":
            ai.append(x[i - 1])
            aj.append(y[j - 1])
            i -= 1
            j -= 1
        elif step == "U":
            ai.append(x[i - 1])
            aj.append("-")
            i -= 1
        else:  # "L"
            ai.append("-")
            aj.append(y[j - 1])
            j -= 1

    ai.reverse()
    aj.reverse()
    return "".join(ai), "".join(aj)

def percent_identity(a: str, b: str) -> float:
    #conta match ignorando colonne con gap
    matches = 0
    aligned = 0
    for x, y in zip(a, b):
        if x == "-" or y == "-":
            continue
        aligned += 1
        if x == y:
            matches += 1
    return (matches / aligned) if aligned else 0.0

# -----------------------------
# 1 Primary library (toy)
# -----------------------------
#per ogni coppia(i,j) fai la NW globale --> calcoli il percentity_identity e scorre la colonna 
#fin quando hai una coppia di residui (non gap) e aggiorni la libreria
def build_primary_library(
    seqs: List[str],
    nw_match: int = 1,
    nw_mismatch: int = -1,
    nw_gap: int = -1,
) -> Library:
    lib: Library = {}
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            ai, aj = needleman_wunsch_global(
                seqs[i], seqs[j], match=nw_match, mismatch=nw_mismatch, gap=nw_gap
            )
            w_source = percent_identity(ai, aj)

            pi = -1
            pj = -1
            for c1, c2 in zip(ai, aj):
                if c1 != "-":
                    pi += 1
                if c2 != "-":
                    pj += 1
                if c1 != "-" and c2 != "-":
                    key = norm_key(i, pi, j, pj)
                    lib[key] = lib.get(key, 0.0) + w_source
    return lib

# -----------------------------
# 2 Library extension (consistency)
# -----------------------------

#costruisce una lista di vicinato neigh
#per ogni coppia (r1↔r2, peso w) aggiunge r2 nei vicini di r1 e viceversa
#poi per ogni cammino a due passi r -> mid -> t aggiunge:
#min(w_rm, w_mt) oppure w_rm*w_mt a seconda di mode
def extend_library(primary: Library, mode: str = "min") -> Library:
    ext = dict(primary)
    neigh: Dict[Resid, List[Tuple[Resid, float]]] = {}

    for (i, a, j, b), w in primary.items():
        r1: Resid = (i, a)
        r2: Resid = (j, b)
        neigh.setdefault(r1, []).append((r2, w))
        neigh.setdefault(r2, []).append((r1, w))

    def f(x: float, y: float) -> float:
        if mode == "min":
            return min(x, y)
        if mode == "prod":
            return x * y
        raise ValueError("mode must be 'min' or 'prod'")

    for r, edges in neigh.items():
        for mid, w_rm in edges:
            for t, w_mt in neigh.get(mid, []):
                if t == r:
                    continue
                (i, a) = r
                (j, b) = t
                if i == j:
                    continue
                key = norm_key(i, a, j, b)
                ext[key] = ext.get(key, 0.0) + f(w_rm, w_mt)

    return ext

# -----------------------------
# 3 Library-based column scoring
# -----------------------------
#calcola uno score : somma dei pesi in libreria per tutte le coppie di residui
#score viene dalla libreria e non dai match/mismatch
def library_column_score(
    colA: List[Optional[Resid]],
    colB: List[Optional[Resid]],
    lib_ext: Library
) -> float:
    score = 0.0
    for ra in colA:
        if ra is None:
            continue
        for rb in colB:
            if rb is None:
                continue
            (i, a) = ra
            (j, b) = rb
            if i == j:
                continue
            score += lib_ext.get(norm_key(i, a, j, b), 0.0)
    return score

# -----------------------------
# 4 Align profiles with DP + traceback (gap affine)
# -----------------------------

#trasforma righe in colonne
def profile_to_columns(profile: Profile) -> List[List[Optional[Resid]]]:
    n_cols = len(profile[0].aligned)
    cols: List[List[Optional[Resid]]] = []
    for c in range(n_cols):
        col: List[Optional[Resid]] = []
        for row in profile:
            pos = row.col_to_pos[c]
            col.append((row.seq_id, pos) if pos is not None else None)
        cols.append(col)
    return cols


#allinea due profili usando la libreria estesa
def align_profiles_with_library(
    profA: Profile,
    profB: Profile,
    lib_ext: Library,
    gap_open: float = -2.0,
    gap_ext: float = -0.5,
) -> Profile:
    colsA = profile_to_columns(profA)
    colsB = profile_to_columns(profB)
    m, n = len(colsA), len(colsB)

    NEG = -1e18
    M  = [[NEG] * (n + 1) for _ in range(m + 1)]
    Ix = [[NEG] * (n + 1) for _ in range(m + 1)]
    Iy = [[NEG] * (n + 1) for _ in range(m + 1)]

    tbM  = [[("?", 0, 0)] * (n + 1) for _ in range(m + 1)]
    tbIx = [[("?", 0, 0)] * (n + 1) for _ in range(m + 1)]
    tbIy = [[("?", 0, 0)] * (n + 1) for _ in range(m + 1)]

    M[0][0] = 0.0

    for i in range(1, m + 1):
        Ix[i][0] = gap_open + (i - 1) * gap_ext
        tbIx[i][0] = ("Ix", i - 1, 0)
    for j in range(1, n + 1):
        Iy[0][j] = gap_open + (j - 1) * gap_ext
        tbIy[0][j] = ("Iy", 0, j - 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = library_column_score(colsA[i - 1], colsB[j - 1], lib_ext)

            candidates = [
                (M[i - 1][j - 1] + s,  ("M",  i - 1, j - 1)),
                (Ix[i - 1][j - 1] + s, ("Ix", i - 1, j - 1)),
                (Iy[i - 1][j - 1] + s, ("Iy", i - 1, j - 1)),
            ]
            M[i][j], tbM[i][j] = max(candidates, key=lambda x: x[0])

            cand_ix = [
                (M[i - 1][j] + gap_open, ("M",  i - 1, j)),
                (Ix[i - 1][j] + gap_ext, ("Ix", i - 1, j)),
            ]
            Ix[i][j], tbIx[i][j] = max(cand_ix, key=lambda x: x[0])

            cand_iy = [
                (M[i][j - 1] + gap_open, ("M",  i, j - 1)),
                (Iy[i][j - 1] + gap_ext, ("Iy", i, j - 1)),
            ]
            Iy[i][j], tbIy[i][j] = max(cand_iy, key=lambda x: x[0])

    end_candidates = [(M[m][n], "M"), (Ix[m][n], "Ix"), (Iy[m][n], "Iy")]
    _, state = max(end_candidates, key=lambda x: x[0])

    ops: List[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if state == "M":
            prev_state, pi, pj = tbM[i][j]
            ops.append("D")
        elif state == "Ix":
            prev_state, pi, pj = tbIx[i][j]
            ops.append("A")
        else:
            prev_state, pi, pj = tbIy[i][j]
            ops.append("B")
        state, i, j = prev_state, pi, pj

    ops.reverse()

    merged: List[AlnRow] = [AlnRow(r.seq_id, "", []) for r in (profA + profB)]

    #aggiunge una colonne al profilo finale
    def append_from_profile(rows: List[AlnRow], src: Profile, col_idx: Optional[int]):
        offset = 0 if src is profA else len(profA)
        for r_i, src_row in enumerate(src):
            target = rows[offset + r_i]
            if col_idx is None:
                target.aligned += "-"
                target.col_to_pos.append(None)
            else:
                target.aligned += src_row.aligned[col_idx]
                target.col_to_pos.append(src_row.col_to_pos[col_idx])

    a_col = 0
    b_col = 0
    for op in ops:
        if op == "D":
            append_from_profile(merged, profA, a_col)
            append_from_profile(merged, profB, b_col)
            a_col += 1
            b_col += 1
        elif op == "A":
            append_from_profile(merged, profA, a_col)
            append_from_profile(merged, profB, None)
            a_col += 1
        else:
            append_from_profile(merged, profA, None)
            append_from_profile(merged, profB, b_col)
            b_col += 1

    return merged

# -----------------------------
# Guide tree (UPGMA) + progressive
# -----------------------------

def make_singleton_profiles(seqs: List[str]) -> Dict[int, Profile]:
    profs: Dict[int, Profile] = {}
    for sid, s in enumerate(seqs):
        profs[sid] = [AlnRow(seq_id=sid, aligned=s, col_to_pos=list(range(len(s))))]
    return profs


def pairwise_distance_matrix(
    seqs: List[str],
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> List[List[float]]:
    n = len(seqs)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = needleman_wunsch_global(seqs[i], seqs[j], match, mismatch, gap)
            pid = percent_identity(ai, aj)
            dist = 1.0 - pid
            D[i][j] = D[j][i] = dist
    return D

def upgma_merges(D: List[List[float]]) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Returns:
      merges: list of (ca, cb, newc)
      root_id: final cluster id
    """
    n = len(D)
    active = {i for i in range(n)}
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    next_id = n
    merges: List[Tuple[int, int, int]] = []

    def avg_dist(ca: int, cb: int) -> float:
        A = members[ca]
        B = members[cb]
        tot = 0.0
        cnt = 0
        for i in A:
            for j in B:
                tot += D[i][j]
                cnt += 1
        return tot / cnt if cnt else 0.0

    last_new = None
    while len(active) > 1:
        best_pair = None
        best_val = 1e18
        act_list = sorted(active)
        for ia in range(len(act_list)):
            for ib in range(ia + 1, len(act_list)):
                ca = act_list[ia]
                cb = act_list[ib]
                v = avg_dist(ca, cb)
                if v < best_val:
                    best_val = v
                    best_pair = (ca, cb)

        assert best_pair is not None
        ca, cb = best_pair

        newc = next_id
        next_id += 1

        merges.append((ca, cb, newc))
        members[newc] = members[ca] + members[cb]
        active.remove(ca)
        active.remove(cb)
        active.add(newc)
        last_new = newc

    assert last_new is not None
    return merges, last_new

def progressive_tcoffee_toy(
    seqs: List[str],
    gap_open: float = -2.0,
    gap_ext: float = -0.5,
    nw_match: int = 1,
    nw_mismatch: int = -1,
    nw_gap: int = -1,
    ext_mode: str = "min",
) -> Profile:
    n = len(seqs)
    if n == 0:
        return []
    if n == 1:
        return [AlnRow(0, seqs[0], list(range(len(seqs[0]))))]

    primary = build_primary_library(seqs, nw_match, nw_mismatch, nw_gap)
    lib_ext = extend_library(primary, mode=ext_mode)
    

    D = pairwise_distance_matrix(seqs, nw_match, nw_mismatch, nw_gap)
    merges, root_id = upgma_merges(D)

    profiles = make_singleton_profiles(seqs)

    for ca, cb, newc in merges:
        merged = align_profiles_with_library(
            profiles[ca], profiles[cb], lib_ext,
            gap_open=gap_open, gap_ext=gap_ext
        )
        profiles[newc] = merged

    final_profile = profiles[root_id]
    final_profile.sort(key=lambda row: row.seq_id)
    return final_profile

def print_profile(profile: Profile, names: Optional[List[str]] = None) -> None:
    for row in profile:
        label = f"S{row.seq_id}"
        if names and row.seq_id < len(names):
            label = names[row.seq_id]
        print(f">{label}\n{row.aligned}")

def run_demo():
    seqs = [
        "ACGTGCA",
        "ACGTCGA",
        "ACGAGCA",
        "ACGTGGA",
    ]
    names = [f"Seq{i+1}" for i in range(len(seqs))]

    # --- evidence: build libraries ---
    primary = build_primary_library(seqs)
    #Il programma ha trovato "primary" relazioni residui-residui 
    ext = extend_library(primary)
    #transitiva
    #nuove relazioni residui-residui
    
    print("Primary library links:", len(primary))
    print("Extended library links:", len(ext))

    # --- run full T-Coffee toy pipeline ---
    msa = progressive_tcoffee_toy(seqs)

    print("\nMSA finale T-Coffee toy:")
    print_profile(msa, names=names)


    assert all(len(r.aligned) == len(msa[0].aligned) for r in msa)

    print("\nOK: pipeline completa e allineamento consistente.")

if __name__ == "__main__":
    run_demo()
