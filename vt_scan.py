#!/usr/bin/env python3
"""
VT Scanner  –  NCC pre-filter + SVM strict classifier
======================================================
Usage:
  python3 vt_scan.py --userid <uid> --dataset "2026-06-05 07:16:48"
  python3 vt_scan.py --userid <uid> --dataset all
  python3 vt_scan.py --userid <uid> --dataset all \
          --startdate "2026-06-09 17:35:33" --enddate "2026-06-10 15:38:03"
  python3 vt_scan.py --userid <uid> --dataset all --update_label vt
"""

import argparse, os, sys
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt, correlate
from scipy.interpolate import interp1d
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from bson import ObjectId
from pymongo import MongoClient, ASCENDING

# ── MongoDB connections (try all) ──────────────────────────────────────────────
MONGO_URIS = {
    "web1": "mongodb://biocalculus:IjN75Ve7RTASu9u@31.97.224.96:27017/biocalculus?authSource=biocalculus",
    "web2": "mongodb://biocalculus:3DnsRG8pQ11ul6u@10.100.200.7:27017/biocalculus?authSource=biocalculus",
    "web4": "mongodb://biocalculus:CAclO8k4k3o5@187.127.143.99:27017/biocalculus?authSource=biocalculus",
    "web6": "mongodb://Biocalculus:uEvp35Pvg01dXn2@187.127.188.197:27017/Biocalculus?authSource=Biocalculus"
}
DB_NAME = "biocalculus"

# ── Template directories ───────────────────────────────────────────────────────
VT_TEMPLATE_DIR  = "/home/rootbio/vt/"
NSR_TEMPLATE_DIR = "/home/rootbio/vt/"

# ── Signal ─────────────────────────────────────────────────────────────────────
FS           = 250
TEMPLATE_LEN = 256
PRE_SAMPLES  = TEMPLATE_LEN // 2
ALIGN_SEARCH = int(0.25 * FS)

# ── Thresholds ─────────────────────────────────────────────────────────────────
NCC_THRESHOLD   = 0.75
SVM_PROB_THRESH = 0.80
SIM_DIFF_THRESH = 0.20
MIN_VT_RUN      = 3
MAX_GAP         = 3

AUG_NOISE = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Gate 1: only these labels pass. PAC/PVC/NSR/Undiagnosed are rejected.
# Beats with NO label (None/empty string) are allowed through.
VT_LABEL_KEYWORDS = ["ventricular tachycardia", "vt", "vtach", "v-tach"]


# ═══════════════════════════════════════════════════════════════════════════════
# MongoDB — connect + resolve user
# ═══════════════════════════════════════════════════════════════════════════════

def connect_and_find_user(userid_str: str):
    """
    Try every configured MongoDB URI.
    Try ObjectId form and plain-string form of userid.
    Return (db_handle, uid_value) for the first connection that has data.
    """
    candidates = []
    try:    candidates.append(ObjectId(userid_str))
    except: pass
    candidates.append(userid_str)

    for name, uri in MONGO_URIS.items():
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=6000)
            client.admin.command("ping")
            db = client[DB_NAME]
            for uid_val in candidates:
                # Check both ecg_dataouts and ecg_value_mins
                for col in ["ecg_dataouts", "ecg_value_mins", "graph_marker"]:
                    n = db[col].count_documents({"userid": uid_val}, limit=1)
                    if n > 0:
                        print(f"  Connected : {name}  (userid as {type(uid_val).__name__})")
                        return db, uid_val
        except Exception as e:
            print(f"  [{name}] unreachable: {e}")
            continue

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Template loading
# ═══════════════════════════════════════════════════════════════════════════════

def _unit_norm(waves):
    norms = np.linalg.norm(waves, axis=1, keepdims=True) + 1e-8
    return waves / norms


def _load_npz(path):
    data  = np.load(path, allow_pickle=True)
    waves = data["templates"].astype(np.float32)
    if waves.shape[1] != TEMPLATE_LEN:
        xo = np.linspace(0, 1, waves.shape[1])
        xn = np.linspace(0, 1, TEMPLATE_LEN)
        waves = np.vstack([
            interp1d(xo, r, kind="linear")(xn).astype(np.float32)
            for r in waves
        ])
    return _unit_norm(waves), waves


def load_banks():
    if not os.path.isdir(VT_TEMPLATE_DIR):
        sys.exit(f"ERROR: VT dir not found: {VT_TEMPLATE_DIR}")

    vt_normed, raw_vt = [], []
    for fname in sorted(f for f in os.listdir(VT_TEMPLATE_DIR) if f.endswith(".npz")):
        try:
            normed, raw = _load_npz(os.path.join(VT_TEMPLATE_DIR, fname))
            vt_normed.append(normed)
            raw_vt.extend(raw.tolist())
        except Exception as e:
            print(f"  [warn] {fname}: {e}")

    if not vt_normed:
        sys.exit(f"ERROR: No VT templates loaded from {VT_TEMPLATE_DIR}")

    vt_bank = np.vstack(vt_normed)
    print(f"  Loaded {len(vt_normed)} VT template file(s)  →  {vt_bank.shape[0]} waveforms")

    nsr_normed = []
    if os.path.isdir(NSR_TEMPLATE_DIR):
        for fname in sorted(f for f in os.listdir(NSR_TEMPLATE_DIR)
                            if f.endswith(".npz") and f.startswith("nsr_")):
            try:
                normed, _ = _load_npz(os.path.join(NSR_TEMPLATE_DIR, fname))
                nsr_normed.append(normed)
            except Exception:
                pass

    if nsr_normed:
        nsr_bank = np.vstack(nsr_normed)
        print(f"  Loaded {len(nsr_normed)} NSR template file(s)  →  {nsr_bank.shape[0]} waveforms")
    else:
        print("  No NSR templates found — using inverted-VT as NSR proxy")
        nsr_bank = _unit_norm(-vt_bank.copy())

    return vt_bank, nsr_bank, raw_vt


# ═══════════════════════════════════════════════════════════════════════════════
# SVM
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_beat(beat):
    b = beat.astype(np.float32)
    b -= np.median(b); b /= (b.std()+1e-8); b /= (np.linalg.norm(b)+1e-8)
    return b


def _sim_feats(bn, vt_bank, nsr_bank):
    sv = float((vt_bank  @ bn).max())
    sn = float((nsr_bank @ bn).max())
    return np.array([sv, sn, sv-sn], dtype=np.float32)


def train_svm(vt_bank, nsr_bank):
    X, y = [], []
    for bank, label in [(vt_bank, 1), (nsr_bank, 0)]:
        for t in bank:
            X.append(_sim_feats(t, vt_bank, nsr_bank)); y.append(label)
            for std in AUG_NOISE:
                noisy = _norm_beat(t + np.random.randn(TEMPLATE_LEN).astype(np.float32)*std)
                X.append(_sim_feats(noisy, vt_bank, nsr_bank)); y.append(label)
    X = np.vstack(X).astype(np.float32); y = np.array(y, dtype=np.int32)
    model = Pipeline([("sc", StandardScaler()),
                      ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                                  probability=True, random_state=42))])
    model.fit(X, y)
    acc = float((model.predict(X)==y).mean())*100
    print(f"  SVM trained: {len(X)} samples  "
          f"(VT={int((y==1).sum())} NSR={int((y==0).sum())})  acc={acc:.1f}%")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Signal utilities
# ═══════════════════════════════════════════════════════════════════════════════

def bandpass(x):
    if len(x) < 50: return x
    nyq = 0.5*FS; b,a = butter(2, [3/nyq,25/nyq], btype="band")
    return filtfilt(b, a, x)


def _znorm(x):
    x = x-np.mean(x); return x/(np.std(x)+1e-9)


def _ncc(seg, tpl):
    c = correlate(_znorm(seg.astype(float)), _znorm(tpl.astype(float)), mode="valid")
    return float(np.max(c)) if len(c)>0 else 0.0


def extract_segment(ecg, rpeak):
    lo = max(0, rpeak-ALIGN_SEARCH); hi = min(len(ecg), rpeak+ALIGN_SEARCH)
    if hi<=lo: return None
    centre = lo+int(np.argmax(np.abs(ecg[lo:hi])))
    s = centre-PRE_SAMPLES; e = centre+(TEMPLATE_LEN-PRE_SAMPLES)
    if s<0 or e>len(ecg): return None
    return ecg[s:e].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════════

VT_LABEL_KEYWORDS = ["ventricular tachycardia", "vt", "vtach", "v-tach"]

def is_vt_label(arrhythmia) -> bool:
    """Return True only if the arrhythmia field explicitly says VT."""
    return any(kw in str(arrhythmia or "").lower() for kw in VT_LABEL_KEYWORDS)


def classify_beat(seg, raw_vt, vt_bank, nsr_bank, svm, arrhythmia=None):
    """
    Three-gate classifier:
      Gate 1 – Label  : arrhythmia field must contain a VT keyword.
                        Beats labelled PAC, PVC, NSR, Undiagnosed etc. are
                        immediately rejected — the model should not override
                        the annotated label for non-VT rhythms.
      Gate 2 – NCC    : max NCC vs VT templates >= NCC_THRESHOLD.
      Gate 3 – SVM    : P(VT) > SVM_PROB_THRESH AND sim_diff > SIM_DIFF_THRESH.
    """
    out = dict(ncc_score=0.0, ncc_pass=False, svm_prob=0.0,
               sim_diff=0.0, max_sim_vt=0.0, max_sim_nsr=0.0,
               is_vt=False, stage="label_reject")

    # Gate 1: label must be VT (or unknown/unlabelled — allow those through)
    # Explicitly labelled non-VT rhythms are excluded immediately
    arr_str = str(arrhythmia or "").strip()
    if arr_str and not is_vt_label(arr_str):
        # Has a label, and it is NOT a VT label → skip
        return out

    out["stage"] = "ncc_fail"

    # Gate 2: NCC
    best = max((_ncc(seg[:min(len(seg), len(np.asarray(t)))],
                     np.asarray(t, dtype=np.float32)[:min(len(seg), len(np.asarray(t)))])
                for t in raw_vt), default=0.0)
    out["ncc_score"] = round(best, 4)
    if best < NCC_THRESHOLD: return out
    out["ncc_pass"] = True; out["stage"] = "svm_fail"

    # Gate 3: SVM
    bn = _norm_beat(seg); feats = _sim_feats(bn, vt_bank, nsr_bank)
    prob = float(svm.predict_proba(feats.reshape(1, -1))[0][1])
    diff = float(feats[2])
    out.update(svm_prob=round(prob, 4), sim_diff=round(diff, 4),
               max_sim_vt=round(float(feats[0]), 4),
               max_sim_nsr=round(float(feats[1]), 4))
    if prob > SVM_PROB_THRESH and diff > SIM_DIFF_THRESH:
        out["is_vt"] = True; out["stage"] = "confirmed_vt"
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Episode detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_episodes(confirmed):
    if not confirmed: return [], []
    beats = sorted(confirmed.keys())
    runs, cur = [], [beats[0]]
    for i in range(1, len(beats)):
        if beats[i] <= beats[i-1]+MAX_GAP: cur.append(beats[i])
        else: runs.append(cur); cur = [beats[i]]
    runs.append(cur)
    episodes = [r for r in runs if len(r)>=MIN_VT_RUN]
    ep_set   = {b for ep in episodes for b in ep}
    isolated = [b for r in runs if len(r)<MIN_VT_RUN for b in r if b not in ep_set]
    return episodes, isolated


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def parse_dt(s):
    if not s: return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try: return datetime.strptime(s, fmt)
        except: pass
    return None


def get_all_datasets(db, uid_val, sd_str=None, ed_str=None):
    """Get all databaseid values for this user, optionally filtered by date."""
    dbids = db["ecg_dataouts"].distinct("databaseid", {"userid": uid_val})
    if not dbids:
        # fallback: check ecg_value_mins using ecgvalue_id field
        dbids = db["ecg_value_mins"].distinct("ecgvalue_id", {"userid": uid_val})

    sd, ed = parse_dt(sd_str), parse_dt(ed_str)
    if not sd and not ed:
        return sorted(str(d) for d in dbids)

    filtered = []
    for dbid in dbids:
        dt = parse_dt(str(dbid))
        if dt is None: continue
        if sd and dt < sd: continue
        if ed and dt > ed: continue
        filtered.append(str(dbid))
    return sorted(filtered)


def load_beats_for_session(db, uid_val, dbid):
    """Load beat metadata from ecg_dataouts for one session."""
    docs = list(db["ecg_dataouts"].find(
        {"userid": uid_val, "databaseid": dbid},
        {"indexcounter": 1, "rpeakindex": 1,
         "arrhythmia": 1, "heartrate": 1, "rr_interval": 1, "peaktime": 1}
    ).sort("indexcounter", ASCENDING))
    return docs


def load_ecg_for_session(db, uid_val, dbid):
    """Stitch all ecg_value_mins counters for this session."""
    ecg, ctr = [], 0
    while True:
        doc = db["ecg_value_mins"].find_one(
            {"userid": uid_val, "ecgvalue_id": dbid, "counter": ctr},
            {"ecgvalue": 1}
        )
        if not doc: break
        ecg.extend(doc["ecgvalue"]); ctr += 1
    return np.asarray(ecg, dtype=float) if ecg else np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
# Process one session
# ═══════════════════════════════════════════════════════════════════════════════

def process_session(db, uid_val, dbid, vt_bank, nsr_bank, raw_vt, svm, update_label):
    print(f"\n{'='*60}")
    print(f"  Session : {dbid}")
    print(f"{'='*60}")

    beats = load_beats_for_session(db, uid_val, dbid)
    if not beats:
        print("  No beats found — skipping.")
        return [], []

    ecg = load_ecg_for_session(db, uid_val, dbid)
    if len(ecg) == 0:
        print("  No ECG signal — skipping.")
        return [], []

    ecg = bandpass(ecg)
    print(f"  Beats : {len(beats)}  |  ECG : {len(ecg)} samples ({len(ecg)/FS:.1f} s)")

    confirmed = {}
    counts = dict(label_reject=0, ncc_fail=0, svm_fail=0, vt=0, skip=0)

    for b in beats:
        beat_no = int(b["indexcounter"])
        rpeak   = b.get("rpeakindex")
        if rpeak is None: counts["skip"] += 1; continue
        seg = extract_segment(ecg, int(float(rpeak)))
        if seg is None: counts["skip"] += 1; continue

        arrhythmia = b.get("arrhythmia")
        res = classify_beat(seg, raw_vt, vt_bank, nsr_bank, svm,
                            arrhythmia=arrhythmia)
        res.update(indexcounter=beat_no, rpeak=int(float(rpeak)),
                   heartrate=b.get("heartrate"), rr_interval=b.get("rr_interval"),
                   peaktime=b.get("peaktime"), arrhythmia=arrhythmia,
                   databaseid=dbid)

        if   res["stage"] == "label_reject":  counts["label_reject"] += 1
        elif res["stage"] == "ncc_fail":      counts["ncc_fail"]     += 1
        elif res["stage"] == "svm_fail":      counts["svm_fail"]     += 1
        elif res["stage"] == "confirmed_vt":  counts["vt"]           += 1

        if res["is_vt"]: confirmed[beat_no] = res

    print(f"  Label rejected: {counts['label_reject']} (PAC/PVC/NSR/etc.)")
    print(f"  NCC rejected  : {counts['ncc_fail']}")
    print(f"  SVM rejected  : {counts['svm_fail']}")
    print(f"  Confirmed VT  : {counts['vt']}")

    episodes, isolated = detect_episodes(confirmed)

    # ── Output ─────────────────────────────────────────────────────────────────
    print("\n--- DETECTED VT BEATS (runs only) ---")
    if not episodes and not isolated:
        print("  No VT detected")
    else:
        if episodes:
            for beat in sorted(b for ep in episodes for b in ep):
                r = confirmed[beat]
                print(f"  Beat {beat:6d}  NCC={r['ncc_score']:.3f}  "
                      f"P(VT)={r['svm_prob']:.3f}  diff={r['sim_diff']:+.3f}  "
                      f"hr={r.get('heartrate','?')}  arr={r.get('arrhythmia','?')}")
        if isolated:
            print(f"\n--- ISOLATED VT BEATS ---")
            for beat in isolated:
                r = confirmed[beat]
                print(f"  Beat {beat:6d}  NCC={r['ncc_score']:.3f}  "
                      f"P(VT)={r['svm_prob']:.3f}  diff={r['sim_diff']:+.3f}  "
                      f"hr={r.get('heartrate','?')}  arr={r.get('arrhythmia','?')}")

    print("\n--- VT EVENTS ---")
    if not episodes:
        print("  No VT runs detected")
    else:
        for eid, ep in enumerate(episodes, 1):
            probs  = [confirmed[b]["svm_prob"]  for b in ep if b in confirmed]
            nccs   = [confirmed[b]["ncc_score"] for b in ep if b in confirmed]
            ptimes = [confirmed[b]["peaktime"]  for b in ep
                      if b in confirmed and confirmed[b].get("peaktime")]
            t = f"  time {ptimes[0]}->{ptimes[-1]}" if ptimes else ""
            print(f"  VT run {eid}: beats {ep[0]}->{ep[-1]}  "
                  f"({len(ep)} beats){t}  "
                  f"mean_P(VT)={np.mean(probs):.3f}  "
                  f"mean_NCC={np.mean(nccs):.3f}")

    print("\n--- TOP 5 VT BEATS BY SVM CONFIDENCE ---")
    all_ep_beats = [b for ep in episodes for b in ep] + isolated
    if not all_ep_beats:
        print("  No VT beats to rank")
    else:
        top5 = sorted(
            [(b, confirmed[b]["svm_prob"]) for b in all_ep_beats if b in confirmed],
            key=lambda x: x[1], reverse=True
        )[:5]
        for beat_no, prob in top5:
            r = confirmed[beat_no]
            print(f"  Beat {beat_no:6d}  P(VT)={prob:.4f}  "
                  f"NCC={r['ncc_score']:.3f}  diff={r['sim_diff']:+.3f}  "
                  f"hr={r.get('heartrate','?')}  arr={r.get('arrhythmia','?')}")

        if update_label and update_label.lower() == "vt":
            updated = 0
            for beat_no, _ in top5:
                r = confirmed[beat_no]
                res = db["ecg_dataouts"].update_one(
                    {"userid": uid_val, "indexcounter": beat_no, "databaseid": dbid},
                    {"$set": {"label": "vt",
                               "vt_ncc_score": r["ncc_score"],
                               "vt_svm_prob":  r["svm_prob"],
                               "vt_sim_diff":  r["sim_diff"]}}
                )
                if res.modified_count: updated += 1
            print(f"\n  MongoDB: {updated}/5 beats tagged with label='vt'")

    return episodes, isolated


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--userid",       required=True)
    ap.add_argument("--dataset",      required=True,
                    help="databaseid e.g. '2026-06-05 07:16:48', or 'all'")
    ap.add_argument("--startdate",    default=None, help="YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--enddate",      default=None, help="YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--update_label", default=None,
                    help="Set to 'vt' to tag confirmed beats in MongoDB")
    args = ap.parse_args()

    print("\n=== VT TEMPLATE SCANNER ===")

    # 1. Load templates + train
    np.random.seed(42)
    vt_bank, nsr_bank, raw_vt = load_banks()
    print("\nTraining SVM ...")
    svm = train_svm(vt_bank, nsr_bank)
    print(f"Gates: NCC>={NCC_THRESHOLD}  P(VT)>{SVM_PROB_THRESH}  "
          f"sim_diff>{SIM_DIFF_THRESH}  min_run={MIN_VT_RUN}  max_gap={MAX_GAP}")

    # 2. Connect — try all MongoDB servers
    print("\nConnecting to MongoDB ...")
    db, uid_val = connect_and_find_user(args.userid)
    if db is None:
        print(f"ERROR: No MongoDB connection found data for userid={args.userid}")
        sys.exit(1)

    # 3. Resolve datasets
    if args.dataset.strip().lower() == "all":
        datasets = get_all_datasets(db, uid_val, args.startdate, args.enddate)
        if not datasets:
            print(f"\nNo sessions found for userid={args.userid}")
            if args.startdate or args.enddate:
                print(f"  startdate={args.startdate}  enddate={args.enddate}")
            # Show what sessions DO exist to help debug
            all_ds = get_all_datasets(db, uid_val)
            if all_ds:
                print(f"  Available sessions ({len(all_ds)}):")
                for ds in all_ds[:10]:
                    print(f"    {ds}")
                if len(all_ds) > 10:
                    print(f"    ... and {len(all_ds)-10} more")
            return
        print(f"\nFound {len(datasets)} session(s):")
        for ds in datasets:
            print(f"  {ds}")
    else:
        datasets = [args.dataset.strip()]

    # 4. Process each session
    all_episodes, all_isolated = [], []
    for dbid in datasets:
        eps, iso = process_session(
            db, uid_val, dbid, vt_bank, nsr_bank, raw_vt, svm,
            args.update_label,
        )
        for ep in eps:  all_episodes.append((dbid, ep))
        for b  in iso:  all_isolated.append((dbid, b))

    # 5. Grand summary (multi-session)
    if len(datasets) > 1:
        print(f"\n{'='*60}")
        print(f"  GRAND SUMMARY  —  {len(datasets)} sessions")
        print(f"{'='*60}")
        if not all_episodes and not all_isolated:
            print("  No VT detected across all sessions.")
        else:
            if all_episodes:
                print(f"  VT episodes : {len(all_episodes)}")
                for dbid, ep in all_episodes:
                    print(f"  [{dbid}]  beats {ep[0]}->{ep[-1]}  ({len(ep)} beats)")
            if all_isolated:
                print(f"  Isolated VT : {len(all_isolated)} beats")
                for dbid, b in all_isolated:
                    print(f"  [{dbid}]  beat {b}")

    print("\n================================")


if __name__ == "__main__":
    main()
