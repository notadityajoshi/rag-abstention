import json
from collections import Counter, defaultdict

KEY = "nli_sym_t30"   # primary threshold, frozen on pilot
rows = [json.loads(l) for l in open("data/processed/nli_scores.jsonl")]

print("Flag rate (score > 0.5) by condition and subset:")
table = defaultdict(lambda: defaultdict(list))
for r in rows:
    table[r["subset"]][r["condition"]].append(r[KEY] > 0.5)
print(f"  {'subset':22s} {'inconsistent':>12s} {'control':>8s} {'unanswerable':>12s}")
for s in sorted(table):
    cells = [f"{sum(v)/len(v):.2f}" if (v := table[s][c]) else "-" for c in ("inconsistent", "control", "unanswerable")]
    print(f"  {s:22s} {cells[0]:>12s} {cells[1]:>8s} {cells[2]:>12s}")

missed = [r for r in rows if r["condition"] == "inconsistent" and r[KEY] <= 0.5]
print(f"\nMissed inconsistent: {len(missed)}/500")
print("  Why missed (orig answer selected, new answer selected):",
      dict(Counter((r["diag_orig_in_selected"], r["diag_new_in_selected"]) for r in missed)))
print("  Missed with zero aligned pairs:", sum(r["n_aligned_pairs"] == 0 for r in missed))

print("\n3 missed examples that had aligned pairs:")
for r in [m for m in missed if m["top_pair"]][:3]:
    print(f"  [{r['subset']}] score={r[KEY]:.3f} sim={r['top_pair_sim']:.2f}")
    print("    A:", r["top_pair"][0][:180])
    print("    B:", r["top_pair"][1][:180])