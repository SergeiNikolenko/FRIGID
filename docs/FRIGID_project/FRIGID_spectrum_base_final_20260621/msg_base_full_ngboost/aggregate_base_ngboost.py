import csv, json
from pathlib import Path
base=Path('/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost')
out=base/'aggregate'; out.mkdir(exist_ok=True)
rows=[]; pred=[]; completed=[]
for shard in sorted((base/'shard_outputs').glob('shard_*')):
    d=shard/'detailed_results.csv'; p=shard/'predictions.csv'; a=shard/'aggregate_statistics.json'
    if not d.exists(): continue
    with d.open(newline='') as f: rs=list(csv.DictReader(f)); rows.extend(rs)
    if p.exists():
        with p.open(newline='') as f: pred.extend(list(csv.DictReader(f)))
    completed.append({'shard': shard.name, 'rows': len(rs), 'aggregate': json.loads(a.read_text()) if a.exists() else None})
def fl(r,k):
    try: return float(r.get(k,0) or 0)
    except Exception: return 0.0
n=len(rows)
agg={'total_spectra':n,'completed_shards':len(completed),'token_model':'token_models/models/best_ngboost_MSG.joblib'}
if n:
    agg.update({
      'exact_match_top1': sum(fl(r,'exact_match_top1') for r in rows)/n,
      'exact_match_top10': sum(fl(r,'exact_match_top10') for r in rows)/n,
      'tanimoto_top1_mean': sum(fl(r,'tanimoto_top1') for r in rows)/n,
      'tanimoto_top10_mean': sum(fl(r,'tanimoto_top10') for r in rows)/n,
      'mist_tanimoto_mean': sum(fl(r,'mist_tanimoto') for r in rows)/n,
      'avg_formula_matches': sum(fl(r,'total_formula_matched') for r in rows)/n,
      'avg_predictions_collected': sum(fl(r,'formula_matches_collected') for r in rows)/n,
      'avg_total_generated': sum(fl(r,'total_generated') for r in rows)/n,
      'formula_match_success_rate': sum(1.0 if fl(r,'total_formula_matched')>0 else 0.0 for r in rows)/n,
    })
agg['paper_target_msg_frigid_base_top1']=0.1609
agg['paper_target_msg_frigid_base_top10']=0.1819
(out/'aggregate_statistics.json').write_text(json.dumps(agg, indent=2))
(out/'completed_shards.json').write_text(json.dumps(completed, indent=2))
if rows:
    with (out/'detailed_results.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if pred:
    with (out/'predictions.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=pred[0].keys()); w.writeheader(); w.writerows(pred)
print(json.dumps(agg, indent=2))
