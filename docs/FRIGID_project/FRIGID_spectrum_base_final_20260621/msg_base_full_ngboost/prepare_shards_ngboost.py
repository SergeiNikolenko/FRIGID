import csv
import json
import math
import os
from pathlib import Path

root = Path('/home/nikolenko/work/Projects/FRIGID')
run = root / 'repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost'
source = root / 'repro_cache/msg'
labels = source / 'labels.tsv'
split = source / 'split.tsv'
shard_size = int(os.environ.get('SHARD_SIZE', '500'))
with split.open() as handle:
    rows = list(csv.DictReader(handle, delimiter='\t'))
test_names = [row['name'] for row in rows if row['split'] == 'test']
num_shards = math.ceil(len(test_names) / shard_size)
for idx in range(num_shards):
    names = set(test_names[idx * shard_size:(idx + 1) * shard_size])
    shard_dir = run / 'shard_data' / f'shard_{idx:03d}'
    shard_dir.mkdir(parents=True, exist_ok=True)
    for name in ['labels.tsv', 'atom_types.txt', 'edge_types.txt', 'n_counts.txt']:
        target = shard_dir / name
        if not target.exists():
            target.symlink_to(source / name)
    for name in ['spec_files', 'subformulae', 'neuraldecipher']:
        target = shard_dir / name
        if not target.exists():
            target.symlink_to(source / name, target_is_directory=True)
    with (shard_dir / 'split.tsv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['name', 'split'], delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({'name': row['name'], 'split': 'test' if row['name'] in names else 'ignore'})
manifest = {
    'source_labels': str(labels),
    'source_split': str(split),
    'test_samples': len(test_names),
    'shard_size': shard_size,
    'num_shards': num_shards,
    'paper_target_msg_frigid_base_top1': 0.1609,
    'paper_target_msg_frigid_base_top10': 0.1819,
    'token_model': 'token_models/models/best_ngboost_MSG.joblib',
    'note': 'FRIGID-base benchmark_spec2mol sharded over detected MSG test split with NGBoost token length model enabled.',
}
(run / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
