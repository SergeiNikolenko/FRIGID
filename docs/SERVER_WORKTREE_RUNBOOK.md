# Server Worktree Runbook

## Safe deployment

Never pull over a dirty or active checkout. Deploy an exact remote branch to a
dedicated worktree:

```bash
cd ~/work/Projects/FRIGID
git fetch origin research/msg-quality-gates
git worktree add ~/work/Projects/FRIGID_quality_gates \
  origin/research/msg-quality-gates
```

For an existing dedicated worktree:

```bash
cd ~/work/Projects/FRIGID_quality_gates
git fetch origin research/msg-quality-gates
git checkout --detach origin/research/msg-quality-gates
```

Before launching a run, record:

```bash
git rev-parse HEAD
git status --short
sha256sum <manifest> <checkpoint>
```

The worktree must be clean. Data, checkpoints, and run outputs must remain
outside Git or under ignored runtime directories. Do not delete or replace an
active run directory.

## Persistent execution

Use `zellij` for jobs expected to run longer than one hour when available;
otherwise use `tmux`. Name the session and run directory after the Linear issue
and frozen experiment variant. Write an exit-status file and preserve stdout,
resolved configuration, environment, code commit, input hashes, and progress
state.

## Compact source semantics

The `micro128/256/512` panels intentionally retain repeated spectra from the
same connectivity block. A MolForge or external-generator launcher must check
unique spectrum IDs, label/formula/InChI alignment, and test-split membership,
but must not require molecule uniqueness for micro panels. `macro64` is the
exception: it must remain molecule-unique and disjoint from all micro panels.

Legacy 1,024-spectrum launchers commonly enforce a molecule-unique manifest and
must be adapted before use. A failed preflight is an orchestration failure, not
model evidence; preserve its log separately and never mix its partial outputs
with a valid run.

For every external generator, convert its ordered JSONL output with
`scripts/convert_molforge_predictions.py` (or the equivalent adapter), verify
that the ordered manifest hash matches, and only then pass the candidate CSV to
`fuse_candidate_sources.py`.

## Server roles

- The local machine orchestrates and performs lightweight paired analysis.
- GPU servers perform inference and training.
- A server's dirty main checkout is historical state, not the deployment target.
- The dedicated `FRIGID_quality_gates` worktree is the source of agent
  instructions and locked benchmark manifests.
