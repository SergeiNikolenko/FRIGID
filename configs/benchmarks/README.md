# MassSpecGym benchmark manifests

The `msg_diverse_*_v1.tsv` files define molecule-diverse, disjoint subsets of
the 17,082-spectrum FRIGID/MassSpecGym overlap. They prevent small decoding
gates from being dominated by contiguous blocks of replicate spectra from the
same molecule.

Selection procedure:

1. Load the exact FRIGID test split and intersect it with the official MS-BART
   MassSpecGym TSV by spectrum identifier.
2. Reduce InChIKeys to the connectivity block and keep one spectrum per unique
   molecule.
3. Sort by `sha256("frigid-msg-diverse-v1\0" + spec_name)`.
4. Use disjoint slices for development (first 64), confirmation (next 200),
   and the 1,024-spectrum gate (next 1,024).

Ordered spectrum-name hashes:

| Manifest | Rows | SHA-256 |
| --- | ---: | --- |
| `msg_diverse_dev64_v1.tsv` | 64 | `c46b6354c05b030c62ef1c1ce9c8e92001d00226f5964809d96c0b4e7fd952a2` |
| `msg_diverse_confirm200_v1.tsv` | 200 | `49702607a38766ee16468e7c04c3cf35f71759b66f92ebc49eb00a0e81a36962` |
| `msg_diverse_gate1024_v1.tsv` | 1,024 | `6e40e6a6f3dba4c34fc0f8b06d739e78fafeb6a7aa587319b70374b732d43763` |

Run a manifest with `benchmark_dlm_fingerprint_robustness.py --spec-manifest
<path>`. Do not combine a manifest with a nonzero `--start-index`, and do not
truncate it with a different `--max-spectra` value.
