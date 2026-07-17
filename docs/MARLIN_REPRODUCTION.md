# MARLIN clean-room reproduction

This branch implements the architecture described in arXiv:2607.04774 without
using unreleased MARLIN code or weights. It must not be described as a bitwise
or author-verified reproduction.

Implemented paper-specified components:

- 896-wide, 12-layer, 14-head Transformer decoder over the 1,880-token SAFE vocabulary;
- width-8 block-causal masked diffusion with continuous-time absorbing NELBO;
- cross-attended Fourier precursor-mass, optional isotope, and sparse Morgan-4096 tokens;
- symmetric fingerprint corruption with `p=0.5` and `rho ~ U(0.1, 0.3)`;
- heavy-atom mass-shell pruning, conservative hydrogen/valence EOS coupling, and 10 ppm acceptance;
- 384 independently perturbed candidates with 0.3 on-bit dropout and fingerprint ranking.

Paper omissions are locked as reproduction assumptions in the config: 100,000
decoder adaptation steps, geometric mass Fourier frequencies, the exact SAFE
prefix grammar, encoder fingerprint threshold, and the precise decoder training
corpus/split. These assumptions must be replaced if the authors release their
implementation.

The formula-free lane uses the official DreaMS encoder plus a frozen
Morgan-4096 probe trained only on the public NPLIB1 training split. The MIST lane
uses a predicted MIST-CF top-1 formula only for peak featurization; the MARLIN
decoder receives only fingerprint and neutral precursor mass.
