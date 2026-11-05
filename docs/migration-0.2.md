# Migration to version 0.2.0

## Protection mappings

Protection dictionaries must cover all trainable parameters, with exact tensor
shapes and finite values. Gradient/update scales must be in [0, 1]; freeze masks
must be binary. Casts to the model dtype must remain finite. Use explicit ones
for unprotected entries rather than leaving parameters out. EWC requires both
anchor and importance dictionaries and a nonnegative finite coefficient.

## Measurement

Importance and evaluation preserve individual module modes. Importance estimation
preserves pre-existing gradients, evaluates without dropout or batch-normalization
updates, consumes exactly the allowed batches, and rejects empty evidence. This
changes the old batch proxy's measurement mode for custom models with training-only
behaviour. The included historical models have no dropout. Loss diagnostics are
weighted by example count. The batch proxy still squares a minibatch-mean gradient;
it is not a per-example Fisher estimate.

## Checkpoint schema 3

Schemas 1 and 2 are rejected. Keep old files and use their exact original source to
read them; changing a schema number does not recover information missing from a file.
Schema 3 includes all registered buffers, including nonpersistent buffers, their
alias topology, parameter trainability, module modes, optimizer state and loader/RNG
state. Persistent-buffer values must agree between checkpoint representations.

Raw optimizer moments are checked before PyTorch can convert their dtype. Each
native group's hyperparameters are validated explicitly; AMSGrad state must include
its maximum second moment. Invalid state is rejected before live objects change.
A rollback also guards failures during the final commit of a validated restore.

Exact restart remains a trusted-local CPU, epoch-boundary contract. Use ordinary
TensorDataset loaders with default collation, independent explicit generators,
unchanged automatic batching, no worker processes, and native SGD or AdamW.
Explicit/custom batch samplers are rejected rather than being misrepresented as
default batching. Mid-epoch, distributed, GPU and worker-prefetch restart are outside
scope. Caller provenance must describe all external state and training settings.
Restricted loading is not a sandbox or protection against arbitrary untrusted files.

## Experimental evidence

Development records must identify unique trained cells. Selection is recomputed
from its journal before held-out training. Nonfinite and duplicate-key JSON is
rejected. Packaged studies require both provenance and their exact checksum manifest.
All listed files are checked, including plots. Hashes establish consistency, not an
external signature, independent timestamp, or proof against an author rewriting an
entire evidence set coherently. Review Git history and retained original artifacts.

Historical checked results remain unchanged. The new review and release evidence
are separately identified, and repeated runs of the same seeds do not increase the
number of independent evaluation cells.
