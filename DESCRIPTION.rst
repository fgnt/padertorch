PyTorch Framework
========================================

When first working with padertorch, have a look at contrib/examples

A simple example on how to use the padertorch Trainer may be found in
contrib/examples/mask_estimator/simple_train.py

For an examples on how to use the Configurable in combination with the Trainer
refer to: contrib/examples/pit/train.py

All other examples show different approaches for using padertorch and may be
interpreted as specific to the use case and the likes of the example owner

# ToDo:

This module contains functions and classes where the vanilla API is messed up.

The general idea is to move all independent axis to the left if possible. The
exception to this rule of thumb are sequences. It is computational more
efficient to use the steps as outer axis. This also aligns well with how
`torch.nn.utils.rnn.PackedSequence` is defined.

Examples, why the API is seriously broken:
- torch.Tensor.size() vs. torch.nn.utils.rnn.PackedSequence().batch_sizes
- torch.randn(d1, d2, ...) vs. torch.randint(low, high, size=(d1, d2, ...))
- torch.transpose(input, dim0, dim1) although input is already defined

Milestones:
2. Make it possible to decode (=predict) both models
   - Does the batch axis stay? Christoph always wants to allow independent axis.
     Christoph investigates if all ops support independent axis.
   - How do I reconstruct the trained model?

51. Sequence normalization and batch norm with tracking from batch to batch
  - Sequence norm
  - Batch norm


Structures:
 - Module (comparable to chain or chain_list in Chainer, building_blocks in PF)
 - Ops (comparable to ops in PF)


Definitions:
packed: Uses `torch.nn.utils.rnn.PackedSequence`
padded: Uses `padded` and `sequence_length`

padded to packed: `pack_padded_sequence` yields `PackedSequence`
packed to padded: `pad_packed_sequence` yields `Tensor`
