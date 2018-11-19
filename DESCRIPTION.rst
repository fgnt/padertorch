Shallow wrapper functions around PyTorch
========================================

This module contains functions and classes, where the vanilla API is messed up.

The general idea is to move all independent axis to the left if possible. The
exception to this rule of thumb are sequences. It is computational more
efficient to use the steps as outer axis. This also aligns well with how
`torch.nn.utils.rnn.PackedSequence` is defined.

Examples, why the API is seriously broken:
- torch.Tensor.size() vs. torch.nn.utils.rnn.PackedSequence().batch_sizes
- torch.randn(d1, d2, ...) vs. torch.randint(low, high, size=(d1, d2, ...))
- torch.transpose(input, dim0, dim1) although input is already defined

Milestones:
1. Implement two standard models
   - Acoustic model (CBJ)
   - Mask estimator (JHeit)
2. Make it possible to decode (=predict) both models
   - Does the batch axis stay? Christoph always wants to allow independent axis.
     Christoph investigates if all ops support independent axis.
   - How do I reconstruct the trained model?
3. Implement how to combine both models
   - Warm-start (use case for Thomas)
   - Fine-tune a combination
   - Does the source code need to support API compatibility?
   - How to remap parts of the model? Important for warm-start.
4. Merry christmas.
5. Can we use `torch.stft` in BeamNet?
50. Shall all wrappers support `PackedSequence`?
51. Sequence normalization and batch norm with tracking from batch to batch
  - Sequence norm
  - Batch norm
99. Trainer/ framework
100. Resume training
