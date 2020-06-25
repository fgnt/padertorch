# Virual minibatch size and multiple GPUs

Note: The code examples here should be interpreted as pseudo code, i.e. they should show what the code does. The actual implementation is more complicated, because it does more things.

## What is (Virual) minibatch size?

In `padertorch` you have two options to use a minibatch size for your model.

The first way is to incude it in your train dataset (i.e. batch multiple examples together inside of your datapreprocessing).
Then you model has to work on multiple examples.
The `padertorch.Trainer` will not recognice this kind of minibatch size, because the trainer simpley forwards the examples from your dataset to the model:

```python
dataset = do_batching(dataset)                      # <-----
for batch in dataset:                             # <-----
    batch = model.example_to_device(batch)
    review = model.review(example, model(batch))  # <-----
    loss = loss_from_review(review)
    report_to_tensorboard(review)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

As an example how the batching can be done on the fly with `lazy_dataset` see `padertorch/contrib/examples/tasnet/train.py`.

The second option is the `virtual_minibatch_size` argument of `padertorch.Trainer`.
With this option you can increase the minibatch size without changing your dataset or your model. Only the trainer will handle the batch size:

```python
i = 0                                               # <-----
for example in dataset:
    example = model.example_to_device(example)
    review = model.review(example, model(example))
    loss = loss_from_review(review)
    report_to_tensorboard(review)
    loss.backward()
    i += 1
    if i == virtual_minibatch_size:                 # <-----
        i = 0                                       # <-----
        optimizer.step()                            # <-----
        optimizer.zero_grad()
```

Both options for the minibatch size can be combined.
The effective minibatch size for the optimizer will the the dataset minibatch size times the `virtual_minibatch_size`. For operations like batch normalization (i.e. operations that work in the minibatch axis) the batch size will be the minibatch that is used to produce the dataset, NOT the virtual batch size.

## Why use (Virtual) minibatch size?
There are multiple arguments why using a minibatch and theoretical arguments.
Here, we will limit us to practical aspects and argue, why you may want to use the minibatch size in your dataset or in the trainer.

When you increase the minibatch size in your dataset in many cases you will observe that the runtime on a GPU only slightly increases.
So, you process more examples in the same time and your training finishes earlier (The converence properties will also change, but this is not a point to discuss here).

The `virtual_minibatch_size` has no speedup effect (The optimizer has no relevant runtime).
So why may you want to use the `virtual_minibatch_size` anyway?
When you increase the minibatch size in your dataset, this will also increase the memory consumption of your model.
So the memory capacity of your GPU limits the maximum minibatch size in your dataset.
If your model has better convergence properties with a larger minibatch size the `virtual_minibatch_size` can be used.

In practice you increase the minibatch size in your dataset to the maximum value that fits on your GPU and than you start to increase the `virtual_minibatch_size` if you want to have a larger minibatch size.

## Multiple GPUs

Your first question to this document may be, why is this document about virual minibatch size and multiple GPUs.
The answer is: we combined the `virtual_minibatch_size` and multiple GPUs in the trainer.

Note: The implementation of multiple GPUs is based on `torch.nn.DataPatallel`, i.e. we use the functions that are used in that class, but slightly different.

Here is some peudo code, how we use multiple GPUs:

```python
def parallel_task(model, example, devive):              # <-----
    example = model.example_to_device(example, devive)
    review = model.review(example, model(example))
    return review

def yield_two_examples(dataset)                         # <-----
    examples = []
    for example in dataset:
        examples.append(example)
        if len(examples) == 2:
            yield examples
            examples = []

i = 0
for example1, example2 in yield_two_examples(dataset):  # <-----

    # model1, model2 = replicate(model, [0, 1])
    review1, review2 = parallel_apply(                  # <-----
        parallel_task,
        [
            (model.to(0), exmaple1, 0),
            (model.to(1), exmaple1, 1),
        ]
    )

    loss = loss_from_review(review1) + loss_from_review(review2)
    report_to_tensorboard(review1)
    report_to_tensorboard(review2)
    loss.backward()
    i += 1
    if i == virtual_minibatch_size // 2:                # <-----
        i = 0
        optimizer.step()
        optimizer.zero_grad()
```

### What is the difference to torch.nn.DataParallel?

We use the functions that are also used in `torch.nn.DataParallel`. However, inside the trainer we have more control and can apply them more elegantly (`torch.nn.DataParallel` is a more general class and cannot do these things).

First, we do not need to use `scatter` to split the example. We simply take two examples that follow each other.
For `torch.nn.DataParallel` you should move the data to the GPU, before it will be scatterd to the other GPUs.
We move the data in the thread to the device.
The `replicate` and `parallel_apply` is the same for both.
For the `gather` operation, we only need to gather the loss and report the remaining stuff directly to tensorboardX. In `torch.nn.DataParallel` erverything has to be gathered.
