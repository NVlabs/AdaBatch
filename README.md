AdaBatch
========

Training deep neural networks with Stochastic Gradient Descent, or its
variants, requires careful choice of both learning rate and batch size.
While smaller batch sizes generally converge in fewer training epochs,
larger batch sizes offer more parallelism and hence better computational
efficiency. We have developed a new training approach that, rather than
statically choosing a single batch size for all epochs, adaptively
increases the batch size during the training process. 

Our method delivers the convergence rate of small batch sizes while
achieving performance similar to large batch sizes. We analyse our
approach using the standard AlexNet, ResNet, and VGG networks operating
on the popular CIFAR-10, CIFAR-100, and ImageNet datasets. Our results
demonstrate that learning with adaptive batch sizes can improve
performance by factors of up to 6.25 on 4 NVIDIA Tesla P100 GPUs while
changing accuracy by less than 1% relative to training with fixed batch
sizes.

Details can be found in our companion paper:

> A. Devarakonda, M. Naumov and M. Garland, "AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks", Technical Report, [ArXiv:1712.02029](https://arxiv.org/abs/1712.02029), December 2017. 


Implementation
--------------

**CIFAR**.  Our implementation of AdaBatch for the CIFAR-10 and
CIFAR-100 datasets is contained in:

    adabatch_cifar.py

This script is based on Wei Yang's [CIFAR example code][] and uses his
PyTorch models, which are contained in the `models` directory.

**ImageNet**.  Our implementation of AdaBatch for the ImageNet dataset
is contained in:

    `adabatch_imagenet.py`

and is based on the PyTorch [ImageNet example code][].

Examples
--------

Scripts to run our experiments can be found in the `tests` directory.
To use these scripts, you need to navigate to the test directory

    $ cd test

Run the CIFAR-10/100 data set with ResNet-20 network and 512 batch size

    $ ./run_cifar_exp.sh resnet 20 512

Run the ImageNet data set with ResNet-50 network and 512 batch size

    $ ./run_adabatch_imagenet.sh resnet 50 512

References
----------

A. Devarakonda, M. Naumov and M. Garland, "AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks", Technical Report, [ArXiv:1712.02029](https://arxiv.org/abs/1712.02029), December 2017. 

[CIFAR example code]: https://github.com/bearpaw/pytorch-classification/tree/master/models/cifar

[ImageNet example code]: https://github.com/pytorch/examples/tree/master/imagenet
