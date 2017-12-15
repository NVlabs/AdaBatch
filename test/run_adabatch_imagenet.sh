# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/bin/bash

#This script performs experiments with the ImageNet data set.
#Make sure that this points to a Python 2.7 executable.
PYTHON=python

#Parameters:
#1. location of the data set
dset=/mnt/datasets/ImageNet

#2. network architecture and batch size
arch=$1
depth=$2

#3. batch size, epochs and trials
batch_size=$3
nepochs=90
ntrials=1 #5

#4. adaptive batch resize factor (1-fixed; k-adaptive, increasing by k)
batch_resize_factor=1 #2

#5. GPUs, IO and trials
ngpu=1 #4
gpu_id=0 #,1,2,3
workers=20 #40

#Run several trials 
for i in `seq 1 $ntrials`;
do
    #Parameters
    echo "Parameters:"
    echo "data_set=$dset"
    echo "arch=$arch$depth"
    echo "batch_size=$batch_size(x$batch_resize_factor)"
    #echo "nepochs=$nepochs"
    echo "ntrials=$ntrials"
    #echo "ngpu=$ngpu"
    #echo "gpu_id=$gpu_id"
    echo "io_workers=$workers"

    #Experiment 
    $PYTHON ../adabatch_imagenet.py $dset -a $arch$depth -b $batch_size  --warmup 5 --baseline-batch 256 --zero-grad-freq 256 --lr-factor .1 --resize-freq 3
0 --resize-factor $batch_resize_factor --workers $workers --filename maxim_imagenet_mb131072_resnet50_fixed_.1lr_accum_lrwarmup 2>&1 | tee imagenet_mb"$batch_size"_"$arch$depth"_dyn_trial"$i".log
done
