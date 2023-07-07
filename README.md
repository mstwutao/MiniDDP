# MiniDDP
Minimal implementation of DataParallel and DistributedDataParallal tutorial

DataParallel (DP) is single-process, multi-thread, and only works on a single machine. We scatter the data throughout the GPUs and perform forward passes in each one of them. Essentially, what happens is that the batch size is divided across the number of workers. the batch size should be divisible by the number of GPUs
