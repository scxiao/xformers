
This branch is to show the impact of the buffer ops on the kernel performance. The kernel is the attn decode kernel with fp8 and bf16 as input.

We use the triton compiler branch: https://github.com/ROCm/triton/tree/pytorch/rocm7.1_internal_testing_hstu_drop_1. This branch contains a few optimizations for the attn decode kernel, so it has better performance than the upstream main branch. 


command line to run the kernel is:

Use buffer ops:
```
AMDGCN_USE_BUFFER_OPS=1 python benchmark_attn_decoding.py
```

Not use buffer ops:
```
AMDGCN_USE_BUFFER_OPS=0 python benchmark_attn_decoding.py
```

Corresponding performance numbers are:
```
bf16_buffer_load: 117.3us
bf16_no_buffer_load: 120.7us
fp8_buffer_load: 70.9us
fp8_no_buffer_load: 93.7us

```

We can see buffer ops for input load can uplift the kernel performance, particularly for the fp8 data type

The corresponding ir dumps are at: 

[bf16_buffer_load](https://github.com/scxiao/xformers/tree/scxiao/attn_decode_buffer_ops_showcase/xformers/benchmarks/ir_dump_buffer_ops_bf16)
[bf16_no_buffer_load](https://github.com/scxiao/xformers/tree/scxiao/attn_decode_buffer_ops_showcase/xformers/benchmarks/ir_dump_no_buffer_ops_bf16)
[fp8_buffer_load](https://github.com/scxiao/xformers/tree/scxiao/attn_decode_buffer_ops_showcase/xformers/benchmarks/ir_dump_buffer_ops_fp8)
[fp8_no_buffer_load](https://github.com/scxiao/xformers/tree/scxiao/attn_decode_buffer_ops_showcase/xformers/benchmarks/ir_dump_no_buffer_ops_fp8)


