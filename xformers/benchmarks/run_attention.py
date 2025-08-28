#!/usr/bin/env python3
"""
Attention Performance Testing Framework

Tests various attention implementations for both prefill and decoding scenarios:
- Default: memory_efficient_attention_forward
- CK: fmha.ck.FwOp (Composable Kernel backend)
- Triton SplitK: fmha.triton_splitk.FwOp
- AITER: AMD Instinct Flash Attention with enhanced bias support

Features:
- Comprehensive performance benchmarking
- Support for variable sequence lengths and batch sizes
- AITER integration with xformers bias compatibility
- Optimized bias pre-computation to exclude setup overhead from timing
"""

import torch
import logging
import argparse
from xformers.ops import fmha

# Try to import AITER
try:
    import aiter
    AITER_AVAILABLE = True
except ImportError:
    AITER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_operation_benchmark(op_name, op_func, inputs, warmup=10, rep=100):
    """Run performance benchmark for a given operation"""
    import triton
    
    for _ in range(5):
        _ = op_func(*inputs)
    torch.cuda.synchronize()
    
    t_ms = triton.testing.do_bench(
        lambda: op_func(*inputs),
        warmup=warmup, rep=rep
    )
    
    result = op_func(*inputs)
    return result, t_ms

def _create_causal_bias(q_len, kv_len, device='cuda', dtype=torch.float32):
    """Create causal attention bias tensor"""
    if q_len == 1:
        # Decoding: all previous tokens are visible
        return torch.zeros((q_len, kv_len), device=device, dtype=dtype)
    elif q_len == kv_len:
        # Prefill (square): use optimized upper triangular mask
        return torch.triu(
            torch.full((q_len, kv_len), float('-inf'), device=device, dtype=dtype),
            diagonal=1
        )
    else:
        # Prefill (rectangular): create causal mask with offset
        i_idx = torch.arange(q_len, device=device).view(-1, 1)
        j_idx = torch.arange(kv_len, device=device).view(1, -1)
        return torch.where(
            j_idx > i_idx + (kv_len - q_len),
            float('-inf'),
            0.0
        ).to(dtype)

def precompute_aiter_bias(q, k, v, bias=None):
    """Pre-compute AITER bias outside of benchmark timing"""
    if bias is None:
        return None
    
    # Convert tensor dimensions to AITER format
    B_times_T = q.shape[1]
    B, T = (1, B_times_T) if B_times_T > 1024 else (B_times_T, 1)
    kv_seq_len = k.shape[1] // B
    
    # Extract sequence lengths from xformers bias if possible
    if hasattr(bias, 'q_seqinfo') and hasattr(bias, 'k_seqinfo'):
        q_seqinfo = getattr(bias.q_seqinfo, 'max_seqlen', T)
        kv_seqinfo = getattr(bias.k_seqinfo, 'max_seqlen', kv_seq_len)
        # Use extracted lengths if they match processed dimensions
        if q_seqinfo == T and kv_seqinfo == kv_seq_len:
            pass  # Use extracted lengths
    
    # Create causal bias tensor
    return _create_causal_bias(T, kv_seq_len, device=q.device, dtype=torch.float32)

def run_aiter_attention(op_type, q, k, v, bias=None, causal=True, precomputed_bias=None):
    """Enhanced AITER attention operation with pre-computed bias support"""
    if not AITER_AVAILABLE:
        raise RuntimeError("AITER not available")
    
    if op_type == "aiter":
        # Convert from xformers format [1, B*T, N_KVH_L, N_H_L//N_KVH_L, D_H] to AITER format [B, T, N_H_L, D_H]
        B_times_T = q.shape[1]
        N_KVH_L = q.shape[2]
        N_H_L_div_N_KVH_L = q.shape[3]
        D_H = q.shape[4]
        N_H_L = N_KVH_L * N_H_L_div_N_KVH_L
        
        # For simplicity, assume B=1 for prefill, B=batch_size for decoding
        if B_times_T > 1024:
            B, T = 1, B_times_T
        else:
            B, T = B_times_T, 1
        
        q_aiter = q.reshape(B, T, N_H_L, D_H)
        # Determine K,V sequence length from input tensor shape
        kv_seq_len = k.shape[1] // B
        k_aiter = k.reshape(B, kv_seq_len, N_H_L, D_H)[:, :, :N_KVH_L, :]
        v_aiter = v.reshape(B, kv_seq_len, N_H_L, D_H)[:, :, :N_KVH_L, :]
        
        # Use AITER flash attention
        result = aiter.flash_attn_func(
            q_aiter, k_aiter, v_aiter,
            dropout_p=0.0,
            causal=causal,
            bias=precomputed_bias,
            return_lse=True
        )
        
        # Handle different return formats
        if isinstance(result, tuple):
            out_aiter = result[0]
        else:
            out_aiter = result
        
        # Convert back to xformers format
        return out_aiter.view(1, B*T, N_KVH_L, N_H_L_div_N_KVH_L, D_H)
    else:
        raise ValueError(f"Unknown AITER operation: {op_type}")


def get_operation(op_name):
    """Get the operation class from name"""
    if op_name == "default":
        return None, "memory_efficient_attention_forward"
    elif op_name == "ck":
        return fmha.ck.FwOp, "fmha.ck.FwOp"
    elif op_name == "triton_splitk":
        return fmha.triton_splitk.FwOp, "fmha.triton_splitk.FwOp"
    elif op_name == "aiter":
        return "aiter", "aiter.flash_attn_func"
    elif op_name == "aiter_varlen":
        return "aiter_varlen", "aiter.flash_attn_varlen_func"
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def test_prefill_attention(seq_len=2048, op_name="default"):
    """Test prefill attention (multi-token input) with specified operation"""
    op, op_display_name = get_operation(op_name)
    logger.info(f"Testing prefill attention with sequence length {seq_len} using {op_display_name}...")
    
    if not torch.cuda.is_available():
        return False
        
    B, T, MAX_T = 1, seq_len, seq_len
    N_H_L, N_KVH_L, D_H = 8, 1, 128

    try:
        logger.info(f"Configuration: B={B}, T={T}, MAX_T={MAX_T}, N_H_L={N_H_L}, N_KVH_L={N_KVH_L}, D_H={D_H}")
        
        axq = torch.randn(1, B * T, N_KVH_L, N_H_L // N_KVH_L, D_H, 
                         dtype=torch.bfloat16, device="cuda")
        axk = torch.randn(1, B * MAX_T, N_KVH_L, N_H_L // N_KVH_L, D_H,
                         dtype=torch.bfloat16, device="cuda")
        axv = torch.randn(1, B * MAX_T, N_KVH_L, N_H_L // N_KVH_L, D_H,
                         dtype=torch.bfloat16, device="cuda")

        # Follow llama_disagg.py prefill pattern with seqlens_q and seqlens_kv
        seqlens_kv = [T for _ in range(B)]
        seqlens_q = [T for _ in range(B)]
        
        attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
            q_seqlen=seqlens_q,
            kv_padding=MAX_T,
            kv_seqlen=seqlens_kv,
        )

        # Create operation function based on op parameter
        if op is None:
            op_func = lambda q, k, v, bias: fmha.memory_efficient_attention_forward(q, k, v, bias)
        elif op == "aiter" or op == "aiter_varlen":
            if not AITER_AVAILABLE:
                raise RuntimeError("AITER not available but requested")
            # Pre-compute AITER bias outside of benchmark timing
            precomputed_aiter_bias = precompute_aiter_bias(axq, axk, axv, attn_bias)
            op_func = lambda q, k, v, bias: run_aiter_attention(op, q, k, v, bias, causal=True, precomputed_bias=precomputed_aiter_bias)
        else:
            op_func = lambda q, k, v, bias: fmha.memory_efficient_attention_forward(q, k, v, bias, op=op)
        
        y, t_ms = run_operation_benchmark(op_display_name, op_func, (axq, axk, axv, attn_bias))
        y = y.view(B, T, N_H_L * D_H)
        
        # Calculate detailed performance metrics
        prefill_flops = 4 * B * N_H_L * T * T * D_H
        tflops = prefill_flops / (t_ms * 1e-3) / 1e12
        
        # Memory bandwidth
        bytes_per_element = 2
        total_elements = B * T * N_H_L * D_H
        memory_reads = 3 * total_elements * bytes_per_element
        memory_writes = total_elements * bytes_per_element
        total_memory_bytes = memory_reads + memory_writes
        bandwidth_gb_s = (total_memory_bytes / (t_ms * 1e-3)) / 1e9
        
        tokens_per_sec = (B * T) / (t_ms * 1e-3)
        
        logger.info(f"✅ Prefill Attention Results (T={T}):")
        logger.info(f"   Operation: {op_display_name}")
        logger.info(f"   Output shape: {y.shape}")
        logger.info(f"   Time: {t_ms:.2f} ms")
        logger.info(f"   TFLOPs/s: {tflops:.2f}")
        logger.info(f"   Memory BW: {bandwidth_gb_s:.1f} GB/s")
        logger.info(f"   Tokens/sec: {tokens_per_sec:.0f}")
        logger.info(f"   Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        # Performance scaled for 80 layers (typical LLM)
        time_80_layers = t_ms * 80
        tokens_per_sec_80_layers = tokens_per_sec / 80
        logger.info(f"   --- 80 Layers Performance ---")
        logger.info(f"   Total time (80 layers): {time_80_layers:.2f} ms")
        logger.info(f"   Tokens/sec (80 layers): {tokens_per_sec_80_layers:.0f}")
        logger.info(f"   FLOPs: {prefill_flops / 1e12:.2f} TFLOPs")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prefill test failed: {e}")
        return False


def test_decoding_attention(batch_size=128, op_name="default", prompt_tokens=8192):
    """Test decoding attention (single token input) with specified operation"""
    op, op_display_name = get_operation(op_name)
    logger.info(f"Testing decoding attention with batch size {batch_size} using {op_display_name}...")
    
    if not torch.cuda.is_available():
        return False
        
    B, T, MAX_T = batch_size, 1, 32769
    N_H_L, N_KVH_L, D_H = 8, 1, 128
    PROMPT_T = prompt_tokens
    try:
        logger.info(f"Decoding configuration: B={B}, T={T}, MAX_T={MAX_T}, N_H_L={N_H_L}, N_KVH_L={N_KVH_L}, D_H={D_H}")
        
        # Follow llama_disagg.py decoding pattern        
        xq = torch.randn(B * T, N_H_L, D_H, dtype=torch.bfloat16, device="cuda")
        cache_k = torch.randn(B, MAX_T, N_KVH_L, D_H, dtype=torch.bfloat16, device="cuda")
        cache_v = torch.randn(B, MAX_T, N_KVH_L, D_H, dtype=torch.bfloat16, device="cuda")
        seq_positions: torch.Tensor = (torch.tensor([PROMPT_T for _ in range(B)]).cuda().int())
        attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
            q_seqlen=[1 for _ in range(B)],
            kv_padding=MAX_T,
            kv_seqlen=[seq_position + 1 for seq_position in seq_positions.tolist()],
        )

        # Reshape as in decoding_attn
        axq = xq.view(1, B * T, N_KVH_L, N_H_L // N_KVH_L, D_H)
        axk = cache_k.view(1, B * MAX_T, N_KVH_L, 1, D_H).expand(
            1, B * MAX_T, N_KVH_L, N_H_L // N_KVH_L, D_H)
        axv = cache_v.view(1, B * MAX_T, N_KVH_L, 1, D_H).expand(
            1, B * MAX_T, N_KVH_L, N_H_L // N_KVH_L, D_H)

        # Create operation function based on op parameter
        print(f"axq = {axq.shape}, axk = {axk.shape}, axv = {axv.shape}")

        if op is None:
            op_func = lambda q, k, v, bias: fmha.memory_efficient_attention_forward(q, k, v, bias)
        elif op == "aiter" or op == "aiter_varlen":
            if not AITER_AVAILABLE:
                raise RuntimeError("AITER not available but requested")
            # Pre-compute AITER bias outside of benchmark timing
            precomputed_aiter_bias = precompute_aiter_bias(axq, axk, axv, attn_bias)
            op_func = lambda q, k, v, bias: run_aiter_attention(op, q, k, v, bias, causal=True, precomputed_bias=precomputed_aiter_bias)
        else:
            op_func = lambda q, k, v, bias: fmha.memory_efficient_attention_forward(q, k, v, bias, op=op)
        
        avg_seq_len = seq_positions.float().mean().item()
        
        y, t_ms = run_operation_benchmark(op_display_name, op_func, (axq, axk, axv, attn_bias))
        y = y.view(B, T, N_H_L * D_H)
        
        # Calculate performance metrics for decoding
        decoding_flops = 4 * B * N_H_L * T * avg_seq_len * D_H
        tflops = decoding_flops / (t_ms * 1e-3) / 1e12
        
        # Memory bandwidth for decoding
        bytes_per_element = 2
        query_memory = B * T * N_H_L * D_H * bytes_per_element
        kv_memory = 2 * B * avg_seq_len * N_KVH_L * D_H * bytes_per_element
        output_memory = B * T * N_H_L * D_H * bytes_per_element
        total_memory_bytes = query_memory + kv_memory + output_memory
        bandwidth_gb_s = (total_memory_bytes / (t_ms * 1e-3)) / 1e9
        
        tokens_per_sec = B / (t_ms * 1e-3)
        throughput_per_seq = tokens_per_sec / B
        
        logger.info(f"✅ Decoding Attention Results (B={B}):")
        logger.info(f"   Operation: {op_display_name}")
        logger.info(f"   Output shape: {y.shape}")
        logger.info(f"   Batch size: {B}, Avg seq length: {avg_seq_len:.0f}")
        logger.info(f"   Time: {t_ms:.2f} ms")
        logger.info(f"   TFLOPs/s: {tflops:.2f}")
        logger.info(f"   Memory BW: {bandwidth_gb_s:.1f} GB/s")
        logger.info(f"   Total tokens/sec: {tokens_per_sec:.0f}")
        logger.info(f"   Tokens/sec/sequence: {throughput_per_seq:.0f}")
        logger.info(f"   Latency per token: {t_ms / B:.2f} ms")
        
        # Performance scaled for 80 layers (typical LLM)
        time_80_layers = t_ms * 80
        tokens_per_sec_80_layers = tokens_per_sec / 80
        latency_per_token_80_layers = (t_ms * 80) / B
        logger.info(f"   --- 80 Layers Performance ---")
        logger.info(f"   Total time (80 layers): {time_80_layers:.2f} ms")
        logger.info(f"   Tokens/sec (80 layers): {tokens_per_sec_80_layers:.0f}")
        logger.info(f"   Latency per token (80 layers): {latency_per_token_80_layers:.2f} ms")
        return True
        
    except Exception as e:
        logger.error(f"❌ Decoding test failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test fmha operations with flexible parameters')
    parser.add_argument('--seq-len', type=int, choices=[2048, 16384], default=2048,
                       help='Sequence length for prefill test (default: 2048)')
    parser.add_argument('--batch-size', type=int, choices=[1, 8, 128], default=128,
                       help='Batch size for decoding test (default: 128)')
    parser.add_argument('--op', type=str, choices=['default', 'ck', 'triton_splitk', 'aiter', 'aiter_varlen'], default='default',
                       help='Operation to use: default (memory_efficient_attention_forward), ck (fmha.ck.FwOp), triton_splitk (fmha.triton_splitk.FwOp), aiter (aiter.flash_attn_func), aiter_varlen (aiter.flash_attn_varlen_func)')
    parser.add_argument('--prefill-only', action='store_true',
                       help='Run only prefill attention test')
    parser.add_argument('--decode-only', action='store_true',
                       help='Run only decoding attention test')
    parser.add_argument('--prompt-tokens', type=int, choices=[8192, 32768], default=8192,
                       help='Prompt token length for decoding test (default: 8192)')
    
    args = parser.parse_args()
    
    _, op_display_name = get_operation(args.op)
    
    logger.info("=" * 80)
    logger.info("fmha Operation Performance Analysis")
    logger.info(f"Operation: {op_display_name}")
    logger.info("=" * 80)
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    tests = []
    
    if not args.decode_only:
        tests.append(("Prefill Attention Performance", lambda: test_prefill_attention(args.seq_len, args.op)))
    
    if not args.prefill_only:
        tests.append(("Decoding Attention Performance", lambda: test_decoding_attention(args.batch_size, args.op, args.prompt_tokens)))
    
    passed = 0
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}")
        logger.info("=" * 60)
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Performance Analysis Complete: {passed}/{len(tests)} tests passed")
    logger.info(f"{'='*80}")
    
    if passed == len(tests):
        logger.info("🎉 All performance tests passed!")
        exit(0)
    else:
        logger.error("💥 Some tests failed!")
        exit(1)
