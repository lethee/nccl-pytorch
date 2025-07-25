import torch
import torch.distributed as dist
import time
import os

def run_benchmark(rank, world_size, local_rank):
    """지정된 텐서 크기에 대해 all_reduce 벤치마크를 실행합니다."""
    
    # 1. 분산 환경 초기화
    if rank == 0:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"NCCL available: {torch.distributed.is_nccl_available()}")

    print(f"dist.init_process_group(nccl, rank={rank}, world_size={world_size}) (local_rank={local_rank})")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # 2. 테스트할 텐서 크기 정의 (단위: Bytes)
    tensor_sizes = [
             256 * 1024 * 1024,  # 256MB
            1024 * 1024 * 1024,  # 1GB
        # 4 * 1024 * 1024 * 1024,  # 4GB
    ]
    
    # 헤더 출력 (rank 0에서만)
    if rank == 0:
        print(f"{'Tensor Size (MB)':<20} {'Avg Bandwidth (GB/s)':<25}")
        print("-" * 45)

    for size_bytes in tensor_sizes:
        # float32 텐서이므로 4바이트로 나눔
        num_elements = size_bytes // 4
        tensor = torch.randn(num_elements, device=local_rank)
        
        # 워밍업: 실제 측정 전 GPU 커널을 예열합니다.
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # 3. 실제 성능 측정
        iterations = 20
        start_time = time.time()
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize() # 모든 GPU 연산이 끝날 때까지 대기
        end_time = time.time()

        # 4. 결과 계산 및 출력 (rank 0에서만)
        if rank == 0:
            total_time = end_time - start_time
            avg_time_per_iter = total_time / iterations
            
            # all_reduce에서 2개 GPU의 유효 데이터 전송량은 텐서 크기와 거의 동일합니다.
            # Bandwidth (GB/s) = (Tensor Size in GB) / time
            bandwidth_gb_s = (size_bytes / 1e9) / avg_time_per_iter
            
            size_mb = size_bytes / (1024 * 1024)
            print(f"{size_mb:<20.0f} {bandwidth_gb_s:<25.2f}")

    # 5. 분산 환경 정리
    dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun이 자동으로 RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT 등을 설정합니다.
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    run_benchmark(rank, world_size, local_rank)