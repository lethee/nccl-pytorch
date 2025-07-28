FROM nvcr.io/nvidia/pytorch:23.07-py3

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY nccl_perf_test.py /nccl_perf_test.py

ENTRYPOINT ["/entrypoint.sh"]
