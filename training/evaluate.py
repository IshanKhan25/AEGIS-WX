from verification.benchmarks import benchmark
from inference.pipeline import AegisPipeline
if __name__ == "__main__": print(benchmark(AegisPipeline().run()).to_string(index=False))
