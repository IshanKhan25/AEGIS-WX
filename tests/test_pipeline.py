from inference.pipeline import AegisPipeline
from verification.benchmarks import benchmark
def test_full_demo_pipeline():
    r=AegisPipeline().run(seed=26082)
    assert "Synthetic Hindcast" in r["data_mode"]
    assert len(benchmark(r)) == 5
