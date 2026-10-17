# Design review

What the original code did against common software-design principles, and what changed. "Open" means not fixed.

| # | Problem found | Principle | Fix |
|---|---|---|---|
| 1 | The same Config, frame extraction, soft-thresholding, SVT, IALM solver and metrics were copy-pasted across `video_decomposition_pipeline.ipynb`, `v1_no_gt/` and `v2_with_gt/` | **DRY**; logic buried in notebooks, so it cannot be tested or reused | Core routines extracted to `rpca_demo.py` and covered by `tests/test_rpca.py`. **Open:** the notebooks still contain their own copies |
| 2 | Absolute paths such as `/home/isha/Desktop/AI_ML_2026/...` and `path/...` placeholders inside the notebooks | Configuration coupled to one machine (**externalise configuration**, dependency on the environment) | Notebooks locate the project root from the repo (`rpca_demo.py` marker); all ten notebooks execute from a clean checkout |
| 3 | A duplicated `@dataclass(frozen=True)` decorator in the root notebooks made the Config cell fail on Python 3.10 | Correctness bug | Removed |
| 4 | README/notebook text promised randomized SVD and Center Location Error, but no code computed them | **YAGNI / speculative generality** and misleading documentation | Both implemented in `rpca_demo.py` (`randomized_svd`, `center_location_error`), tested, exposed in the demo; notebook text now points to them |
| 5 | The demo's decomposition choices were fixed in code (exact SVD only, one threshold) | **Open/Closed** (extension required editing the solver) | `svd=` option on `rpca_ialm` and a threshold slider. **Open:** this is a string flag; a strategy object would be the stricter fix |
| 6 | No automated tests | Testability | 8 pytest tests (operators, both SVD engines, metrics, video reading) |
| 7 | Cloud demo was unusably slow on a throttled CPU (BLAS thread oversubscription, heavy defaults) | Resource assumptions baked into defaults | Single BLAS thread and lighter defaults in the app |
