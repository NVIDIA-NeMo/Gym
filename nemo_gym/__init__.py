from os import environ
from os.path import abspath, dirname, join

import sys


ROOT_DIR = dirname(abspath(__file__))
PARENT_DIR = abspath(join(ROOT_DIR, ".."))
CACHE_DIR = join(PARENT_DIR, "cache")
RESULTS_DIR = join(PARENT_DIR, "results")

sys.path.append(PARENT_DIR)

# TODO: Maybe eventually we want an override for OMP_NUM_THREADS ?

# Turn off HF tokenizers paralellism
environ["TOKENIZERS_PARALLELISM"] = "false"

# Huggingface related caching directory overrides to local folders.
environ["HF_DATASETS_CACHE"] = join(CACHE_DIR, "huggingface")
environ["TRANSFORMERS_CACHE"] = environ["HF_DATASETS_CACHE"]
# TODO When `TRANSFORMERS_CACHE` is no longer supported in transformers>=5.0.0, migrate to `HF_HOME`
# environ["HF_HOME"] = join(CACHE_DIR, "huggingface")

# UV caching directory overrides to local folders.
environ["UV_CACHE_DIR"] = join(CACHE_DIR, "uv")

# Turn off Gradio analytics
environ["GRADIO_ANALYTICS_ENABLED"] = "False"
