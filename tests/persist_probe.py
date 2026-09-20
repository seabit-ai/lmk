"""One process = one life of the server. Loads the model with a persistent
cache directory, prefills one fixed prompt, prints what the cache gave it."""
import json
import sys
from pathlib import Path

from lmk.engine import MlxEngine

model_dir, cache_dir = Path(sys.argv[1]), Path(sys.argv[2])
engine = MlxEngine("probe", model_dir, 32768, cache_dir=cache_dir)
messages = [{"role": "system", "content": "".join(f"Persist rule {i}: one short sentence only. " for i in range(250))},
            {"role": "user", "content": "Say hello."}]
prompt = engine.chat_format().render(messages, None)
gen = engine.generate(prompt, max_tokens=24, request_id="persist-probe", on_prefill=lambda *_: True,
                      sampling={"temp": 0.0})
text = "".join(gen)
engine.close()
print("PROBE " + json.dumps({"prompt_tokens": gen.stats.prompt_tokens, "cached_tokens": gen.stats.cached_tokens,
                             "text": text}))
