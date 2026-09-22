#!/bin/sh
# usage: ./run.sh <base_url> <model>   — writes raw/*.json
set -e
cd "$(dirname "$0")"; mkdir -p raw
U=$1; M=$2
ask() { # name body
  curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$2" > "raw/$1.json"
  python3 -c "
import json,sys; d=json.load(open('raw/$1.json')); m=d['choices'][0]['message']
print('$1', '|', d['choices'][0]['finish_reason'], '| content:', repr(m.get('content'))[:80], '| reasoning:', repr(m.get('reasoning') or m.get('reasoning_content'))[-100:], '| tokens', d['usage']['completion_tokens'])"
}
Q='Write the word BANANA in capitals, then the word APPLE. Nothing else.'
ask stop-in-thinking "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"stop\":[\"BANANA\"],\"max_tokens\":400}"
ask no-stop         "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":400}"
P='In one sentence, describe a colour to someone who has never seen it.'
ask temp0-a "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"temperature\":0,\"max_tokens\":300}"
ask temp0-b "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"temperature\":0,\"max_tokens\":300}"
ask default-a "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":300}"
ask default-b "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":300}"
