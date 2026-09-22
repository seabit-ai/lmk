#!/bin/sh
set -e
cd "$(dirname "$0")"; mkdir -p raw
U=$1; M=$2
Q='Write the word BANANA in capitals, then the word APPLE. Nothing else.'
show() { python3 -c "
import json,sys; d=json.load(open('raw/$1.json')); m=d['choices'][0]['message']
r=m.get('reasoning') or m.get('reasoning_content') or ''
print('$1', '|', d['choices'][0]['finish_reason'], '| content:', repr(m.get('content')), '| reasoning has BANANA:', 'BANANA' in r, '| tokens', d['usage']['completion_tokens'])"; }
curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"stop\":[\"BANANA\"],\"max_tokens\":600}" > raw/stop-banana.json; show stop-banana
curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"stop\":[\"APPLE\"],\"max_tokens\":600}" > raw/stop-apple.json; show stop-apple
curl -s -m 300 -N "$U/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"stop\":[\"APPLE\"],\"max_tokens\":600,\"stream\":true}" > raw/stop-apple-stream.txt
python3 -c "
import json
chunks=[json.loads(l[6:]) for l in open('raw/stop-apple-stream.txt') if l.startswith('data: ') and l.strip()!='data: [DONE]']
c=''.join(x['choices'][0]['delta'].get('content','') for x in chunks if x['choices'])
f=[x['choices'][0]['finish_reason'] for x in chunks if x['choices'] and x['choices'][0]['finish_reason']]
print('stop-apple-stream | content:', repr(c), '| finish', f)"
