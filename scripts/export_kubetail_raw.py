#!/usr/bin/env python3
"""Export retained TFA02 Kubetail rows for 20 and 21 October 2026, without KPI parsing."""
import ast
import base64
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/kubetail_raw_2026-10-20_21' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
TZ = ZoneInfo('Europe/Berlin')
START = datetime(2026, 10, 20, tzinfo=TZ)
END = datetime(2026, 10, 22, tzinfo=TZ)
# Include pod names from the user's console URL as well as current discovery.
KNOWN_PODS = ['tfa02-ff95f9555-82wz6', 'tfa02-58b8b96d96-568lf', 'tfa02-567db959bc-9qq6q']


def stamp(text):
    normalized = re.sub(r'(\.\d{6})\d+', r'\1', text).replace('Z', '+00:00')
    result = datetime.fromisoformat(normalized)
    if result.tzinfo is None:
        raise ValueError('Backend timestamp has no timezone: ' + text)
    return result


def main():
    notebook = json.loads((ROOT / 'tfa02_log_kpis.ipynb').read_text())
    config = {}
    for node in ast.parse(''.join(notebook['cells'][1]['source'])).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in ('GRAPHQL_URL', 'LOG_USERNAME', 'LOG_PASSWORD', 'NAMESPACE', 'LABEL_SELECTOR', 'CONTAINER_NAME'):
                    config[name.id] = ast.literal_eval(node.value)
    def request(query, variables):
        credentials = (config['LOG_USERNAME'] + ':' + config['LOG_PASSWORD']).encode()
        req = urllib.request.Request(config['GRAPHQL_URL'],
            data=json.dumps(dict(query=query, variables=variables)).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + base64.b64encode(credentials).decode()})
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.load(response)
        if result.get('errors'):
            raise RuntimeError(json.dumps(result['errors']))
        return result['data']
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(source=config['GRAPHQL_URL'], namespace=config['NAMESPACE'],
        container=config['CONTAINER_NAME'], timezone=str(TZ), start_inclusive=START.isoformat(),
        end_exclusive=END.isoformat(), pods=[], errors=[],
        note='Only logs retained and exposed by Kubetail are recoverable. Exhausted retained history does not prove full-day coverage. Original timestamps/messages are preserved; no timing extraction or deduplication is applied.')
    pod_names = set(KNOWN_PODS)
    cursor = ''
    try:
        while True:
            data = request('''query Pods($namespace:String!, $labelSelector:String!, $continue:String) {
              coreV1PodsList(namespace:$namespace, options:{labelSelector:$labelSelector, continue:$continue}) {
                metadata { continue } items { metadata { name creationTimestamp } }
              }
            }''', dict(namespace=config['NAMESPACE'],labelSelector=config['LABEL_SELECTOR'], **{'continue':cursor}))['coreV1PodsList']
            pod_names.update(p['metadata']['name'] for p in data['items'])
            new_cursor = data['metadata'].get('continue') or ''
            if not new_cursor: break
            if new_cursor == cursor: raise RuntimeError('Repeated pod-discovery cursor')
            cursor = new_cursor
    except Exception as exc:
        manifest['errors'].append(dict(stage='pod_discovery', error=str(exc)))
        print('Pod discovery failed; trying known pods:', exc, flush=True)
    print('Output directory:', OUT, flush=True)
    all_exported = []
    for pod in sorted(pod_names):
        before = None; seen = set(); pages = []; first = None; last = None
        info = dict(pod=pod, pages=0, scanned_rows=0, exported_rows=0)
        manifest['pods'].append(info)
        try:
            with (OUT / (pod + '_api_pages.jsonl')).open('w') as audit:
                # No sample-count or page-count cutoff: exhaust retained history or cross the lower date boundary.
                while True:
                    result = request('''query Tail($namespace:String!, $name:String!, $container:String, $before:ID, $last:Int) {
                      podLogTail(namespace:$namespace,name:$name,container:$container,before:$before,last:$last) {
                        results { timestamp message }
                        pageInfo { hasPreviousPage startCursor }
                      }
                    }''',dict(namespace=config['NAMESPACE'],name=pod,container=config['CONTAINER_NAME'],before=before,last=1000))['podLogTail']
                    audit.write(json.dumps(result)+'\n'); audit.flush()
                    rows = result['results']; pages.append(rows)
                    info['pages'] += 1; info['scanned_rows'] += len(rows)
                    times = [stamp(r['timestamp']) for r in rows]
                    if times:
                        first = min([first] + times) if first else min(times)
                        last = max([last] + times) if last else max(times)
                    print(pod, 'page', info['pages'], 'rows', len(rows), 'scanned', info['scanned_rows'], flush=True)
                    if times and min(times) < START:
                        info['stop_reason'] = 'crossed_start_boundary'; break
                    page_info = result['pageInfo']; cursor = page_info.get('startCursor')
                    if not page_info.get('hasPreviousPage'):
                        info['stop_reason'] = 'retained_history_exhausted'; break
                    if not cursor or cursor in seen:
                        raise RuntimeError('Missing or repeated pagination cursor before history exhausted')
                    seen.add(cursor); before = cursor
        except Exception as exc:
            info['stop_reason'] = 'error'
            info['error'] = str(exc)
            print('Could not finish pod',pod,':',exc,flush=True)
        for page in reversed(pages):
            for row in page:
                if START <= stamp(row['timestamp']) < END:
                    all_exported.append(dict(pod=pod, **row)); info['exported_rows'] += 1
        info.update(earliest_scanned=first.isoformat() if first else None, latest_scanned=last.isoformat() if last else None)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    all_exported.sort(key=lambda r: stamp(r['timestamp']))
    counts = {}
    for day in ('2026-10-20','2026-10-21'):
        rows = [r for r in all_exported if stamp(r['timestamp']).astimezone(TZ).date().isoformat()==day]
        counts[day] = len(rows)
        with (OUT/(day+'.jsonl')).open('w') as js, (OUT/(day+'.log')).open('w') as text, (OUT/(day+'.csv')).open('w',newline='') as cs:
            writer=csv.DictWriter(cs,fieldnames=['pod','timestamp','message']);writer.writeheader()
            for row in rows:
                js.write(json.dumps(row)+'\n');text.write(row['timestamp']+' ['+row['pod']+'] '+row['message']+'\n');writer.writerow(row)
    with (OUT/'all_rows.jsonl').open('w') as handle:
        for row in all_exported: handle.write(json.dumps(row)+'\n')
    manifest.update(rows_per_day=counts,total_exported_rows=len(all_exported),finished_at_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Export completed:', json.dumps(counts), '\nSaved in:', OUT,flush=True)


if __name__ == '__main__':
    main()
