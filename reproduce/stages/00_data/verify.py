"""Verify the public DrugCombDB and NCI-ALMANAC inputs by SHA-256."""
import hashlib
import json
import paths

HASHES = {
    'drugcombdb/drugcombs_response.csv': '3f67aab5efc4df3ae60fc95122b6daf5016f6fde40abc1be115a9683ad69bf73',
    'drugcombdb/drugcombs_scored.csv': '094e01cd3a468de345bc2637389a027f26fc4dceada9ce7ac9bfe00e8fa9d433',
    'nci_almanac/ComboDrugGrowth_Nov2017.csv': 'b6e1309b189026c76db3564bab9c49595e099c1aaab01478aa3888df1407a24d',
    'nci_almanac/NCI60_CELLNAME_to_Combo.txt': 'd7eebb91748999f2a99c0214636d41f00d138856c28b52498f29d97b3da9d44a',
    'nci_almanac/NCI_IOA_AOA_drugs.tsv': '66d776c93b29123131a88bf3eedf165b02d2bd104471ebec17366855b6c4eacd',
}

def main():
    result = {}
    for name, expected in HASHES.items():
        path = paths.raw_file(name)
        if not path.is_file():
            raise SystemExit(f'Place the public input at data/raw/{name}; see reproduce/data/README.md.')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                h.update(chunk)
        observed = h.hexdigest()
        if observed != expected:
            raise SystemExit(f'{name}: SHA-256 mismatch: {observed}; expected {expected}')
        result[name] = observed
        print(f'{name}: SHA-256 matches', flush=True)
    (paths.OUT / 'raw_checksums.json').write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__':
    main()
