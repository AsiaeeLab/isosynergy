"""Metadata-defined prospective panels: anchor plus ten and five-drug panels."""
import paths
import itertools
import numpy as np
import pandas as pd
SEED = 20260922

def main():
    df = pd.read_parquet(paths.INPUTS / 'drugcomb_matrices.parquet').drop_duplicates('experiment_id')
    df = df.rename(columns={'experiment_id': 'matrix_id', 'drugA': 'drug_a_name', 'drugB': 'drug_b_name', 'study_source': 'source_study'})
    df = df[df.cell_line.notna()].copy()
    pairs = [tuple(sorted([str(a), str(b)])) for (a, b) in zip(df.drug_a_name, df.drug_b_name)]
    df['a'] = [p[0] for p in pairs]
    df['b'] = [p[1] for p in pairs]
    df = df[df.a != df.b]
    counts = df.groupby(['source_study', 'cell_line', 'a', 'b']).size().rename('records').reset_index()
    records = df.sort_values('matrix_id').drop_duplicates(['source_study', 'cell_line', 'a', 'b'])
    rng = np.random.default_rng(SEED)
    out = []
    memberships = []
    for ((source, cell), g) in records.groupby(['source_study', 'cell_line'], sort=True):
        lookup = {(r.a, r.b): r for r in g.itertuples()}
        drugs = sorted(set(g.a) | set(g.b))
        neighbors = {d: sorted(set(g.loc[g.a == d, 'b']) | set(g.loc[g.b == d, 'a'])) for d in drugs}
        plans = []
        for anchor in drugs:
            if len(neighbors[anchor]) >= 10:
                for repeat in range(3):
                    chosen = sorted(rng.choice(neighbors[anchor], 10, replace=False).tolist())
                    plans.append(('anchor_plus_10', anchor, repeat, [tuple(sorted((anchor, d))) for d in chosen]))
        complete = 0
        for attempt in range(200):
            if len(drugs) < 5 or complete >= 30:
                break
            chosen = sorted(rng.choice(drugs, 5, replace=False).tolist())
            pp = list(itertools.combinations(chosen, 2))
            if all((p in lookup for p in pp)):
                plans.append(('five_drug_panel', '|'.join(chosen), complete, pp))
                complete += 1
        for (kind, anchor, repeat, pp) in plans:
            rr = [lookup[p] for p in pp]
            pid = f'{source}:{cell}:{kind}:{anchor}:{repeat}'
            for r in rr:
                memberships.append(dict(panel_id=pid, matrix_id=r.matrix_id))
    pd.DataFrame(memberships).to_csv(paths.INPUTS / 'panel_membership.csv', index=False)
if __name__ == '__main__':
    main()
