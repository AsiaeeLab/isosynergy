"""Generate shared synthetic arrays and R seeds for the BIGL comparison."""
import hashlib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bigl_design as C
from bigl_design import X

def main():
    os.makedirs(C.DATA, exist_ok=True)
    C.write_meta()
    lines = []
    for (r, surf, noise) in C.all_settings():
        st = C.setting_name(r, surf, noise)
        for cond in C.CONDS:
            n = C.n_for(cond)
            Y = C.generate(r, surf, noise, cond, n)
            p = C.data_path(r, surf, noise, cond)
            Y.astype('<f8').tofile(p)
            seeds = np.array([X.seed_for(bytes([114, 55, 98]).decode('ascii'), 'bigl', st, cond, i) % 2147483647 for i in range(n)], dtype='<i4')
            seeds.tofile(p.replace('.f64', '.seeds.i32'))
            h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            lines.append(f'{h}  {os.path.basename(p)}')
            print(st, cond, Y.shape, h[:12], flush=True)
    with open(os.path.join(C.DATA, 'sha256.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
if __name__ == '__main__':
    main()
