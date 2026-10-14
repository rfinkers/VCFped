"""
Tests for vcfped/testfiles/generate_breeding_sim.py -- the plant-breeding
validation-cohort simulator.

Two layers:
  * building blocks (meiosis, pedigree kinship, relationship classes, truth IBD)
    on tiny hand-built pedigrees whose correct answers are known analytically;
  * one full default run (module-scoped fixture, ~6 s) whose outputs are checked
    for the properties the validation set promises: size, anonymity, error-free
    Mendelian-consistent truth, duplicates, planted quality problems, edge-case
    coverage, and that every truth file parses with the project's own loaders.
"""
import collections
import csv
import gzip
import os

import pytest

from conftest import (import_module_from_path, run_script, requires_numpy,
                      TESTFILES_DIR, UTILITIES_DIR)

pytestmark = requires_numpy

GENERATOR = os.path.join(TESTFILES_DIR, 'generate_breeding_sim.py')


@pytest.fixture(scope='module')
def gen():
    return import_module_from_path('generate_breeding_sim', GENERATOR)


def tiny_sim(gen, seed=1, n_chr=3, snps=30):
    import numpy as np
    rp, rg = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    return gen.Simulation(gen.Genome(n_chr, snps, rg), 5, 0.04, rp, rg)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def test_crossover_rate_matches_map_length_and_interference(gen):
    """~1 crossover per Morgan; nu=1 is Poisson, nu>1 is under-dispersed."""
    import numpy as np
    rng = np.random.default_rng(0)
    for nu, var_ratio in ((1, (0.85, 1.15)), (5, (0.0, 0.8))):
        n = np.array([len(gen.crossover_positions(100.0, nu, rng)) for _ in range(4000)])
        assert n.mean() == pytest.approx(1.0, abs=0.06)
        assert var_ratio[0] <= n.var() / n.mean() <= var_ratio[1]


def test_gamete_is_mosaic_of_parental_haplotypes(gen):
    """Every allele/label of a child haplotype comes from one of the parent's two haplotypes."""
    import numpy as np
    sim = tiny_sim(gen)
    a, b = sim.add_founder('t'), sim.add_founder('t')
    c = sim.add_child(a, b, 't')
    A, B, C = sim.inds[a], sim.inds[b], sim.inds[c]
    for k, parent in ((0, A), (1, B)):
        from_lab = (C.lab[k] == parent.lab[0]) | (C.lab[k] == parent.lab[1])
        assert from_lab.all()
        assert ((C.hap[k] == parent.hap[0]) | (C.hap[k] == parent.hap[1])).all()
        assert len(set(C.lab[k])) <= 2                      # a mosaic of at most 2 haplotypes


def test_gametes_recombine_at_the_map_rate(gen):
    """Label switches along real gametes average ~ map length / 100 per chromosome."""
    import numpy as np
    sim = tiny_sim(gen, n_chr=4, snps=60)
    parent = sim.add_founder('t')
    mate = sim.add_founder('t')
    switches = np.zeros(sim.g.n_chr)
    n_kids = 300
    for _ in range(n_kids):
        lab = sim.inds[sim.add_child(parent, mate, 't')].lab[0]       # paternal gamete
        for c, sl in enumerate(sim.g.slices):
            switches[c] += (np.diff(lab[sl]) != 0).sum()
    expected = sim.g.len_cM / 100.0
    assert switches.sum() / n_kids == pytest.approx(expected.sum(), rel=0.15)


def test_same_seed_same_haplotypes_different_seed_differs(gen):
    import numpy as np
    def build(seed):
        sim = tiny_sim(gen, seed)
        a, b = sim.add_founder('t'), sim.add_founder('t')
        sim.add_child(a, b, 't')
        return list(sim.inds.values())[-1].hap
    assert np.array_equal(build(5), build(5))
    assert not np.array_equal(build(5), build(6))


def test_pedigree_kinship_and_inbreeding_known_values(gen):
    sim = tiny_sim(gen)
    p1, p2 = sim.add_founder('t'), sim.add_founder('t')
    f1 = sim.add_child(p1, p2, 't')
    bc1 = sim.add_child(f1, p1, 't')            # backcross to a parent
    s1, s2 = sim.add_child(p1, p2, 't'), sim.add_child(p1, p2, 't')
    sibmated = sim.add_child(s1, s2, 't')       # full-sib mating
    selfed = sim.add_child(p1, p1, 't')
    idx, A = gen.numerator_relationship(sim)
    phi = lambda x, y: A[idx[x], idx[y]] / 2
    assert phi(f1, p1) == 0.25
    assert phi(s1, s2) == 0.25
    assert phi(bc1, p1) == pytest.approx(0.375)         # PO and grandchild
    assert A[idx[sibmated], idx[sibmated]] - 1 == 0.25  # F of full-sib offspring
    assert A[idx[selfed], idx[selfed]] - 1 == 0.5       # F of a selfed child
    assert phi(selfed, p1) == 0.5


def test_relationship_classes(gen):
    sim = tiny_sim(gen)
    gp1, gp2, mate, other, other2 = (sim.add_founder('t') for _ in range(5))
    p = sim.add_child(gp1, gp2, 't')
    unc = sim.add_child(gp1, gp2, 't')                  # full sib of p
    hs = sim.add_child(gp1, other, 't')                 # half sib of p
    child = sim.add_child(p, mate, 't')
    nephew = sim.add_child(unc, other, 't')             # p's nephew
    cousin = sim.add_child(unc, other2, 't')            # child's first cousin
    clone = sim.add_clone(child)
    cls = lambda x, y: gen.classify_pair(sim, x, y)
    assert cls(p, child) == 'parent_offspring'
    assert cls(p, unc) == 'full_sib'
    assert cls(p, hs) == 'half_sib'
    assert cls(gp1, child) == 'grandparent'
    assert cls(unc, child) == 'avuncular'
    assert cls(p, nephew) == 'avuncular'
    assert cls(child, cousin) == 'first_cousin'
    assert cls(child, clone) == 'duplicate'
    assert cls(child, sim.add_founder('t')) == 'other'


def test_truth_ibd_of_parent_offspring_covers_whole_genome(gen):
    """A child's two haplotypes jointly match the parent over every SNP; none of the other parent's."""
    import numpy as np
    sim = tiny_sim(gen)
    a, b = sim.add_founder('t'), sim.add_founder('t')
    c = sim.add_child(a, b, 't')
    g = sim.g
    segs = gen.ibd_segments(g, sim.inds[a].lab, sim.inds[c].lab, min_cM=0)
    assert segs and all(s[1] == 1 for s in segs)        # only the paternal child haplotype matches a
    covered = np.zeros(g.M, bool)
    for _, _, chrom, s, e, _ in segs:
        covered |= (g.pos_bp >= s) & (g.pos_bp <= e) & (np.array(g.chr_names)[g.chrom_of] == chrom)
    assert covered.all()
    assert gen.ibd_segments(g, sim.inds[a].lab, sim.inds[b].lab, min_cM=0) == []   # founders: no IBD


def test_selfed_child_hbd_nonzero_outbred_child_zero(gen):
    sim = tiny_sim(gen)
    a, b = sim.add_founder('t'), sim.add_founder('t')
    outbred = sim.add_child(a, b, 't')
    selfed = sim.add_child(a, a, 't')
    for _ in range(5):
        selfed = sim.add_child(selfed, selfed, 't')
    g = sim.g
    assert gen.ibd_segments(g, sim.inds[outbred].lab, sim.inds[outbred].lab, 0, True) == []
    assert gen.ibd_segments(g, sim.inds[selfed].lab, sim.inds[selfed].lab, 0, True)


# ---------------------------------------------------------------------------
# Full default run
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def sim_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('breeding_sim'))
    result = run_script(GENERATOR, ['--out-dir', out], timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def _path(d, suffix):
    return os.path.join(d, 'breeding_sim' + suffix)


def _load_vcf(path):
    import numpy as np
    opener = gzip.open if path.endswith('.gz') else open
    samples, rows, fixed = None, [], []
    with opener(path, 'rt') as f:
        for line in f:
            if line.startswith('##'):
                continue
            parts = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                samples = parts[9:]
            else:
                fixed.append(parts[:9])
                rows.append(parts[9:])
    return samples, fixed, np.array(rows)


def _dosage(arr, sep):
    import numpy as np
    out = np.full(arr.shape, -1, dtype=np.int8)
    for gt, v in ((f'0{sep}0', 0), (f'0{sep}1', 1), (f'1{sep}0', 1), (f'1{sep}1', 2)):
        out[arr == gt] = v
    return out.T                                     # (samples, snps)


def _tsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


@pytest.fixture(scope='module')
def vcfs(sim_dir):
    samples, fixed, obs = _load_vcf(_path(sim_dir, '.vcf'))
    t_samples, _, truth = _load_vcf(_path(sim_dir, '_truth_phased.vcf.gz'))
    assert samples == t_samples
    return samples, fixed, _dosage(obs, '/'), _dosage(truth, '|')


@pytest.fixture(scope='module')
def samples_tsv(sim_dir):
    return {r['ID']: r for r in _tsv(_path(sim_dir, '_truth_samples.tsv'))}


def test_cohort_dimensions(sim_dir, vcfs):
    samples, fixed, obs, truth = vcfs
    assert len(samples) == 500 and obs.shape == (500, 1000)
    assert collections.Counter(r[0] for r in fixed) == {f'ch{c:02d}': 50 for c in range(1, 21)}
    gz_samples, gz_fixed, _ = _load_vcf(_path(sim_dir, '.vcf.gz'))
    assert gz_samples == samples and gz_fixed == fixed


def test_sample_ids_are_anonymous_and_order_carries_no_family_signal(vcfs, samples_tsv):
    samples = vcfs[0]
    assert samples == [f'S{i:04d}' for i in range(1, 501)]
    fids = [samples_tsv[s]['FID'] for s in samples]
    adjacent = sum(a == b for a, b in zip(fids, fids[1:])) / (len(fids) - 1)
    chance = sum((n / 500) ** 2 for n in collections.Counter(fids).values())
    assert adjacent < chance + 0.05


def test_genetic_map_is_monotone_and_matches_vcf(sim_dir, vcfs):
    fixed = vcfs[1]
    rows = [l.split('\t') for l in open(_path(sim_dir, '_truth.map')).read().splitlines()]
    assert [(r[0], r[3]) for r in rows] == [(f[0], f[1]) for f in fixed]
    last = {}
    for chrom, _, cm, _ in rows:
        assert float(cm) >= last.get(chrom, -1)
        last[chrom] = float(cm)


def test_truth_is_mendelian_consistent_and_covers_both_trios_and_selfing(sim_dir, vcfs):
    samples, _, _, truth = vcfs
    col = {s: i for i, s in enumerate(samples)}
    n_checked = n_selfed = 0
    for line in open(_path(sim_dir, '_truth.fam')):
        _, iid, pat, mat, *_ = line.split()
        if pat == '0' or mat == '0':
            continue
        c, p, m = truth[col[iid]], truth[col[pat]], truth[col[mat]]
        impossible = (((p == 0) & (c == 2)) | ((m == 0) & (c == 2)) |
                      ((p == 2) & (c == 0)) | ((m == 2) & (c == 0)) |
                      ((p == 0) & (m == 0) & (c > 0)) | ((p == 2) & (m == 2) & (c < 2)))
        assert not impossible.any(), f"Mendelian violation in truth for {iid}"
        n_checked += 1
        n_selfed += pat == mat
    assert n_checked >= 200 and n_selfed >= 20


def test_duplicates_identical_in_truth_and_noisy_when_observed(sim_dir, vcfs, samples_tsv):
    import numpy as np
    samples, _, obs, truth = vcfs
    col = {s: i for i, s in enumerate(samples)}
    pairs = [(r['ID1'], r['ID2']) for r in _tsv(_path(sim_dir, '_truth_duplicates.tsv'))]
    clones = [s for s, r in samples_tsv.items() if r['DUPLICATE_OF']]
    assert len(clones) == 30 and len(pairs) >= 30
    concordance = []
    for a, b in pairs:
        assert (truth[col[a]] == truth[col[b]]).all()
        ok = (obs[col[a]] >= 0) & (obs[col[b]] >= 0)
        concordance.append((obs[col[a]] == obs[col[b]])[ok].mean())
    assert max(concordance) < 1.0 and np.mean(concordance) > 0.97     # "slight" genotyping error
    for c in clones:                                                  # clone -> its source is a listed pair
        assert tuple(sorted((c, samples_tsv[c]['DUPLICATE_OF']))) in {tuple(sorted(p)) for p in pairs}


def test_planted_quality_problems(sim_dir, vcfs, samples_tsv):
    import numpy as np
    samples, _, obs, truth = vcfs
    flags = {s: set(samples_tsv[s]['FLAGS'].split(';')) for s in samples}
    by = {k: [i for i, s in enumerate(samples) if k in flags[s]]
          for k in ('low_callrate', 'high_error', 'contaminated')}
    assert all(len(v) >= 1 for v in by.values()), by
    assert np.mean([(obs[i] < 0).mean() for i in by['low_callrate']]) > 0.08
    snps = _tsv(_path(sim_dir, '_truth_snps.tsv'))
    bad = [j for j, r in enumerate(snps) if r['BAD_MARKER'] == 'Y']
    good = [j for j in range(1000) if j not in set(bad)]
    err = lambda cols: ((obs[:, cols] != truth[:, cols]) & (obs[:, cols] >= 0)).sum() / (obs[:, cols] >= 0).sum()
    assert len(bad) == 8 and err(bad) > 0.10 and err(good) < 0.01
    normal = [i for i, s in enumerate(samples) if not (flags[s] & {'low_callrate', 'high_error', 'contaminated'})]
    normal_err = ((obs[normal][:, good] != truth[normal][:, good]) & (obs[normal][:, good] >= 0)).sum() \
        / (obs[normal][:, good] >= 0).sum()
    assert 0.001 < normal_err < 0.006
    # Bad markers are visible to vcfped's QUAL filter.
    qual = {j: float(r[5]) for j, r in enumerate(_load_vcf(_path(sim_dir, '.vcf'))[1])}
    assert max(qual[j] for j in bad) < min(qual[j] for j in good)


def test_edge_case_coverage(sim_dir, samples_tsv):
    rows = list(samples_tsv.values())
    scenarios = {r['SCENARIO'] for r in rows}
    flags = {f for r in rows for f in r['FLAGS'].split(';') if f}
    for needed in ('trio', 'cross_pop_trio', 'fullsib', 'fullsib_noparents', 'large_sibship',
                   'one_parent_genotyped', 'halfsib', 'halfsib_shared_ungenotyped', 'multigen',
                   'multigen_gaps', 'backcross', 'backcross_other', 'grandparent_cross', 'selfing',
                   'inbred_mating', 'hybrid', 'cryptic_cousins', 'cryptic_sibs_as_founders',
                   'cryptic_uncle_niece', 'role_switch', 'missing_middle', 'unrelated'):
        assert needed in scenarios, f"missing scenario {needed}"
    for needed in ('selfed', 'inbred', 'parent_x_own_offspring', 'parents_are_relatives',
                   'parent_ungenotyped', 'has_duplicate', 'is_duplicate', 'BC1', 'BC2', 'BC3',
                   'F6', 'hub_both_roles'):
        assert needed in flags, f"missing flag {needed}"
    genotyped = [r for r in rows if r['GENOTYPED'] == 'Y']
    assert len(genotyped) == 500 and len(rows) > 550              # ungenotyped ancestors exist
    assert max(float(r['F_PED']) for r in genotyped) > 0.9        # deeply selfed lines
    assert max(int(r['GENERATION']) for r in genotyped) >= 5      # several generations deep
    assert collections.Counter(r['FID'] for r in genotyped).most_common(1)[0][1] >= 12


def test_role_switching_hub_is_pollen_parent_and_seed_parent(sim_dir, samples_tsv):
    hubs = [s for s, r in samples_tsv.items() if 'hub_both_roles' in r['FLAGS'].split(';')]
    fam = [l.split() for l in open(_path(sim_dir, '_truth.fam'))]
    for hub in hubs:
        assert any(f[2] == hub for f in fam) and any(f[3] == hub for f in fam)


def test_relationships_realised_vs_expected_kinship(sim_dir):
    import numpy as np
    rows = _tsv(_path(sim_dir, '_truth_relationships.tsv'))
    simple = lambda rel: [float(r['PHI_REALISED']) for r in rows
                          if r['REL'] == rel and r['COMPLEX'] == 'N']
    assert set(simple('parent_offspring')) == {0.25}
    assert set(simple('duplicate')) == {0.5}
    assert np.mean(simple('full_sib')) == pytest.approx(0.25, abs=0.02)
    assert np.mean(simple('half_sib')) == pytest.approx(0.125, abs=0.02)
    assert any(r['COMPLEX'] == 'Y' for r in rows)                # backcross/inbred multi-path pairs


def test_truth_files_parse_with_project_loaders(sim_dir):
    """The truth files are in the formats the pipeline's own utilities read."""
    ibd_blocks = import_module_from_path('analyze_ibd_blocks',
                                         os.path.join(UTILITIES_DIR, 'analyze_ibd_blocks.py'))
    hbd = import_module_from_path('analyze_hbd', os.path.join(UTILITIES_DIR, 'analyze_hbd.py'))
    cer = import_module_from_path('calibrate_error_rate',
                                  os.path.join(UTILITIES_DIR, 'calibrate_error_rate.py'))
    f2t = import_module_from_path('fam_to_trio', os.path.join(UTILITIES_DIR, 'fam_to_trio.py'))

    pair_stats = ibd_blocks.load_ibd_pair_stats(_path(sim_dir, '_truth_ibd.ibd.gz'))
    assert len(pair_stats) > 1000
    by_individual, id_mismatches = hbd.load_hbd(_path(sim_dir, '_truth_hbd.hbd.gz'))
    assert by_individual and id_mismatches == 0
    assert len(cer.load_confirmed_duplicate_pairs(_path(sim_dir, '_truth_duplicates.tsv'))) >= 30
    assert len(f2t.fam_to_pairs(_path(sim_dir, '_truth.fam'))) >= 200


def test_hbd_tracks_inbreeding(sim_dir, samples_tsv):
    """Deeply selfed lines are mostly autozygous; pedigree-outbred individuals have no HBD."""
    total_cm, last = 0.0, {}
    for line in open(_path(sim_dir, '_truth.map')):
        chrom, _, cm, _ = line.split()
        last[chrom] = float(cm)
    total_cm = sum(last.values())
    cm_by_ind = collections.defaultdict(float)
    with gzip.open(_path(sim_dir, '_truth_hbd.hbd.gz'), 'rt') as f:
        for line in f:
            p = line.split('\t')
            cm_by_ind[p[0]] += float(p[7])
    genotyped = [(s, float(r['F_PED'])) for s, r in samples_tsv.items() if r['GENOTYPED'] == 'Y']
    inbred = [s for s, F in genotyped if F > 0.9]
    outbred = [s for s, F in genotyped if F == 0]
    assert inbred and outbred
    assert sum(cm_by_ind[s] for s in inbred) / len(inbred) / total_cm > 0.7
    assert all(cm_by_ind[s] == 0 for s in outbred)


def test_conserved_region_marked_in_snp_table(sim_dir):
    region = _tsv(_path(sim_dir, '_truth_conserved_regions.tsv'))
    assert len(region) == 1
    flagged = [r for r in _tsv(_path(sim_dir, '_truth_snps.tsv')) if r['CONSERVED_BLOCK'] == 'Y']
    assert flagged and {r['CHROM'] for r in flagged} == {region[0]['CHROM']}
    assert int(region[0]['START']) == min(int(r['POS']) for r in flagged)
