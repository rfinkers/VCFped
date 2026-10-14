#!/usr/bin/env python3
"""
Simulate a plant-breeding validation cohort for VCFped: ~500 diploid samples,
1000 SNPs on 20 chromosomes, many small multi-generation pedigrees, duplicates,
genotyping error -- plus phased ground truth in PLINK/BEAGLE/hap-IBD formats.

DESIGN (see breeding_sim/README.md for the literature behind each choice)
------------------------------------------------------------------------
* Real meiosis, not Mendelian-per-locus: every gamete is built by crossing over
  two parental haplotypes along a genetic map (gamma interference model,
  pericentromeric suppression).  Linkage is therefore correct, which matters
  for hap-IBD / BEAGLE and for the realised (vs expected) kinship of relatives.
* Every haplotype carries a founder-haplotype label per SNP, so TRUE IBD/HBD
  segments and realised kinship fall out exactly (no inference involved).
* Founders come from 3 sub-populations (Balding-Nichols, --fst 0.04), SNPs
  independent within a population (sequoia-style); a conserved block is planted
  on one chromosome to create a population-wide IBD "hotspot".
* Scripted edge-case families (SCENARIOS) guarantee coverage; small random
  families fill the rest, unrelated controls make up the target sample size.
* The observation model is applied last and independently per VCF column:
  genotyping error, missingness, a few poor-quality/contaminated samples, a few
  bad markers.  Clones get independent noise, so duplicates differ slightly.
* Sample IDs are anonymous and column order is shuffled (like real cohorts), so
  nothing about the truth can be inferred from names or positions.

Outputs (<out-dir>/, prefix breeding_sim)
-----------------------------------------
  breeding_sim.vcf(.gz)                 unphased input for vcfped (errors + missing)
  breeding_sim_truth.fam                PLINK/BEAGLE .fam of the genotyped samples
  breeding_sim_truth_phased.vcf.gz      error-free phased haplotypes (hap1 = paternal gamete)
  breeding_sim_truth.map                true genetic map (PLINK .map, cM): hap-IBD/BEAGLE map=
  breeding_sim_truth_ibd.ibd.gz         true pedigree-IBD segments (hap-IBD 8-column format)
  breeding_sim_truth_hbd.hbd.gz         true autozygous segments (id1 == id2)
  breeding_sim_truth_duplicates.tsv     ID1/ID2 pairs of true duplicates
  breeding_sim_truth_relationships.tsv  pairwise class + expected/realised kinship
  breeding_sim_truth_samples.tsv        every individual incl. ungenotyped ancestors
  breeding_sim_truth_snps.tsv           per-SNP info incl. planted bad markers
  breeding_sim_truth_conserved_regions.tsv  planted hotspot window

Usage
-----
  python generate_breeding_sim.py                    # seed 42, 500 samples x 1000 SNPs
  python generate_breeding_sim.py --seed 7 --snps-per-chr 200
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict
from itertools import combinations

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = "breeding_sim"

N_POPS = 3
KING_3RD_DEGREE = 0.0442    # lowest kinship KING / the pipeline's --king-table-filter reports
NOMINAL_PHI = {'duplicate': 0.5, 'parent_offspring': 0.25, 'full_sib': 0.25,
               'half_sib': 0.125, 'grandparent': 0.125, 'avuncular': 0.125,
               'first_cousin': 0.0625}


# ---------------------------------------------------------------------------
# Genome: chromosomes, SNP positions, genetic map
# ---------------------------------------------------------------------------

class Genome:
    """n_chr chromosomes x snps_per_chr SNPs, flattened to one SNP axis of length M."""

    def __init__(self, n_chr, snps_per_chr, rng):
        self.n_chr = n_chr
        self.chr_names = [f"ch{c + 1:02d}" for c in range(n_chr)]
        self.len_cM = np.round(rng.uniform(70, 130, n_chr), 1)       # plant-typical
        self.len_bp = (rng.uniform(30, 60, n_chr) * 1e6).astype(int)
        grid = np.linspace(0, 1, 501)
        pos_bp, pos_cM, self.slices = [], [], []
        for c in range(n_chr):
            # Crossover rate dips around a pericentromere (typical of plants): genetic
            # position = len_cM * cumulative(rate profile) at the physical fraction.
            centre = rng.uniform(0.3, 0.7)
            rate = 1.0 - 0.85 * np.exp(-0.5 * ((grid - centre) / 0.1) ** 2)
            cum = np.concatenate([[0], np.cumsum((rate[1:] + rate[:-1]) / 2)])
            cum /= cum[-1]
            frac = np.sort(rng.choice(np.arange(1000, 999000), snps_per_chr, replace=False)) / 1e6
            pos_bp.append((frac * self.len_bp[c]).astype(int))
            pos_cM.append(np.interp(frac, grid, cum) * self.len_cM[c])
            self.slices.append(slice(c * snps_per_chr, (c + 1) * snps_per_chr))
        self.pos_bp = np.concatenate(pos_bp)
        self.pos_cM = np.concatenate(pos_cM)
        self.chrom_of = np.repeat(np.arange(n_chr), snps_per_chr)
        self.M = n_chr * snps_per_chr
        self.same_chrom_as_prev = np.concatenate([[False], self.chrom_of[1:] == self.chrom_of[:-1]])


# ---------------------------------------------------------------------------
# Meiosis and the simulated population
# ---------------------------------------------------------------------------

def crossover_positions(length_cM, nu, rng):
    """
    Crossover positions (cM) in one gamete.  Chiasmata form a stationary gamma
    renewal process (shape nu, mean spacing 0.5 M); a gamete inherits each with
    p=1/2, giving 1 crossover per Morgan on average (Broman et al. 2002; the
    chi-square model in R/qtl).  nu=1 is Haldane (no interference).
    """
    L = length_cM / 100.0
    pts = []
    x = rng.gamma(rng.integers(1, nu + 1), 1.0 / (2 * nu))     # equilibrium first point
    while x < L:
        pts.append(x)
        x += rng.gamma(nu, 1.0 / (2 * nu))
    pts = np.array(pts)
    return np.sort(pts[rng.random(len(pts)) < 0.5]) * 100.0


class Individual:
    __slots__ = ('iid', 'pat', 'mat', 'gen', 'tag', 'genotyped', 'hap', 'lab', 'dup_of', 'flags')

    def __init__(self, iid, pat, mat, gen, tag, genotyped):
        self.iid, self.pat, self.mat = iid, pat, mat
        self.gen, self.tag, self.genotyped = gen, tag, genotyped
        self.hap = self.lab = None       # (2, M) alleles / founder-haplotype labels
        self.dup_of = None
        self.flags = set()


class Simulation:
    """Holds every individual (genotyped or not) in topological order, with true haplotypes."""

    def __init__(self, genome, nu, fst, rng_ped, rng_gen):
        self.g, self.nu, self.fst = genome, nu, fst
        self.rp = rng_ped            # pedigree-structure decisions
        self.rg = rng_gen            # allele / meiosis randomness
        self.inds = {}               # iid -> Individual (insertion order is topological)
        self.n_hap_labels = 0
        self._build_population_freqs()

    def _build_population_freqs(self):
        M = self.g.M
        anc = np.where(self.rg.random(M) < 0.9, self.rg.uniform(0.15, 0.5, M),
                       self.rg.uniform(0.02, 0.15, M))
        anc = np.where(self.rg.random(M) < 0.5, anc, 1 - anc)     # random minor allele
        a, b = anc * (1 - self.fst) / self.fst, (1 - anc) * (1 - self.fst) / self.fst
        self.pop_freq = np.clip(self.rg.beta(a, b, size=(N_POPS, M)), 0.005, 0.995)
        # Conserved block: every founder carries ~the same haplotype over the first 40%
        # of ch05 (introgression / sweep) -> a population-wide IBD hotspot for hap-IBD.
        sl = self.g.slices[min(4, self.g.n_chr - 1)]
        self.conserved_idx = np.arange(sl.start, sl.start + (sl.stop - sl.start) * 2 // 5)
        self.conserved_hap = (self.rg.random(len(self.conserved_idx)) < 0.5).astype(np.int8)

    def _founder_haplotype(self, pop):
        hap = (self.rg.random(self.g.M) < self.pop_freq[pop]).astype(np.int8)
        keep = self.rg.random(len(self.conserved_idx)) < 0.98     # rare divergence
        hap[self.conserved_idx] = np.where(keep, self.conserved_hap, hap[self.conserved_idx])
        return hap

    def _new_id(self):
        return f"I{len(self.inds) + 1:05d}"

    def add_founder(self, tag, pop=None, genotyped=True):
        pop = int(self.rp.integers(N_POPS)) if pop is None else pop
        ind = Individual(self._new_id(), None, None, 0, tag, genotyped)
        ind.hap = np.stack([self._founder_haplotype(pop) for _ in range(2)])
        ind.lab = np.stack([np.full(self.g.M, self.n_hap_labels + h, dtype=np.int32)
                            for h in range(2)])
        self.n_hap_labels += 2
        ind.flags.add(f'pop{pop}')
        self.inds[ind.iid] = ind
        return ind.iid

    def _gamete(self, parent):
        hap = np.empty(self.g.M, dtype=np.int8)
        lab = np.empty(self.g.M, dtype=np.int32)
        for c, sl in enumerate(self.g.slices):
            xs = crossover_positions(self.g.len_cM[c], self.nu, self.rg)
            which = (np.searchsorted(xs, self.g.pos_cM[sl]) & 1) ^ int(self.rg.integers(2))
            cols = np.arange(sl.start, sl.stop)
            hap[sl], lab[sl] = parent.hap[which, cols], parent.lab[which, cols]
        return hap, lab

    def add_child(self, pat, mat, tag, genotyped=True):
        """pat/mat = pollen / seed parent; pat == mat is selfing."""
        p, m = self.inds[pat], self.inds[mat]
        ind = Individual(self._new_id(), pat, mat, max(p.gen, m.gen) + 1, tag, genotyped)
        (h0, l0), (h1, l1) = self._gamete(p), self._gamete(m)
        ind.hap, ind.lab = np.stack([h0, h1]), np.stack([l0, l1])
        self.inds[ind.iid] = ind
        return ind.iid

    def add_clone(self, src):
        s = self.inds[src]
        ind = Individual(self._new_id(), None, None, s.gen, 'duplicate', True)
        ind.hap, ind.lab, ind.dup_of = s.hap, s.lab, src          # shared arrays: identical genome
        self.inds[ind.iid] = ind
        return ind.iid

    def cross(self, a, b, tag, genotyped=True):
        """Cross two hermaphrodites; which is pollen vs seed parent is random (reciprocal)."""
        if self.rp.random() < 0.5:
            a, b = b, a
        return self.add_child(a, b, tag, genotyped)

    def n_genotyped(self):
        return sum(i.genotyped for i in self.inds.values())

    def rollback(self, n_keep):
        for iid in list(self.inds)[n_keep:]:
            del self.inds[iid]


# ---------------------------------------------------------------------------
# Scenario builders: each grows sim.inds and tags its members.
# ---------------------------------------------------------------------------

def _kids(sim, a, b, n, tag, p_geno=1.0):
    return [sim.cross(a, b, tag, genotyped=bool(sim.rp.random() < p_geno)) for _ in range(n)]


def sc_trio(sim):
    a, b = sim.add_founder('trio'), sim.add_founder('trio')
    _kids(sim, a, b, 1, 'trio')


def sc_cross_pop_trio(sim):
    a, b = sim.add_founder('cross_pop_trio', pop=0), sim.add_founder('cross_pop_trio', pop=2)
    _kids(sim, a, b, 1, 'cross_pop_trio')


def sc_fullsib(sim, n_kids, parents_genotyped=True):
    """Full-sib family; with parents_genotyped=False only the kinship signal identifies it."""
    tag = 'fullsib' if parents_genotyped else 'fullsib_noparents'
    a = sim.add_founder(tag, genotyped=parents_genotyped)
    b = sim.add_founder(tag, genotyped=parents_genotyped)
    _kids(sim, a, b, n_kids, tag)


def sc_one_parent_only(sim, n_kids):
    """Seed parent genotyped, pollen donor lost: parent-offspring 'duos' that are also full sibs."""
    a = sim.add_founder('one_parent_genotyped')
    b = sim.add_founder('one_parent_genotyped', genotyped=False)
    _kids(sim, a, b, n_kids, 'one_parent_genotyped')


def sc_large_sibship(sim, n_kids):
    a, b = sim.add_founder('large_sibship'), sim.add_founder('large_sibship')
    _kids(sim, a, b, n_kids, 'large_sibship')


def sc_halfsib(sim, n_mates, kids_each, shared_genotyped=True):
    tag = 'halfsib' if shared_genotyped else 'halfsib_shared_ungenotyped'
    shared = sim.add_founder(tag, genotyped=shared_genotyped)
    for _ in range(n_mates):
        mate = sim.add_founder(tag, genotyped=bool(sim.rp.random() < 0.7))
        _kids(sim, shared, mate, kids_each, tag)


def sc_multigen(sim, depth, tag='multigen', p_missing=0.0, max_breeders=3):
    """
    Random multi-generation family.  Up to max_breeders members of each generation
    outcross to a new founder, back-cross to a parent, mate with a sibling/cousin,
    or self.  p_missing = probability a descendant is not genotyped.
    """
    r = sim.rp
    a, b = sim.add_founder(tag), sim.add_founder(tag)
    members = [sim.cross(a, b, tag, genotyped=bool(r.random() >= p_missing))
               for _ in range(int(r.integers(2, 5)))]
    for _ in range(depth - 2):
        breeders = [members[k] for k in r.permutation(len(members))[:max_breeders]]
        nxt = []
        for m in breeders:
            x = r.random()
            others = [o for o in members if o != m]
            if x < 0.70:                                          # outcross to a new founder
                mate = sim.add_founder(tag, genotyped=bool(r.random() >= p_missing))
            elif x < 0.80 and others:                             # sibling / cousin mating
                mate = others[int(r.integers(len(others)))]
            elif x < 0.92:                                        # back-cross to a parent
                mate = sim.inds[m].pat
            else:                                                 # selfing
                mate = m
            nxt += _kids(sim, m, mate, int(r.integers(1, 4)), tag, p_geno=1 - p_missing)
        members = nxt or members


def sc_backcross(sim, rounds, extra_sibs=1):
    """
    P1 x P2 -> F1; F1 x P1 -> BC1; BC1 x P1 -> BC2 ... (recurrent parent P1).  In every
    cross one parent is the other's parent/ancestor, so the two 'parents' are first-degree
    relatives of each other -- the 'inverted / grandparent chain' trap for trio logic.
    """
    p1, p2 = sim.add_founder('backcross'), sim.add_founder('backcross')
    cur = sim.cross(p1, p2, 'backcross')
    sim.inds[cur].flags.add('F1')
    for r in range(rounds):
        sibs = _kids(sim, cur, p1, 1 + extra_sibs, 'backcross')
        for s in sibs:
            sim.inds[s].flags.add(f'BC{r + 1}')
        cur = sibs[0]


def sc_backcross_other_parent(sim):
    """F1 x P2 (the OTHER original parent); two F1 sibs each backcrossed to a different parent."""
    p1, p2 = sim.add_founder('backcross_other'), sim.add_founder('backcross_other')
    f1a, f1b = _kids(sim, p1, p2, 2, 'backcross_other')
    _kids(sim, f1a, p2, 2, 'backcross_other')
    _kids(sim, f1b, p1, 1, 'backcross_other')


def sc_grandparent_cross(sim):
    """P1xP2 -> F1; F1xP3 -> G; G x P1 (G's grandparent) -> H; G x F1' (G's aunt/uncle) -> H'."""
    p1, p2, p3 = (sim.add_founder('grandparent_cross') for _ in range(3))
    f1, f1b = _kids(sim, p1, p2, 2, 'grandparent_cross')
    g = _kids(sim, f1, p3, 1, 'grandparent_cross')[0]
    _kids(sim, g, p1, 2, 'grandparent_cross')
    _kids(sim, g, f1b, 1, 'grandparent_cross')


def sc_selfing(sim, depth, branch=2, p_geno=0.7):
    """Selfing tree (bulk F2/F3, then single-seed descent); the root is the F1."""
    level = [sim.add_founder('selfing')]
    for g in range(depth):
        nxt = []
        for m in level:
            nxt += [sim.add_child(m, m, 'selfing', genotyped=bool(sim.rp.random() < p_geno))
                    for _ in range(branch if g < 2 else 1)]
        for m in nxt:
            sim.inds[m].flags.add(f'F{g + 2}')
        level = nxt


def sc_inbred_mating(sim):
    """Full-sib mating (F=1/4), half-sib mating (F=1/8), first-cousin mating (F=1/16)."""
    t = 'inbred_mating'
    a, b, c = (sim.add_founder(t) for _ in range(3))
    fs = _kids(sim, a, b, 3, t)
    hs = _kids(sim, a, c, 2, t)
    _kids(sim, fs[0], fs[1], 2, t)
    _kids(sim, fs[2], hs[0], 1, t)
    d1 = _kids(sim, fs[0], sim.add_founder(t), 1, t)[0]
    d2 = _kids(sim, fs[1], sim.add_founder(t), 1, t)[0]
    _kids(sim, d1, d2, 1, t)


def make_inbred_line(sim, gens=5, pop=None):
    """Founder selfed `gens` times (-> F6, ~97% homozygous); only the final line is genotyped."""
    cur = sim.add_founder('hybrid', pop=pop, genotyped=False)
    for g in range(gens):
        cur = sim.add_child(cur, cur, 'hybrid', genotyped=(g == gens - 1))
    sim.inds[cur].flags.add(f'F{gens + 1}')
    return cur


def sc_hybrid(sim, n_f2=3):
    """Two inbred lines -> F1 hybrids -> F2 (sib-cross and selfed).  T2 (BB+BB=BB) is uninformative."""
    l1 = make_inbred_line(sim, pop=int(sim.rp.integers(N_POPS)))
    l2 = make_inbred_line(sim, pop=int(sim.rp.integers(N_POPS)))
    f1a, f1b = _kids(sim, l1, l2, 2, 'hybrid')
    _kids(sim, f1a, f1b, n_f2, 'hybrid')
    _kids(sim, f1a, f1a, 2, 'hybrid')


def sc_cryptic(sim, kind):
    """
    Genotyped individuals that are related through UNGENOTYPED ancestors:
      cousins           first cousins whose parents/grandparents were lost
      sibs_as_founders  two full sibs (parents lost) each starting a family
      uncle_niece       uncle x niece mating (parents lost)
    """
    t = f'cryptic_{kind}'
    gp1, gp2 = sim.add_founder(t, genotyped=False), sim.add_founder(t, genotyped=False)
    sibs = _kids(sim, gp1, gp2, 3, t, p_geno=0.0)
    if kind == 'sibs_as_founders':
        for s in sibs[:2]:
            sim.inds[s].genotyped = True
            _kids(sim, s, sim.add_founder(t), 2, t)
    elif kind == 'cousins':
        for s in sibs[:2]:
            _kids(sim, s, sim.add_founder(t, genotyped=False), 2, t)
    else:
        niece = _kids(sim, sibs[0], sim.add_founder(t, genotyped=False), 1, t)[0]
        sim.inds[sibs[2]].genotyped = True
        _kids(sim, sibs[2], niece, 2, t)


def sc_role_switch(sim):
    """One hermaphrodite used as pollen parent in two crosses and as seed parent in two others."""
    hub, x, y = (sim.add_founder('role_switch') for _ in range(3))
    sim.inds[hub].flags.add('hub_both_roles')
    for pat, mat in ((hub, x), (hub, x), (y, hub), (y, hub)):
        sim.add_child(pat, mat, 'role_switch')


def sc_missing_middle(sim):
    """Grandparents and grandchildren genotyped; the parent generation was lost."""
    t = 'missing_middle'
    gp1, gp2 = sim.add_founder(t), sim.add_founder(t)
    for m in _kids(sim, gp1, gp2, 2, t, p_geno=0.0):
        _kids(sim, m, sim.add_founder(t), 2, t)


def random_family(sim):
    """A small random family (bounded size) used to fill the cohort."""
    r = sim.rp
    kind = ('multigen', 'multigen', 'fullsib', 'halfsib', 'trio')[int(r.integers(5))]
    if kind == 'multigen':
        sc_multigen(sim, int(r.integers(3, 5)), tag='random_multigen', p_missing=0.12)
    elif kind == 'fullsib':
        sc_fullsib(sim, int(r.integers(2, 6)))
    elif kind == 'halfsib':
        sc_halfsib(sim, 2, int(r.integers(1, 3)))
    else:
        sc_trio(sim)


# The guaranteed edge-case set (builder, kwargs).
SCENARIOS = (
    [(sc_trio, {})] * 6
    + [(sc_cross_pop_trio, {})] * 3
    + [(sc_fullsib, {'n_kids': k}) for k in (2, 3, 4, 6)]
    + [(sc_fullsib, {'n_kids': k, 'parents_genotyped': False}) for k in (2, 3, 4, 5)]
    + [(sc_large_sibship, {'n_kids': 12})]
    + [(sc_one_parent_only, {'n_kids': 3}), (sc_one_parent_only, {'n_kids': 2})]
    + [(sc_halfsib, {'n_mates': 4, 'kids_each': 2}), (sc_halfsib, {'n_mates': 3, 'kids_each': 1}),
       (sc_halfsib, {'n_mates': 3, 'kids_each': 2, 'shared_genotyped': False})]
    + [(sc_multigen, {'depth': d}) for d in (3, 4, 5)]
    + [(sc_multigen, {'depth': 4, 'p_missing': 0.25, 'tag': 'multigen_gaps'})]
    + [(sc_backcross, {'rounds': r}) for r in (1, 2, 3, 4)]
    + [(sc_backcross_other_parent, {}), (sc_grandparent_cross, {}), (sc_grandparent_cross, {})]
    + [(sc_selfing, {'depth': d}) for d in (4, 5, 5)]
    + [(sc_inbred_mating, {})] * 2
    + [(sc_hybrid, {'n_f2': n}) for n in (3, 4, 2)]
    + [(sc_cryptic, {'kind': k}) for k in ('cousins', 'sibs_as_founders', 'uncle_niece')]
    + [(sc_role_switch, {})] * 2
    + [(sc_missing_middle, {})]
)


def add_duplicates(sim, n_target):
    """
    Add clone samples of genotyped individuals: pairs and some triplicates, half of
    them clones of individuals used as parents (the case that matters for trios).
    """
    rng = sim.rp
    parents = {p for i in sim.inds.values() for p in (i.pat, i.mat) if p}
    cand = [i.iid for i in sim.inds.values() if i.genotyped and i.dup_of is None]
    as_parent = [c for c in cand if c in parents]
    as_leaf = [c for c in cand if c not in parents]
    made = 0
    while made < n_target:
        pool = as_parent if (as_parent and rng.random() < 0.5) else as_leaf
        src = pool[int(rng.integers(len(pool)))]
        for _ in range(min(2 if rng.random() < 0.15 else 1, n_target - made)):
            sim.add_clone(src)
            made += 1
        sim.inds[src].flags.add('has_duplicate')


def build_cohort(sim, n_samples, n_dups, min_unrelated=25, max_attempts=200):
    """Scripted families, then random families that fit, then clones, then unrelated controls."""
    for builder, kw in SCENARIOS:
        builder(sim, **kw)
    budget = n_samples - n_dups - min_unrelated
    if sim.n_genotyped() > budget:
        sys.exit(f"Scripted scenarios already give {sim.n_genotyped()} genotyped samples; "
                 f"--n-samples must exceed {sim.n_genotyped() + n_dups + min_unrelated}.")
    for _ in range(max_attempts):
        before = len(sim.inds)
        random_family(sim)
        if sim.n_genotyped() > budget:               # does not fit: discard and retry
            sim.rollback(before)
    add_duplicates(sim, n_dups)
    while sim.n_genotyped() < n_samples:
        sim.add_founder('unrelated')


# ---------------------------------------------------------------------------
# Pedigree-derived truth: inbreeding, kinship, relationship classes
# ---------------------------------------------------------------------------

def canonical(sim, iid):
    return sim.inds[iid].dup_of or iid


def numerator_relationship(sim):
    """Tabular method over non-clone individuals: A = 2 * kinship, A_ii = 1 + F_i."""
    ids = [i for i, ind in sim.inds.items() if ind.dup_of is None]
    idx = {iid: k for k, iid in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)))
    for k, iid in enumerate(ids):
        ind = sim.inds[iid]
        if ind.pat is None:
            A[k, k] = 1.0
            continue
        s, d = idx[ind.pat], idx[ind.mat]
        A[k, :k] = A[:k, k] = 0.5 * (A[s, :k] + A[d, :k])
        A[k, k] = 1.0 + 0.5 * A[s, d]
    return idx, A


def components(sim):
    """Family label (connected component of the parent-child graph, clones joined to their source)."""
    parent = {i: i for i in sim.inds}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for iid, ind in sim.inds.items():
        for p in (ind.pat, ind.mat, ind.dup_of):
            if p:
                parent[find(iid)] = find(p)
    labels = {}
    return {iid: labels.setdefault(find(iid), f"F{len(labels) + 1:03d}") for iid in sim.inds}


def parents_of(sim, iid):
    """(pat, mat) of the underlying individual (a clone has its source's parents)."""
    c = sim.inds[canonical(sim, iid)]
    return c.pat, c.mat


def classify_pair(sim, a, b):
    """Graph-based relationship class of two genotyped samples."""
    ca, cb = canonical(sim, a), canonical(sim, b)
    if ca == cb:
        return 'duplicate'

    def par(x):
        return {sim.inds[x].pat, sim.inds[x].mat} - {None}
    pa, pb = par(ca), par(cb)
    if cb in pa or ca in pb:
        return 'parent_offspring'
    if pa and pa == pb:
        return 'full_sib'
    if pa & pb:
        return 'half_sib'
    gpa = set().union(*[par(p) for p in pa]) if pa else set()
    gpb = set().union(*[par(p) for p in pb]) if pb else set()
    if cb in gpa or ca in gpb:
        return 'grandparent'
    if any(par(p) & pb for p in pa) or any(par(p) & pa for p in pb):
        return 'avuncular'                               # one is a (half-)sib of the other's parent
    if gpa & gpb:
        return 'first_cousin'
    return 'other'


# ---------------------------------------------------------------------------
# Observation model: genotyping error, missingness, poor-quality samples
# ---------------------------------------------------------------------------

def observe(true_dosage, err_rate, miss_rate, n_bad_markers, rng, offender_rows=()):
    """
    true_dosage: (n_samples, M) in {0,1,2}.  Returns (observed with -1 = missing,
    {row: set of QC flags}, indices of bad markers).  Error model: a miscalled
    homozygote becomes het (85%) or the opposite homozygote (15%); a miscalled het
    becomes a random homozygote.  offender_rows are preferred for 'high_error'
    (parents of several children -> Mendelian-error 'repeat offenders').
    """
    n, M = true_dosage.shape
    sample_err = np.full(n, err_rate)
    sample_miss = miss_rate * rng.gamma(4.0, 0.25, n)                # heterogeneity, mean 1x
    qc = {i: set() for i in range(n)}

    order = rng.permutation(n)
    for i in order[:4]:
        sample_miss[i] = rng.uniform(0.10, 0.25)
        qc[i].add('low_callrate')
    pref = [i for i in rng.permutation(list(offender_rows)) if not qc[i]] if len(offender_rows) else []
    high_err = list(pref[:2]) + [i for i in order[4:] if i not in pref[:2]][:1]
    for i in high_err:
        sample_err[i] = rng.uniform(0.03, 0.05)
        qc[i].add('high_error')
    contaminated = [i for i in order[6:] if not qc[i] and i not in high_err][:2]
    for i in contaminated:
        qc[i].add('contaminated')

    bad = rng.choice(M, n_bad_markers, replace=False)
    is_err = rng.random((n, M)) < sample_err[:, None]
    is_err[:, bad] = rng.random((n, len(bad))) < rng.uniform(0.15, 0.30, len(bad))[None, :]
    for i in contaminated:                                            # foreign DNA at ~15% of loci
        is_err[i] |= rng.random(M) < 0.15
    was_het = true_dosage == 1
    miscall_het = rng.random((n, M)) < 0.85
    miscall = np.where(was_het, np.where(rng.random((n, M)) < 0.5, 0, 2),
                       np.where(miscall_het, 1, 2 - true_dosage))
    obs = np.where(is_err, miscall, true_dosage)

    miss_mult = np.ones(M)
    miss_mult[bad] = 4.0                                              # bad markers also miss more
    miss = rng.random((n, M)) < np.clip(sample_miss[:, None] * miss_mult[None, :], 0, 0.9)
    return np.where(miss, -1, obs).astype(np.int8), qc, bad


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _open(path):
    return gzip.open(path, 'wt') if path.endswith('.gz') else open(path, 'w')


def write_vcf(path, genome, sample_ids, snp_info, geno_text):
    """geno_text: (M, n_samples) array of genotype strings."""
    with _open(path) as f:
        f.write("##fileformat=VCFv4.2\n##source=generate_breeding_sim.py\n")
        for c, name in enumerate(genome.chr_names):
            f.write(f"##contig=<ID={name},length={int(genome.len_bp[c])}>\n")
        f.write('##FILTER=<ID=PASS,Description="All filters passed">\n'
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(sample_ids) + "\n")
        for j in range(genome.M):
            s = snp_info[j]
            f.write("\t".join([genome.chr_names[genome.chrom_of[j]], str(genome.pos_bp[j]), ".",
                               s['ref'], s['alt'], f"{s['qual']:.0f}", "PASS", ".", "GT"])
                    + "\t" + "\t".join(geno_text[j]) + "\n")


def write_map(path, genome):
    """PLINK .map: chrom, marker id, cM, bp  (the map format hap-IBD and BEAGLE take)."""
    with open(path, 'w') as f:
        for j in range(genome.M):
            c, bp = genome.chr_names[genome.chrom_of[j]], genome.pos_bp[j]
            f.write(f"{c}\t{c}_{bp}\t{genome.pos_cM[j]:.4f}\t{bp}\n")


def ibd_segments(genome, lab_a, lab_b, min_cM, same_individual=False):
    """
    True pedigree-IBD segments between two (2, M) founder-label arrays: maximal runs of
    SNPs (within one chromosome) whose haplotype labels agree, reported hap-IBD style
    (first/last marker, length in cM).  same_individual=True compares an individual's own
    two haplotypes (HBD).
    """
    out = []
    combos = [(0, 1)] if same_individual else [(a, b) for a in (0, 1) for b in (0, 1)]
    for ha, hb in combos:
        eq = lab_a[ha] == lab_b[hb]
        if not eq.any():
            continue
        cont = eq[1:] & eq[:-1] & genome.same_chrom_as_prev[1:]
        starts = np.flatnonzero(eq & ~np.concatenate([[False], cont]))
        ends = np.flatnonzero(eq & ~np.concatenate([cont, [False]]))
        for s, e in zip(starts, ends):
            cm = genome.pos_cM[e] - genome.pos_cM[s]
            if cm >= min_cM:
                out.append((ha + 1, hb + 1, genome.chr_names[genome.chrom_of[s]],
                            int(genome.pos_bp[s]), int(genome.pos_bp[e]), cm))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-chr", type=int, default=20)
    ap.add_argument("--snps-per-chr", type=int, default=50, help="default 50 -> 1000 SNPs")
    ap.add_argument("--n-samples", type=int, default=500, help="VCF columns incl. duplicates")
    ap.add_argument("--n-duplicates", type=int, default=30, help="clone samples among them")
    ap.add_argument("--error-rate", type=float, default=0.003, help="per-genotype miscall rate")
    ap.add_argument("--missing-rate", type=float, default=0.015)
    ap.add_argument("--n-bad-markers", type=int, default=8)
    ap.add_argument("--fst", type=float, default=0.04,
                    help="differentiation of the 3 founder sub-populations (Balding-Nichols)")
    ap.add_argument("--interference-nu", type=int, default=5,
                    help="gamma interference shape (1 = Haldane / no interference)")
    ap.add_argument("--min-ibd-cm", type=float, default=2.0,
                    help="shortest truth IBD segment (hap-IBD min-output default)")
    ap.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "breeding_sim"))
    args = ap.parse_args()

    # Independent streams: changing error rates must not change the pedigree or haplotypes.
    s_ped, s_gen, s_obs, s_shuf = (np.random.default_rng(s)
                                   for s in np.random.SeedSequence(args.seed).spawn(4))
    genome = Genome(args.n_chr, args.snps_per_chr, s_gen)
    sim = Simulation(genome, args.interference_nu, args.fst, s_ped, s_gen)
    build_cohort(sim, args.n_samples, args.n_duplicates)
    inds = sim.inds

    geno_ids = [i for i, ind in inds.items() if ind.genotyped]
    shuffled = [geno_ids[k] for k in s_shuf.permutation(len(geno_ids))]
    vcf_name = {iid: f"S{r + 1:04d}" for r, iid in enumerate(shuffled)}
    vcf_order = sorted(geno_ids, key=vcf_name.get)               # column order = anonymous order
    row_of = {iid: r for r, iid in enumerate(vcf_order)}
    fam_of = components(sim)
    idx, A = numerator_relationship(sim)
    F = {iid: A[idx[iid], idx[iid]] - 1.0 for iid in idx}

    names, n_u = dict(vcf_name), 0                                # + U0001.. for ungenotyped
    for iid in inds:
        if iid not in names:
            n_u += 1
            names[iid] = f"U{n_u:04d}"

    def sname(iid):                                               # '0' for none / ungenotyped
        return names[iid] if iid and inds[canonical(sim, iid)].genotyped else '0'

    true_hap = np.stack([inds[i].hap for i in vcf_order])         # (n, 2, M)
    n_children = defaultdict(int)
    for ind in inds.values():
        if ind.pat and ind.genotyped:
            n_children[ind.pat] += 1
            n_children[ind.mat] += ind.mat != ind.pat
    offender_rows = [row_of[i] for i in vcf_order if n_children[i] >= 3 and inds[i].dup_of is None]
    obs, qc, bad = observe(true_hap.sum(axis=1), args.error_rate, args.missing_rate,
                           args.n_bad_markers, s_obs, offender_rows)
    bad_set = set(bad.tolist())

    bases = [("A", "T"), ("A", "C"), ("G", "T"), ("G", "C"), ("A", "G"), ("C", "T")]
    snp_info = []
    for j in range(genome.M):
        ref, alt = bases[int(s_obs.integers(len(bases)))]
        # Bad markers carry low QUAL so vcfped's QUAL-percentile filter can see them.
        qual = s_obs.uniform(10, 45) if j in bad_set else s_obs.uniform(50, 500)
        snp_info.append({'ref': ref, 'alt': alt, 'qual': qual})

    os.makedirs(args.out_dir, exist_ok=True)

    def out(suffix):
        return os.path.join(args.out_dir, f"{PREFIX}{suffix}")

    sample_ids = [vcf_name[i] for i in vcf_order]
    gt_obs = np.array(['0/0', '0/1', '1/1', './.'])[np.where(obs < 0, 3, obs)].T
    write_vcf(out(".vcf"), genome, sample_ids, snp_info, gt_obs)
    write_vcf(out(".vcf.gz"), genome, sample_ids, snp_info, gt_obs)
    gt_true = np.array(['0|0', '0|1', '1|0', '1|1'])[true_hap[:, 0, :] * 2 + true_hap[:, 1, :]].T
    write_vcf(out("_truth_phased.vcf.gz"), genome, sample_ids, snp_info, gt_true)
    write_map(out("_truth.map"), genome)

    with open(out("_truth.fam"), 'w') as f:
        for iid in vcf_order:
            pat, mat = parents_of(sim, iid)
            f.write(f"{fam_of[iid]} {vcf_name[iid]} {sname(pat)} {sname(mat)} 0 -9\n")

    groups = defaultdict(list)
    for iid in vcf_order:
        groups[canonical(sim, iid)].append(vcf_name[iid])
    dup_pairs = [p for g in groups.values() if len(g) > 1 for p in combinations(sorted(g), 2)]
    with open(out("_truth_duplicates.tsv"), 'w') as f:
        f.write("ID1\tID2\n")
        f.writelines(f"{a}\t{b}\n" for a, b in dup_pairs)

    # Pairwise truth: expected (pedigree) vs realised (from haplotype labels) kinship + IBD.
    n_rel = 0
    with open(out("_truth_relationships.tsv"), 'w') as f, gzip.open(out("_truth_ibd.ibd.gz"), 'wt') as fibd:
        f.write("ID1\tID2\tREL\tPHI_PED\tPHI_REALISED\tNOMINAL_PHI\tCOMPLEX\n")
        for a, b in combinations(vcf_order, 2):
            phi = A[idx[canonical(sim, a)], idx[canonical(sim, b)]] / 2.0
            if phi <= 0:
                continue
            la, lb = inds[a].lab, inds[b].lab
            realised = (la[:, None, :] == lb[None, :, :]).sum(axis=(0, 1)).mean() / 4.0
            rel = classify_pair(sim, a, b)
            if phi >= KING_3RD_DEGREE or rel != 'other':
                nominal = NOMINAL_PHI.get(rel)
                cplx = 'NA' if nominal is None else 'Y' if abs(phi - nominal) > 0.02 else 'N'
                f.write(f"{vcf_name[a]}\t{vcf_name[b]}\t{rel}\t{phi:.4f}\t{realised:.4f}\t"
                        f"{'NA' if nominal is None else nominal}\t{cplx}\n")
                n_rel += 1
            for ha, hb, chrom, s, e, cm in ibd_segments(genome, la, lb, args.min_ibd_cm):
                x, y = sorted([(vcf_name[a], ha), (vcf_name[b], hb)])
                fibd.write(f"{x[0]}\t{x[1]}\t{y[0]}\t{y[1]}\t{chrom}\t{s}\t{e}\t{cm:.3f}\n")

    with gzip.open(out("_truth_hbd.hbd.gz"), 'wt') as f:
        for iid in vcf_order:
            for _, _, chrom, s, e, cm in ibd_segments(genome, inds[iid].lab, inds[iid].lab,
                                                      args.min_ibd_cm, same_individual=True):
                f.write(f"{vcf_name[iid]}\t1\t{vcf_name[iid]}\t2\t{chrom}\t{s}\t{e}\t{cm:.3f}\n")

    def flags_of(iid):
        ind, fl = inds[iid], set(inds[iid].flags)
        c = inds[canonical(sim, iid)]
        fl |= c.flags
        if c.pat:
            P, Mo = inds[c.pat], inds[c.mat]
            if c.pat == c.mat:
                fl.add('selfed')
            else:
                if {P.pat, P.mat} & ({Mo.pat, Mo.mat} - {None}):
                    fl.add('parents_are_relatives')
                if c.pat in (Mo.pat, Mo.mat) or c.mat in (P.pat, P.mat):
                    fl.add('parent_x_own_offspring')
            if not (P.genotyped and Mo.genotyped):
                fl.add('parent_ungenotyped')
        if F[canonical(sim, iid)] > 0.01:
            fl.add('inbred')
        if ind.dup_of:
            fl.add('is_duplicate')
        return fl

    with open(out("_truth_samples.tsv"), 'w') as f:
        f.write("ID\tFID\tPAT\tMAT\tGENERATION\tSCENARIO\tGENOTYPED\tDUPLICATE_OF\t"
                "F_PED\tN_MISSING\tFLAGS\n")
        for iid, ind in inds.items():
            pat, mat = parents_of(sim, iid)
            fl = flags_of(iid) | (qc[row_of[iid]] if ind.genotyped else set())
            nmiss = int((obs[row_of[iid]] < 0).sum()) if ind.genotyped else ''
            f.write("\t".join([
                names[iid], fam_of[iid], names[pat] if pat else '0', names[mat] if mat else '0',
                str(inds[canonical(sim, iid)].gen), inds[canonical(sim, iid)].tag,
                'Y' if ind.genotyped else 'N', names[ind.dup_of] if ind.dup_of else '',
                f"{F[canonical(sim, iid)]:.4f}", str(nmiss), ";".join(sorted(fl))]) + "\n")

    cons = set(sim.conserved_idx.tolist())
    with open(out("_truth_snps.tsv"), 'w') as f:
        f.write("CHROM\tPOS\tCM\tPOP0_FREQ\tPOP1_FREQ\tPOP2_FREQ\tBAD_MARKER\tCONSERVED_BLOCK\n")
        for j in range(genome.M):
            f.write(f"{genome.chr_names[genome.chrom_of[j]]}\t{genome.pos_bp[j]}\t"
                    f"{genome.pos_cM[j]:.3f}\t"
                    + "\t".join(f"{sim.pop_freq[p, j]:.3f}" for p in range(N_POPS))
                    + f"\t{'Y' if j in bad_set else 'N'}\t{'Y' if j in cons else 'N'}\n")

    lo, hi = sim.conserved_idx[0], sim.conserved_idx[-1]
    with open(out("_truth_conserved_regions.tsv"), 'w') as f:
        f.write("CHROM\tSTART\tEND\tSTART_CM\tEND_CM\tNOTE\n")
        f.write(f"{genome.chr_names[genome.chrom_of[lo]]}\t{genome.pos_bp[lo]}\t{genome.pos_bp[hi]}\t"
                f"{genome.pos_cM[lo]:.2f}\t{genome.pos_cM[hi]:.2f}\t"
                "near-fixed haplotype in all founders: population-wide hap-IBD hotspot\n")

    print(f"Simulated {len(vcf_order)} genotyped samples ({len(dup_pairs)} duplicate pairs, "
          f"{n_u} ungenotyped ancestors), {genome.M} SNPs on {genome.n_chr} chromosomes "
          f"(seed={args.seed})")
    print(f"{n_rel} related pairs in {os.path.basename(out('_truth_relationships.tsv'))}; "
          f"outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
