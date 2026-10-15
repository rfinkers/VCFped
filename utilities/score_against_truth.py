#!/usr/bin/env python3
"""
Score VCFped pipeline outputs against the ground truth of the simulated breeding
cohort (vcfped/testfiles/breeding_sim/, see its README.md).

Each input you pass switches on one section; give as many as you have:

  --pair FILE             step-1 .pair (sample NAMES; use *_named.pair)       duplicates, MZ/PO verdicts
  --purge-log FILE        purge_mz_twins.py stdout                            what the purge really removed
  --trio FILE             .trio, raw or filtered (names)                      trio recall / precision
  --fam FILE              reconstructed .fam (e.g. *_refined.fam)             final parent assignments
  --king FILE             PLINK2 .kin0                                        kinship bands vs true relationships
  --genotype-quality FILE check_genotype_quality.py <out>_genotype_quality.tsv    planted sample problems
  --mendel-individual FILE analyze_mendel_errors.py <out>_individual_analysis.tsv planted high-error samples
  --mendel-artifacts FILE  variant IDs flagged by the PLINK script's Mendel step   planted bad markers
  --ibd FILE (repeatable) hap-IBD .ibd.gz                                     IBD coverage vs true IBD
  --hbd FILE (repeatable) hap-IBD .hbd.gz                                     autozygosity vs true HBD

Truth is read from <truth>_truth_*.tsv/.fam/.ibd.gz written by generate_breeding_sim.py.

Scoring conventions (also in the cohort README):
  * Duplicates are interchangeable: every sample is mapped to its duplicate class and
    parents/children are compared as classes, never as individual clones.
  * Parent sets, not PAT/MAT slots, are compared (hermaphrodites: roles are arbitrary).
  * Denominators use the ORIGINAL sample set (--samples), so individuals lost to a wrong
    purge count as misses, reported separately as `lost_by_purge`.
  * Selfed children can never be vcfped trios; they are reported but excluded from recall.

Outputs: <out>_score_report.txt, <out>_score_children.tsv (one row per truth child class),
<out>_score_summary.json (machine-readable, for comparing runs/seeds).
"""

import argparse
import collections
import csv
import gzip
import json
import math
import os
import re
import sys

KING_BANDS = [(0.354, 'dup'), (0.177, '1st'), (0.0884, '2nd'), (0.0442, '3rd')]
TRUTH_BAND = {'duplicate': 'dup', 'parent_offspring': '1st', 'full_sib': '1st',
              'half_sib': '2nd', 'grandparent': '2nd', 'avuncular': '2nd', 'first_cousin': '3rd'}
TRIO_KINDS = ('simple', 'inbred_child', 'related_parents', 'backcross')   # two distinct genotyped parents
KINDS = list(TRIO_KINDS) + ['one_parent', 'selfed', 'no_parent']


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _open(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path)


def _tsv(path):
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def ratio(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return float('nan')
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100.0 * len(xs)))] if xs else float('nan')


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float('nan')
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def fmt_table(headers, rows):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    line = lambda cells: "  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\n".join([line(headers), line(['-' * w for w in widths])] + [line(r) for r in rows])


class Report:
    def __init__(self):
        self.lines = []
        self.summary = {}

    def h(self, title):
        self.lines += ["", title, "=" * len(title)]

    def p(self, text=""):
        self.lines.append(text)

    def text(self):
        return "\n".join(self.lines).lstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

class Truth:
    def __init__(self, prefix):
        self.prefix = prefix
        self.row = {r['ID']: r for r in _tsv(prefix + '_truth_samples.tsv') if r['GENOTYPED'] == 'Y'}
        self.canon = {s: (r['DUPLICATE_OF'] or s) for s, r in self.row.items()}
        self.flags = {s: set(r['FLAGS'].split(';')) - {''} for s, r in self.row.items()}
        self.f_ped = {s: float(r['F_PED']) for s, r in self.row.items()}
        self.classes = sorted({self.canon[s] for s in self.row})
        self.members = collections.defaultdict(set)
        for s in self.row:
            self.members[self.canon[s]].add(s)
        self.parents = {}
        with open(prefix + '_truth.fam') as f:
            for line in f:
                p = line.split()
                if len(p) >= 4:
                    self.parents[p[1]] = (p[2], p[3])
        # sample-level and class-level relationship lookup
        self.rel = {}
        self.rel_class = {}
        for r in _tsv(prefix + '_truth_relationships.tsv'):
            key = tuple(sorted((r['ID1'], r['ID2'])))
            self.rel[key] = (r['REL'], r['COMPLEX'])
            ckey = tuple(sorted((self.canon[r['ID1']], self.canon[r['ID2']])))
            self.rel_class.setdefault(ckey, r['REL'])
        self.dup_pairs = {tuple(sorted((r['ID1'], r['ID2']))) for r in _tsv(prefix + '_truth_duplicates.tsv')}
        self.snps = _tsv(prefix + '_truth_snps.tsv')
        self.conserved = _tsv(prefix + '_truth_conserved_regions.tsv')

    def cls(self, name):
        return self.canon.get(name, name)

    def child_items(self):
        """class -> dict(kind, parents=frozenset of parent classes, scenario)."""
        items = {}
        for c in self.classes:
            pat, mat = self.parents.get(c, ('0', '0'))
            ps = frozenset(self.canon[p] for p in (pat, mat) if p != '0' and p in self.canon)
            fl = self.flags[c]
            if len(ps) == 2:
                kind = ('backcross' if 'parent_x_own_offspring' in fl else
                        'related_parents' if 'parents_are_relatives' in fl else
                        'inbred_child' if 'inbred' in fl else 'simple')
            elif pat == mat != '0':
                kind = 'selfed'
            elif len(ps) == 1:
                kind = 'one_parent'
            else:
                kind = 'no_parent'
            items[c] = dict(kind=kind, parents=ps, scenario=self.row[c]['SCENARIO'])
        return items

    def pair_rel(self, a, b):
        return self.rel.get(tuple(sorted((a, b))), ('unrelated', ''))[0]


def load_samples(arg, truth):
    """Sample universe from a VCF header or a one-name-per-line file; default: all genotyped."""
    if not arg:
        return set(truth.row)
    names = []
    with _open(arg) as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                names = line.rstrip('\n').split('\t')[9:]
                break
            if line.strip() and not line.startswith('#'):
                names.append(line.split()[0])
    unknown = [n for n in names if n not in truth.row]
    if unknown:
        print(f"WARNING: {len(unknown)} sample(s) in {arg} are not in the truth (e.g. {unknown[:3]}); ignored.",
              file=sys.stderr)
    return {n for n in names if n in truth.row}


class Resolver:
    """Maps vcfped's 1-based sample numbers to names when a .log is given; names pass through."""

    def __init__(self, log=None):
        self.map = {}
        if log:
            with open(log) as f:
                for line in f:
                    m = re.match(r'^\s*(\d+):\s*(\S+)\s*$', line)
                    if m:
                        self.map[m.group(1)] = m.group(2)

    def __call__(self, token):
        token = token.strip()
        if token.isdigit():
            if not self.map:
                sys.exit("Error: input uses sample NUMBERS; pass --log <vcfped .log> or the *_named file.")
            return self.map[token]
        return token


# ---------------------------------------------------------------------------
# 1 & 2. Duplicates, MZ purge, pairwise verdicts
# ---------------------------------------------------------------------------

def read_pairs(path, resolve):
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            a, b = (resolve(x) for x in row['Pair'].split(','))
            out.setdefault(tuple(sorted((a, b))), row['Verdict'])
    return out


def score_pairs(rep, truth, present, pairs):
    rep.h("2. vcfped pairwise verdicts")
    both = lambda k: k[0] in present and k[1] in present
    verdict = {k: v for k, v in pairs.items() if both(k)}
    inbred = lambda k: sum(truth.f_ped[s] > 0.9 for s in k)
    summ = {}

    true_dups = {k for k in truth.dup_pairs if both(k)}
    mz = {k for k, v in verdict.items() if v == 'MZ twins'}
    tp = len(true_dups & mz)
    fp = mz - true_dups
    fp_by_rel = collections.Counter(truth.pair_rel(*k) for k in fp)
    fp_inbred = collections.Counter(inbred(k) for k in fp)
    rep.p(f"MZ twins: true duplicate pairs called  {ratio(tp, len(true_dups))}")
    rep.p(f"         non-duplicate pairs called    {len(fp)}   "
          f"(both members F>0.9: {fp_inbred[2]}, one: {fp_inbred[1]}, neither: {fp_inbred[0]})")
    rep.p("         non-duplicate calls by true relationship: "
          + (", ".join(f"{k} {v}" for k, v in fp_by_rel.most_common()) or "none"))
    summ['mz'] = dict(true_pairs=len(true_dups), found=tp, false_calls=len(fp),
                      false_both_inbred=fp_inbred[2], false_one_inbred=fp_inbred[1])

    po_rows = [(k, truth.rel[k]) for k in truth.rel if both(k) and truth.rel[k][0] == 'parent_offspring']
    po = {k for k, v in verdict.items() if v == 'Parent-child'}
    for cflag, label, key in (('N', 'simple', 'po_simple'), ('Y', 'inbred/multi-path', 'po_inbred')):
        keys = {k for k, v in po_rows if v[1] == cflag}
        rep.p(f"Parent-child: true {label} PO pairs found  {ratio(len(keys & po), len(keys))}")
        summ[key] = dict(true_pairs=len(keys), found=len(keys & po))
    true_po = {k for k, _ in po_rows}
    fp_po = collections.Counter(truth.pair_rel(*k) for k in po - true_po)
    rep.p(f"              calls that are not parent-offspring: {sum(fp_po.values())}   "
          + ", ".join(f"{k} {v}" for k, v in fp_po.most_common()))
    fs_simple = {k for k, v in truth.rel.items() if both(k) and v == ('full_sib', 'N')}
    rep.p(f"              simple full-sib pairs mistaken for parent-child  {ratio(len(fs_simple & po), len(fs_simple))}")
    summ['po_false_calls'] = sum(fp_po.values())
    rep.summary['pairs'] = summ


def score_purge(rep, truth, present, log_path):
    """Parse purge_mz_twins output ('keep=A, remove=B, C'); return the set of removed samples."""
    rep.h("1. Duplicate purge (what purge_mz_twins actually removed)")
    removed, clusters = set(), []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'keep=(\S+?),\s*remove=(.+?)\s*$', line)
            if m:
                keep = m.group(1)
                rem = [x.strip() for x in m.group(2).split(',') if x.strip()]
                clusters.append((keep, rem))
                removed.update(rem)
    wrong, correct_rm = [], 0
    for keep, rem in clusters:
        for r in rem:
            if truth.cls(r) == truth.cls(keep):
                correct_rm += 1
            else:
                wrong.append((r, keep))
    kept = {s for s in present if s not in removed}
    lost = {truth.cls(r) for r, _ in wrong} - {truth.cls(s) for s in kept}
    survivors = sum(1 for n in collections.Counter(truth.cls(s) for s in kept).values() if n > 1)
    mixed = sum(1 for keep, rem in clusters if any(truth.cls(r) != truth.cls(keep) for r in rem))
    rep.p(f"clusters {len(clusters)} (mixed, i.e. merging distinct individuals: {mixed}); "
          f"samples removed {len(removed)}")
    rep.p(f"  removals that were true duplicates   {ratio(correct_rm, len(removed))}")
    rep.p(f"  removals of DISTINCT individuals     {len(wrong)}   -> individuals lost from the trio stage: {len(lost)}")
    rep.p(f"  duplicate classes still having >1 sample after purge: {survivors}")
    rep.summary['purge'] = dict(clusters=len(clusters), removed=len(removed), true_removals=correct_rm,
                                wrong_removals=len(wrong), individuals_lost=len(lost),
                                duplicate_classes_surviving=survivors)
    return removed, lost


# ---------------------------------------------------------------------------
# 3 & 4. Trios and final pedigree
# ---------------------------------------------------------------------------

def read_trios(path, resolve):
    calls = collections.defaultdict(lambda: collections.defaultdict(set))   # child -> verdict -> {parent sets}
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            members = [resolve(x) for x in row['Triple'].split(',')]
            child = resolve(row['Pivot'])
            calls[child][row['Verdict']].add(frozenset(m for m in members if m != child))
    return calls


def read_fam(path):
    fam = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                fam[p[1]] = (p[2], p[3])
    return fam


def _outcome_table(rep, per_kind, outcomes, title):
    rows = []
    for kind in KINDS:
        c = per_kind.get(kind)
        if c:
            rows.append([kind, sum(c.values())] + [c.get(o, 0) for o in outcomes])
    rep.p(title)
    rep.p(fmt_table(['truth kind', 'n'] + outcomes, rows))


def score_trios(rep, truth, present, lost_classes, calls, children_rows):
    rep.h("3. Trio stage (vcfped .trio)")
    items = truth.child_items()
    present_cls = {truth.cls(s) for s in present}
    outcomes = ['regular_unique', 'regular_ambiguous', 'wrong_regular', 'inverted_only', 'missed',
                'lost_by_purge', 'not_in_input', 'spurious_regular', 'correctly_empty']
    per_kind = collections.defaultdict(collections.Counter)
    by_cls_calls = collections.defaultdict(lambda: collections.defaultdict(set))
    for child, verdicts in calls.items():
        for verdict, psets in verdicts.items():
            for ps in psets:
                by_cls_calls[truth.cls(child)][verdict].add(frozenset(truth.cls(p) for p in ps))
    regular_rows = regular_right = 0
    for c in truth.classes:
        if c not in present_cls:
            continue
        it = items[c]
        v = by_cls_calls.get(c, {})
        regular = set(v.get('Regular trio', ()))
        inverted = set(v.get('Inverted/generational trio', ()))
        regular_rows += len(regular)
        regular_right += sum(1 for r in regular if r == it['parents'] and len(r) == 2)
        if it['kind'] in TRIO_KINDS:
            if not all(p in present_cls for p in it['parents']):
                out = 'not_in_input'                              # a parent is not in the analysed samples
            elif c in lost_classes or any(p in lost_classes for p in it['parents']):
                out = 'lost_by_purge'
            elif it['parents'] in regular:
                out = 'regular_unique' if len(regular) == 1 else 'regular_ambiguous'
            elif regular:
                out = 'wrong_regular'
            elif it['parents'] in inverted:
                out = 'inverted_only'
            else:
                out = 'missed'
        else:                                                     # no genotyped parent pair to find
            out = 'spurious_regular' if regular else 'correctly_empty'
        per_kind[it['kind']][out] += 1
        children_rows[c]['KIND'] = it['kind']
        children_rows[c]['TRIO_OUTCOME'] = out
        children_rows[c]['TRIO_REGULAR_CALLS'] = ';'.join(sorted('+'.join(sorted(r)) for r in regular))
    _outcome_table(rep, per_kind, outcomes, "Outcome per truth child class (clone-equivalent parents accepted):")
    summ = {}
    for kind in TRIO_KINDS:
        c = per_kind.get(kind, collections.Counter())
        n = sum(c.values()) - c['not_in_input']
        found = c['regular_unique'] + c['regular_ambiguous']
        rep.p(f"  {kind:16s} recall (true pair among Regular calls)  {ratio(found, n)}   "
              f"unique {c['regular_unique']}, ambiguous {c['regular_ambiguous']}, "
              f"inverted-only {c['inverted_only']}")
        summ[kind] = dict(n=n, found=found, unique=c['regular_unique'], inverted_only=c['inverted_only'],
                          wrong_regular=c['wrong_regular'], missed=c['missed'], lost_by_purge=c['lost_by_purge'])
    rep.p(f"  precision: Regular (child, parents) calls that are the true pair  {ratio(regular_right, regular_rows)}")
    summ['no_pair_to_find'] = {k: dict(per_kind[k]) for k in ('one_parent', 'selfed', 'no_parent') if k in per_kind}
    summ['regular_calls'] = regular_rows
    summ['regular_calls_correct'] = regular_right
    rep.summary['trios'] = summ


def score_fam(rep, truth, present, fam, children_rows):
    rep.h("4. Final pedigree (.fam parent assignments)")
    items = truth.child_items()
    present_cls = {truth.cls(s) for s in present}
    outcomes = ['correct', 'partial', 'wrong', 'unassigned', 'not_in_input', 'not_representable',
                'correct_empty', 'spurious']
    per_kind = collections.defaultdict(collections.Counter)
    per_scen = collections.defaultdict(lambda: [0, 0])
    wrong_rel = collections.Counter()
    for c in truth.classes:
        if c not in present_cls:
            continue
        it = items[c]
        assigned = frozenset()
        for m in sorted(truth.members[c] & present):
            pat, mat = fam.get(m, ('0', '0'))
            s = frozenset(truth.cls(p) for p in (pat, mat) if p != '0')
            if s:
                assigned = s
                break
        tp = it['parents']
        if it['kind'] in TRIO_KINDS and not all(p in present_cls for p in tp):
            out = 'not_in_input'
        elif not tp:                                              # no genotyped parent: nothing to assign
            out = 'correct_empty' if not assigned else 'spurious'
        elif it['kind'] in ('one_parent', 'selfed'):             # a parent *pair* cannot be right
            out = ('not_representable' if not assigned else 'correct' if assigned == tp else
                   'partial' if assigned & tp else 'spurious')
        elif not assigned:
            out = 'unassigned'
        elif assigned == tp:
            out = 'correct'
        elif assigned & tp:
            out = 'partial'
        else:
            out = 'wrong'
        for a in assigned - tp:
            wrong_rel[truth.rel_class.get(tuple(sorted((a, c))), 'unrelated')] += 1
        per_kind[it['kind']][out] += 1
        if it['kind'] in TRIO_KINDS and out != 'not_in_input':
            per_scen[it['scenario']][1] += 1
            per_scen[it['scenario']][0] += out == 'correct'
        children_rows[c].update(KIND=it['kind'], SCENARIO=it['scenario'],
                                TRUTH_PARENTS=';'.join(sorted(tp)), FAM_OUTCOME=out,
                                FAM_ASSIGNED=';'.join(sorted(assigned)))
    _outcome_table(rep, per_kind, outcomes, "Outcome per truth child class:")
    summ = {}
    n_assigned = n_correct = 0
    for kind, c in per_kind.items():
        n_assigned += sum(c[o] for o in ('correct', 'partial', 'wrong', 'spurious'))
        n_correct += c['correct']
    for kind in TRIO_KINDS:
        c = per_kind.get(kind, collections.Counter())
        n = sum(c.values()) - c['not_in_input']
        rep.p(f"  {kind:16s} parents fully correct  {ratio(c['correct'], n)}")
        summ[kind] = dict(n=n, **{o: c[o] for o in outcomes})
    rep.p(f"  precision: assigned children whose parent set is exactly right  {ratio(n_correct, n_assigned)}")
    if wrong_rel:
        rep.p("  wrongly assigned parents are truly: "
              + ", ".join(f"{k} {v}" for k, v in wrong_rel.most_common()))
    rep.p("  fully-correct share by scenario (children with two genotyped parents):")
    rep.p(fmt_table(['scenario', 'correct/n'], [[s, f"{a}/{b}"] for s, (a, b) in sorted(per_scen.items())]))
    summ['assigned'] = n_assigned
    summ['assigned_correct'] = n_correct
    summ['wrongly_assigned_parent_is'] = dict(wrong_rel)
    rep.summary['fam'] = summ


# ---------------------------------------------------------------------------
# 5. KING
# ---------------------------------------------------------------------------

def kin_band(k):
    for thr, name in KING_BANDS:
        if k >= thr:
            return name
    return '<3rd'


def read_king(path):
    out = {}
    with open(path) as f:
        header = f.readline().lstrip('#').split()
        col = {h: i for i, h in enumerate(header)}
        i1 = col.get('IID1', col.get('ID1'))
        i2 = col.get('IID2', col.get('ID2'))
        for line in f:
            p = line.split()
            ibs0 = float(p[col['IBS0']]) if 'IBS0' in col else float('nan')
            out[tuple(sorted((p[i1], p[i2])))] = (float(p[col['KINSHIP']]), ibs0)
    return out


def score_king(rep, truth, present, king):
    rep.h("5. KING kinship vs true relationships")
    both = lambda k: k[0] in present and k[1] in present
    king = {k: v for k, v in king.items() if both(k)}
    rows, summ = [], {}
    for rel in ('duplicate', 'parent_offspring', 'full_sib', 'half_sib', 'grandparent', 'avuncular', 'first_cousin'):
        for cflag in ('N', 'Y'):
            keys = [k for k, v in truth.rel.items() if both(k) and v == (rel, cflag)]
            if not keys:
                continue
            listed = [k for k in keys if k in king]
            exact = sum(1 for k in listed if kin_band(king[k][0]) == TRUTH_BAND[rel]) if cflag == 'N' else None
            rows.append([rel, 'simple' if cflag == 'N' else 'inbred', len(keys),
                         f"{100.0 * len(listed) / len(keys):.0f}%",
                         '' if exact is None else f"{100.0 * exact / len(keys):.0f}%",
                         f"{median([king[k][0] for k in listed]):.3f}" if listed else '-'])
            summ[f"{rel}_{'simple' if cflag == 'N' else 'inbred'}"] = dict(n=len(keys), listed=len(listed), exact_band=exact)
    rep.p(fmt_table(['true relationship', 'type', 'pairs', 'listed (>=0.0442)', 'in correct KING band', 'median KING'], rows))
    bands = collections.Counter(kin_band(v[0]) for k, v in king.items() if k not in truth.rel)
    rep.p(f"KING pairs >= 0.0442 absent from the truth table (true kinship below 0.0442, or none): {sum(bands.values())}   "
          + ", ".join(f"{b} {n}" for b, n in sorted(bands.items())))
    summ['false_relatives'] = dict(bands)
    for rel in ('parent_offspring', 'full_sib'):
        ibs0 = [king[k][1] for k, v in truth.rel.items() if both(k) and v == (rel, 'N') and k in king]
        if ibs0:
            rep.p(f"IBS0 of simple {rel:16s} 5th/50th/95th pct: {percentile(ibs0, 5):.4f} / "
                  f"{percentile(ibs0, 50):.4f} / {percentile(ibs0, 95):.4f}")
            summ[f'ibs0_{rel}'] = [percentile(ibs0, p) for p in (5, 50, 95)]
    rep.summary['king'] = summ


# ---------------------------------------------------------------------------
# 6. QC plants
# ---------------------------------------------------------------------------

def score_quality(rep, truth, present, gq_path, mendel_ind_path, artifact_path):
    rep.h("6. Planted quality problems")
    summ = {}
    planted = lambda tag: {s for s in present if tag in truth.flags[s]}
    if gq_path:
        rows = {r['ID']: r for r in _tsv(gq_path)}
        miss_flag = {s for s, r in rows.items() if r['Miss_Flagged'].upper() == 'YES'}
        het_flag = {s for s, r in rows.items() if r['Het_Flagged'].upper() == 'YES'}
        low, con = planted('low_callrate') & set(rows), planted('contaminated') & set(rows)
        rep.p(f"check_genotype_quality: {len(rows)} samples, flagged for missingness {len(miss_flag)}, "
              f"for heterozygosity {len(het_flag)}  (rank-based, so ~5% are flagged by design)")
        rep.p(f"  planted low call rate flagged for missingness     {ratio(len(low & miss_flag), len(low))}")
        rep.p(f"  planted contaminated flagged for heterozygosity   {ratio(len(con & het_flag), len(con))}")
        summ['genotype_quality'] = dict(low_callrate=[len(low & miss_flag), len(low)],
                                        contaminated=[len(con & het_flag), len(con)])
    if mendel_ind_path:
        rows = {r['Individual']: r for r in _tsv(mendel_ind_path)}
        flagged = {s for s, r in rows.items() if r['Flagged'].strip().lower() == 'true'}
        err = planted('high_error')
        elig = err & set(rows)
        rep.p(f"analyze_mendel_errors: {len(rows)} parents evaluated, {len(flagged)} flagged as repeat offenders")
        rep.p(f"  planted high-error samples evaluated as parents {len(elig)}/{len(err)}; "
              f"flagged {ratio(len(elig & flagged), len(elig))}")
        summ['mendel_individual'] = dict(evaluated=len(elig), flagged=len(elig & flagged), planted=len(err))
    if artifact_path:
        ids = [l.strip() for l in open(artifact_path) if l.strip()]
        if not ids:
            rep.p("Mendel artifact list is empty: no marker scoring (was --mendel available in this plink2?).")
        else:
            norm = lambda chrom: re.sub(r'^(?:ch|chr)0*', '', chrom, flags=re.I)
            called = {(norm(i.split(':')[0]), int(i.split(':')[1])) for i in ids if i.count(':') >= 1}
            bad = {(norm(r['CHROM']), int(r['POS'])) for r in truth.snps if r['BAD_MARKER'] == 'Y'}
            rep.p(f"Mendel artifact markers: planted bad markers found {ratio(len(called & bad), len(bad))}; "
                  f"other markers flagged {len(called - bad)}")
            summ['bad_markers'] = dict(found=len(called & bad), planted=len(bad), extra=len(called - bad))
    rep.summary['quality'] = summ


# ---------------------------------------------------------------------------
# 7. IBD / HBD
# ---------------------------------------------------------------------------

def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_bp(a, b):
    i = j = tot = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def read_segments(path, min_cm, hbd=False):
    """key -> chrom -> merged [[start, end]]; key = sorted sample pair (IBD) or sample (HBD)."""
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    with _open(path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 8:
                continue
            a, b, chrom = p[0], p[2], p[4]
            if (a == b) != hbd:
                continue
            try:
                s, e, cm = int(p[5]), int(p[6]), float(p[7])
            except ValueError:
                continue
            if cm < min_cm:
                continue
            raw[a if hbd else tuple(sorted((a, b)))][chrom].append((s, e))
    return {k: {c: merge(v) for c, v in d.items()} for k, d in raw.items()}


def total_bp(d):
    return sum(e - s for ivs in d.values() for s, e in ivs)


def overlap_dicts(a, b):
    return sum(overlap_bp(a[c], b[c]) for c in a if c in b)


def bp_outside(called, region):
    """Called bp that lies outside the planted conserved window(s)."""
    tot = total_bp(called)
    for chrom, lo, hi in region:
        if chrom in called:
            tot -= overlap_bp(called[chrom], [[lo, hi]])
    return tot


def score_ibd(rep, truth, present, ibd_paths, hbd_paths, min_cm):
    rep.h("7. IBD / HBD vs truth")
    both = lambda k: k[0] in present and k[1] in present
    region = [(r['CHROM'], int(r['START']), int(r['END'])) for r in truth.conserved]
    n_pairs = len(present) * (len(present) - 1) // 2
    for path in ibd_paths:
        rep.p(f"-- {os.path.basename(path)}")
        truth_ibd = {k: v for k, v in read_segments(truth.prefix + '_truth_ibd.ibd.gz', 0).items() if both(k)}
        called = {k: v for k, v in read_segments(path, min_cm).items() if both(k)}
        by_class = collections.defaultdict(lambda: [0, 0, []])      # pooled overlap, pooled truth, per-pair recalls
        for k, tv in truth_ibd.items():
            rel, cflag = truth.rel.get(k, ('distant (not in relationships table)', ''))
            label = rel + ('*' if cflag == 'Y' else '')
            tb = total_bp(tv)
            ob = overlap_dicts(tv, called.get(k, {}))
            by_class[label][0] += ob
            by_class[label][1] += tb
            by_class[label][2].append(ob / tb if tb else 0.0)
        rows = [[lab, len(v[2]), f"{100.0 * v[0] / v[1]:.1f}%", f"{100.0 * median(v[2]):.1f}%",
                 f"{100.0 * sum(r >= 0.8 for r in v[2]) / len(v[2]):.0f}%"] for lab, v in sorted(by_class.items())]
        rep.p("Coverage of true IBD by called segments (called >= %.1f cM; * = inbred/multi-path pairs):" % min_cm)
        rep.p(fmt_table(['true relationship', 'pairs', 'pooled bp recall', 'median per-pair', 'pairs >=80%'], rows))
        fp_pairs = [k for k in called if k not in truth_ibd]
        fp_out = sum(bp_outside(called[k], region) for k in fp_pairs)
        in_region = sum(1 for d in called.values()
                        if any(chrom in d and overlap_bp(d[chrom], [[lo, hi]]) > 0 for chrom, lo, hi in region))
        rep.p(f"Pairs with called IBD but no true IBD: {len(fp_pairs)}   "
              f"{fp_out / 1e6:.0f} Mb outside the planted conserved block")
        rep.p(f"Pairs with a called segment in the planted conserved block: {ratio(in_region, n_pairs)}")
        rep.summary.setdefault('ibd', {})[os.path.basename(path)] = dict(
            recall={lab: v[0] / v[1] for lab, v in by_class.items()},
            false_pairs=len(fp_pairs), false_mb_outside_block=fp_out / 1e6, pairs_in_conserved_block=in_region)
    for path in hbd_paths:
        rep.p(f"-- {os.path.basename(path)}")
        truth_hbd = {k: v for k, v in read_segments(truth.prefix + '_truth_hbd.hbd.gz', 0, hbd=True).items() if k in present}
        called = {k: v for k, v in read_segments(path, min_cm, hbd=True).items() if k in present}
        samples = sorted(present)
        t = [total_bp(truth_hbd.get(s, {})) / 1e6 for s in samples]
        c = [total_bp(called.get(s, {})) / 1e6 for s in samples]
        auto = [s for s in samples if s in truth_hbd]
        rec = [min(1.0, overlap_dicts(truth_hbd[s], called.get(s, {})) / total_bp(truth_hbd[s])) for s in auto]
        spurious = sum(1 for s, ci in zip(samples, c) if s not in truth_hbd and ci > 0)
        rep.p(f"per-sample total HBD (Mb): Pearson r called vs true = {pearson(t, c):.3f} over {len(samples)} samples")
        rep.p(f"samples with true HBD: {len(auto)}, median fraction of it recovered {100.0 * median(rec):.1f}%; "
              f"samples with called HBD but none true: {spurious}")
        rep.summary.setdefault('hbd', {})[os.path.basename(path)] = dict(
            pearson=pearson(t, c), median_recovered=median(rec), spurious_samples=spurious)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score pipeline outputs against the breeding_sim ground truth",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("Scoring conventions")[0])
    ap.add_argument('--truth', required=True, help="truth prefix, e.g. vcfped/testfiles/breeding_sim/breeding_sim")
    ap.add_argument('--samples', help="VCF (or name list) the pipeline was run on (default: all genotyped samples)")
    ap.add_argument('--log', help="vcfped .log, only if --pair/--trio use sample numbers instead of names")
    ap.add_argument('--pair')
    ap.add_argument('--purge-log')
    ap.add_argument('--trio')
    ap.add_argument('--fam')
    ap.add_argument('--king')
    ap.add_argument('--genotype-quality')
    ap.add_argument('--mendel-individual')
    ap.add_argument('--mendel-artifacts')
    ap.add_argument('--ibd', action='append', default=[])
    ap.add_argument('--hbd', action='append', default=[])
    ap.add_argument('--min-cm', type=float, default=2.0, help="shortest called IBD/HBD segment to score (default 2.0)")
    ap.add_argument('-o', '--output', default='score', help="output prefix (default: score)")
    args = ap.parse_args()

    given = [args.pair, args.purge_log, args.trio, args.fam, args.king, args.genotype_quality,
             args.mendel_individual, args.mendel_artifacts, args.ibd, args.hbd]
    if not any(given):
        ap.error("give at least one input to score (--pair, --trio, --fam, --king, --ibd, ...)")

    truth = Truth(args.truth)
    present = load_samples(args.samples, truth)
    resolve = Resolver(args.log)
    rep = Report()
    rep.p(f"Truth: {args.truth}   samples scored: {len(present)} of {len(truth.row)} genotyped "
          f"({len({truth.cls(s) for s in present})} duplicate classes)")
    children_rows = collections.defaultdict(dict)

    removed, lost = set(), set()
    if args.purge_log:
        removed, lost = score_purge(rep, truth, present, args.purge_log)
    if args.pair:
        score_pairs(rep, truth, present, read_pairs(args.pair, resolve))
    if args.trio:
        score_trios(rep, truth, present, lost, read_trios(args.trio, resolve), children_rows)
    if args.fam:
        score_fam(rep, truth, present, read_fam(args.fam), children_rows)
    if args.king:
        score_king(rep, truth, present, read_king(args.king))
    if args.genotype_quality or args.mendel_individual or args.mendel_artifacts:
        score_quality(rep, truth, present, args.genotype_quality, args.mendel_individual, args.mendel_artifacts)
    if args.ibd or args.hbd:
        score_ibd(rep, truth, present, args.ibd, args.hbd, args.min_cm)

    text = rep.text()
    print(text)
    with open(args.output + '_score_report.txt', 'w') as f:
        f.write(text)
    with open(args.output + '_score_summary.json', 'w') as f:
        json.dump(rep.summary, f, indent=2, sort_keys=True, default=str)
    if children_rows:
        cols = ['CLASS', 'KIND', 'SCENARIO', 'TRUTH_PARENTS', 'TRIO_OUTCOME', 'TRIO_REGULAR_CALLS',
                'FAM_OUTCOME', 'FAM_ASSIGNED']
        with open(args.output + '_score_children.tsv', 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for c in sorted(children_rows):
                row = children_rows[c]
                f.write('\t'.join([c] + [str(row.get(k, '')) for k in cols[1:]]) + '\n')
    print(f"Written: {args.output}_score_report.txt, {args.output}_score_summary.json"
          + (f", {args.output}_score_children.tsv" if children_rows else ""))


if __name__ == '__main__':
    main()
