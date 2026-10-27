#!/usr/bin/env python

"""
Optimized Module for identifying trios and close pairwise relationships in a multisample VCF file.

Major optimizations, aimed at large datasets (1000s of markers x 1000s of individuals):
1. Shared memory arrays (multiprocessing.shared_memory) to avoid copying large matrices
2. Dynamic load balancing across all available CPU cores
3. Memory-efficient data structures with explicit dtype (int8 genotype matrix, -1 for missing)
4. Batch processing for massive trio analysis
5. Numba JIT compilation for vectorized score calculations
6. Efficient file I/O with buffering, including gzip (.vcf.gz) support
7. Smart memory management and garbage collection
8. Progress tracking and performance monitoring
9. Online best-tracking: Only keeps best result per pair/trio (massive memory savings!)
10. Dynamic work distribution: adaptive batch sizes based on workload and core count

Parallelism Strategy:
- Pure PROCESS-level parallelism (multiprocessing.Pool, one worker per core)
- Each worker runs single-threaded Numba code; thread env vars are forced to 1
  to avoid oversubscription from numpy/numba's own internal threading
- Batch sizes are adaptive based on workload and number of jobs

Memory Optimization:
- Online best-tracking: Updates best results on-the-fly instead of storing all results
- For N samples: Stores ~N(N-1)/2 pair results and ~N(N-1)(N-2)/6 trio results
  (one "best so far" entry per pair/trio, instead of one per percentile)
- Uses dictionaries during processing, converts to lists only for output

Missing genotypes:
- Variants are NOT dropped just because some sample has a missing genotype
  (this would discard nearly all variants in large cohorts). Instead, missing
  genotypes are coded as -1 in the genotype matrix and excluded only from the
  per-pair / per-trio counts that involve them.

Author: Magnus Dehli Vigeland (original)
Optimized for large-scale analysis (shared memory + batching + JIT + online best-tracking)
"""

__version__ = "1.5.0-optimized"
__author__ = "Magnus Dehli Vigeland"

import collections
import operator
import gzip
import time
import itertools
import random
import sys
import os
import argparse
from tabulate import tabulate

# CRITICAL: Disable threading to prevent oversubscription
# Must be set BEFORE importing numba and numpy
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import multiprocessing as mp
from multiprocessing import Pool, cpu_count, shared_memory
import numpy as np
from functools import lru_cache, partial
import gc
import sqlite3
import tempfile
try:
    from . import checkped
except ImportError:
    import checkped

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, *args):
            pass
        def close(self):
            pass

# Hard coding coordinates of pseudo-autosomal regions on X (hg19)
_PAR1_ = (60000, 2699521)
_PAR2_ = (154931043, 155270561)

# Genotype mapping
_GT2AB_ = {'0/0':'AA', '0|0':'AA', '0/1':'AB', '0|1':'AB', '1|0':'AB', '1/1':'BB', '1|1':'BB'}
_GT2NUM_ = {'AA': 0, 'AB': 1, 'BB': 2, '-': -1}
_NUM2GT_ = {0: 'AA', 1: 'AB', 2: 'BB', -1: '-'}

def XminusPAR(linesplit, chromcol):
    ''' Determine if a variant is on X outside PARs.'''
    if 'X' not in linesplit[chromcol]:
        return False
    pos = float(linesplit[chromcol+1])
    par = _PAR1_[0] < pos < _PAR1_[1] or _PAR2_[0] < pos < _PAR2_[1]
    return not par

def _writeout(txt, log, quiet=False):
    if log:
        print(txt, file=log)
    if not quiet:
        print(txt)

def _default_n_jobs(n_jobs, total_cores):
    """Resolve the worker pool size.

    An explicit -j/--jobs always wins. Otherwise, on machines with more than
    8 CPUs, cap the default at 25% of the core count rather than claiming
    every core — avoids oversubscribing shared/multi-tenant hosts when no
    thread budget was requested.
    """
    if n_jobs is not None:
        return max(1, n_jobs)
    if total_cores > 8:
        return max(1, round(total_cores * 0.25))
    return total_cores

def _smart_open(filename, mode='r'):
    """Open plain or gzip-compressed files transparently."""
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    return open(filename, mode)

def inspectFile(filename, log, quiet=False):
    """Unchanged from original - file inspection is not a bottleneck"""
    _writeout("\n====FILE INFO====", log, quiet)
    _writeout("File path: " + os.path.abspath(filename), log, quiet)

    with _smart_open(filename) as f:
        preamble_firstline = f.readline().rstrip()
        line = preamble_firstline
        while line.startswith("##"): 
            line = f.readline()
        
        headers = line.rstrip().split('\t')
        startpos = f.tell()
        firstvar = f.readline().rstrip().split('\t')
        
        if headers[0] == "#CHROM" and headers[5] == "QUAL" and headers[8] == 'FORMAT' and firstvar[8].startswith("GT"):
            if 'fileformat=' in preamble_firstline:
                fileformat = preamble_firstline[preamble_firstline.index('=')+1:]
            else: 
                fileformat = "Not stated in the first line, but looks like VCF format"
            chromcol, qualcol, formatcol = 0,5,8
        else:
            try:
                chromcol = next(i for i,h in enumerate(headers) if h.lower() in ("chr", "chrom", "chromosome"))
                formatcol = next(i for i,field in enumerate(firstvar) if field.startswith("GT:") or field=="GT")
            except:
                raise RuntimeError("Could not identify chromosome and/or format columns")
            try:
                qualcol = next(i for i,field in enumerate(headers) if field.lower() in ['qual', 'vcf_qual'])
                float(firstvar[qualcol])
            except:
                qualcol = None
            fileformat = "Non-standard VCF"
        
        _writeout("File format: " + fileformat, log, quiet)
        _writeout("Chromosome column [index]: %s [%d]" %(headers[chromcol], chromcol), log, quiet)
        _writeout("QUAL column [index]: %s [%d]" %(headers[qualcol], qualcol), log, quiet)
        _writeout("FORMAT column [index]: %s [%d]" %(headers[formatcol], formatcol), log, quiet)

        format = firstvar[formatcol]
        formatfields = format.split(':')
        ADind = formatfields.index('AD') if 'AD' in formatfields else None
        DPind = formatfields.index('DP') if 'DP' in formatfields else None
        GQind = formatfields.index('GQ') if 'GQ' in formatfields else None
        _writeout("Format fields: " + ", ".join(formatfields), log, quiet)
        
        available_vars = [v for v,ind in zip(['QUAL','DP','GQ','AD'], [qualcol, DPind, GQind, ADind]) if ind is not None]
        _writeout("Available filtering variables: " + ", ".join(available_vars), log, quiet)
        
        samples = headers[(formatcol + 1):]
        nSamples = len(samples)
        
        _writeout("Number of samples: %d" % nSamples, log, quiet)

        # Always write ALL sample names to log file (for reconstruction)
        if log:
            _writeout("Sample names:", log, quiet=True)
            for i, s in enumerate(samples):
                _writeout('  %d: %s' % (i+1, s), log, quiet=True)

        # Show abbreviated list on screen if many samples
        if not quiet:
            if nSamples <= 20:
                _writeout("Sample names:", log=None, quiet=False)
                for i, s in enumerate(samples):
                    _writeout('  %d: %s' % (i+1, s), log=None, quiet=False)
            else:
                _writeout("Sample names (first 10):", log=None, quiet=False)
                for i, s in enumerate(samples[:10]):
                    _writeout('  %d: %s' % (i+1, s), log=None, quiet=False)
                _writeout(f'  ... and {nSamples-10} more (full list in log file)', log=None, quiet=False)
        
        linecount = None
        skipsize = 2000
        for i in range(skipsize-1): 
            if f.readline() == '':
                linecount = i
                break
            
        if linecount is None:
            if filename.endswith('.gz'):
                # For gzipped files, count remaining lines (seek-based estimation is unreliable)
                linecount = skipsize + sum(1 for _ in f)
                counttype = 'Exact'
                endpos = None
            else:
                after_skip = f.tell()
                f.seek(0, 2)
                endpos = float(f.tell())
                linecount = int((endpos-startpos)/(after_skip-startpos)*skipsize)
                counttype = 'Approximate'
        else:
            counttype = "Exact"
            endpos = None
        _writeout("%s variant count: %d" %(counttype,linecount), log, quiet)
        
    return dict(filename=filename, fileformat=fileformat, chromcol=chromcol, formatcol=formatcol, qualcol=qualcol, 
                formatfields=formatfields, DPind=DPind, GQind=GQind, ADind=ADind, counttype=counttype, available_vars=available_vars,
                linecount=linecount, samples=samples, nSamples=nSamples, startpos=startpos, endpos=endpos)

def getGTtuple(linesplit, firstSampleCol):
    """Mapping VCF genotypes to {AA, AB, BB}, anything else to '-'"""
    return tuple(_GT2AB_.get(gt[:3], '-') for gt in linesplit[firstSampleCol:])

def getGTtuple_numeric(linesplit, firstSampleCol):
    """Convert genotypes to numeric codes for faster processing"""
    return tuple(_GT2NUM_.get(_GT2AB_.get(gt[:3], '-'), -1) for gt in linesplit[firstSampleCol:])

class Filter(object):
    """Unchanged from original"""
    def __init__(self, formatinfo, PASS=None, DPmin=None, GQmin=None, QUALmin=None, ADmin=None):
        self.nSamples=formatinfo['nSamples']
        self.qualcol=formatinfo['qualcol']
        self.ADind=formatinfo['ADind']
        self.DPind=formatinfo['DPind']
        self.GQind=formatinfo['GQind']
        self.PASS=PASS
        self.DPmin=DPmin
        self.GQmin=GQmin
        self.QUALmin=QUALmin
        self.ADmin=ADmin
        
    def __str__(self):
        pss =  'PASS: ' + str(self.PASS)
        qual = 'QUAL >= %d' %self.QUALmin if self.QUALmin is not None else 'QUAL: NA'
        dp = 'DP >= %d' %self.DPmin if self.DPmin is not None else 'DP: NA'
        gq = 'GQ >= %d' %self.GQmin if self.GQmin is not None else 'GQ: NA'
        ad = 'ALT ratio: %d - %d' %(self.ADmin, 100-self.ADmin) if self.ADmin is not None else 'ALT ratio: NA'
        return '\n'.join([pss, qual, dp, gq, ad])
        
    def filter(self):
        nSamples, qualcol, ADind, DPind, GQind = self.nSamples, self.qualcol, self.ADind, self.DPind, self.GQind
        PASS, QUALmin, DPmin, GQmin, ADmin = self.PASS, self.QUALmin, self.DPmin, self.GQmin, self.ADmin
        PASSind = qualcol + 1 if qualcol is not None else None
        
        PASStest = PASS and PASSind is not None
        QUALtest = QUALmin is not None and QUALmin > 0 and qualcol is not None
        ADtest = ADmin is not None and ADmin > 0 and ADind is not None
        DPtest = DPmin is not None and DPmin > 0 and DPind is not None
        GQtest = GQmin is not None and GQmin > 0 and GQind is not None

        if ADmin is not None:
            ADlow = ADmin/100.0
            ADhigh = 1 - ADlow
        
        split_it = ADtest or DPtest or GQtest
        pass_strings = ('PASS', '.')
        
        def _tester(linesplit):
            try:
                if PASStest and linesplit[PASSind] not in pass_strings: return 0
                if QUALtest and float(linesplit[qualcol]) < QUALmin: return 0
                if split_it:
                    for sample in linesplit[-nSamples:]:
                        spl = sample.split(':')
                        if DPtest and int(spl[DPind]) < DPmin: return 0
                        if GQtest and int(spl[GQind]) < GQmin: return 0
                        if ADtest and spl[0][0] != spl[0][2]:
                            ad = list(map(float, spl[ADind].split(',')))
                            if not ADlow <= ad[1]/(ad[0]+ad[1]) <= ADhigh:
                                return 0
                return 1
            except Exception as e:
                return 0
        
        return _tester

def all_variants(file, info, log, quiet):
    """Optimized variant loading with buffering"""
    _writeout('\n====VARIANT SAMPLE (BEFORE FILTERING)====', log, quiet)
    chromcol = info['chromcol']
    firstSampleCol = info['formatcol'] + 1
    
    AUTOSOM = collections.defaultdict(list)
    XCHR = collections.defaultdict(list)
    AABB = 0
    
    # Use larger buffer for faster I/O
    with _smart_open(file) as f:
        if file.endswith('.gz'):
            for line in f:
                if not line.startswith('#'):
                    break
            # Skip the first data line too, to match the non-gzip path
            # (which always lands mid-way through the 1st variant after seek)
            next(f, None)
        else:
            f.seek(info['startpos'])
            f.readline()
        for line in f:
            if 'PASS' not in line and '\t.\t' not in line: 
                continue
            linesplit = line.split('\t')
            if 'Y' in linesplit[chromcol]:
                continue
            GTtuple = getGTtuple(linesplit, firstSampleCol)
            if not GTtuple:
                continue
            if XminusPAR(linesplit, chromcol):
                XCHR[GTtuple].append(linesplit)
            else:
                AUTOSOM[GTtuple].append(linesplit)
                if 'AA' in GTtuple and 'BB' in GTtuple:
                    AABB += 1
    
    if log or not quiet:
        AUTant = sum(len(v) for v in AUTOSOM.values())
        Xant = sum(len(v) for v in XCHR.values())
        summary = 'Using all PASS variants:\n %d autosomal (including %d with AA+BB)\n %d X-linked.'%(AUTant, AABB, Xant)
        _writeout(summary, log, quiet)
    
    return AUTOSOM, XCHR

def sample_variants(file, maxsize, AABBsize, Xsize, info, log, quiet, chunks=4):
    """Unchanged from original"""
    _writeout('\n====VARIANT SAMPLE (BEFORE FILTERING)====', log, quiet)
    chromcol = info['chromcol']
    firstSampleCol = info['formatcol'] + 1
    startpos, endpos = info['startpos'], info['endpos']
    randint = random.randint
    def _randpos():
        return randint(startpos, endpos)
    
    AUTOSOM = collections.defaultdict(list)
    XCHR = collections.defaultdict(list)
    total, AUTant, Xant, AABB = 0, 0, 0, 0
    with open(file, 'r', buffering=8*1024*1024) as f:  # 8MB buffer
        while total < maxsize and (AABB < AABBsize or Xant < Xsize):
            total += 1
            f.seek(_randpos())
            f.readline()
        
            for _ in range(chunks):
                line = f.readline()
                while line and 'PASS' not in line:
                    line = f.readline()
                if not line:
                    break
                linesplit = line.split('\t')
                if 'Y' in linesplit[chromcol]:
                    continue
                GTtuple = getGTtuple(linesplit, firstSampleCol)
                if not GTtuple:
                    continue
                if XminusPAR(linesplit, chromcol):
                    XCHR[GTtuple].append(linesplit)
                    Xant += 1
                else:
                    AUTOSOM[GTtuple].append(linesplit)
                    AUTant += 1
                    if 'AA' in GTtuple and 'BB' in GTtuple:
                        AABB += 1
    _writeout('Sampled random variants:\n %d autosomal variants (including %d with AA+BB)\n %d on X.'%(AUTant, AABB, Xant), log, quiet)
     
    return AUTOSOM, XCHR

def qualityDistrib(variantlines, formatinfo, variables=['QUAL', 'DP', 'GQ', 'AD']):
    """Unchanged from original"""
    if isinstance(variantlines, dict):
        variantlines = itertools.chain(*variantlines.values())
    useVariables = [v for v in formatinfo['available_vars'] if v in variables]
    ADind = formatinfo['ADind'] if 'AD' in useVariables else None
    DPind = formatinfo['DPind'] if 'DP' in useVariables else None
    GQind = formatinfo['GQind'] if 'GQ' in useVariables else None
    qualcol = formatinfo['qualcol'] if 'QUAL' in useVariables else None
    firstSampleCol = formatinfo['formatcol'] + 1
    distrib_data = collections.defaultdict(list)
    
    def _ALTratio(ADfield):
        ad0, ad1 = map(float, ADfield.split(','))
        rat = ad1/(ad0+ad1)
        return round(min(rat, 1-rat) * 100, 1)
    
    hets = ('0/1', '0|1', '1/0', '1|0')
    for s in variantlines:
        try:
            if qualcol: distrib_data['QUAL'].append(float(s[qualcol]))
            gtcols = [a.split(':') for a in s[firstSampleCol:]]
            if DPind: distrib_data['DP'].append(min(int(b[DPind]) for b in gtcols))
            if GQind: distrib_data['GQ'].append(min(int(b[GQind]) for b in gtcols))
            if ADind: 
                ad = [_ALTratio(b[ADind]) for b in gtcols if b[0] in hets]
                if ad:
                    distrib_data['AD'].append(min(ad))
        except Exception as e:
            continue
    
    for key in distrib_data:
        distrib_data[key].sort()
    return distrib_data

def qualityPercentile(distrib_data, p, log, quiet):
    """Unchanged from original"""
    if isinstance(p, list):
        res = {key : [data[int(pp/100.0*len(data))] for pp in p] for key,data in distrib_data.items()}
    else:
        res = {key : data[int(p/100.0*len(data))] for key,data in distrib_data.items()}
        
    if log or not quiet:
        _writeout("\n====SCORE PERCENTILES (%s)====" % ", ".join(map(str, p)), log, quiet)
        rows = ["  %s: %s" % (key, res[key]) for key in ['QUAL', 'DP', 'GQ', 'AD'] if key in res]
        _writeout('\n'.join(rows), log, quiet)
    
    return res

# OPTIMIZED FUNCTIONS

def build_genotype_matrix_optimized(AUTOSOM_filt, nSamples):
    """
    Optimized genotype matrix builder.
    Returns a compact numpy array with explicit memory layout.

    IMPORTANT: Now includes variants with missing data (coded as -1).
    For large datasets (4000+ individuals), excluding variants with ANY missing data
    would exclude nearly all variants!
    """
    variant_keys = []
    gt_rows = []

    for gtcomb, lines in AUTOSOM_filt.items():
        # FIXED: Don't skip variants with missing data - include them!
        # Missing genotypes are coded as -1 and handled in calculations
        gt_numeric = tuple(_GT2NUM_[gt] for gt in gtcomb)
        n_lines = len(lines)
        if n_lines > 0:
            # Pre-allocate to reduce memory allocations
            gt_rows.extend([gt_numeric] * n_lines)
            variant_keys.extend([gtcomb] * n_lines)
    
    if not gt_rows:
        return np.array([], dtype=np.int8).reshape(0, nSamples), []
    
    # Use int8 for memory efficiency (1 byte per genotype)
    # C-contiguous array for cache efficiency
    gt_matrix = np.array(gt_rows, dtype=np.int8, order='C')
    
    # Verify shape
    assert gt_matrix.shape[1] == nSamples, f"Shape mismatch: {gt_matrix.shape[1]} != {nSamples}"
    
    return gt_matrix, variant_keys

# Numba-optimized computational kernels with parallel processing
if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True, cache=True)
    def calculate_trio_scores_batch(gt_matrix, trio_indices):
        """
        Batch process multiple trios with Numba JIT (single-threaded).
        Parallelism: ONLY at process-level to avoid oversubscription.

        trio_indices: shape (n_trios, 3) array of sample indices
        Returns: shape (n_trios, 4) array of [t1_hits, t1_total, t2_hits, t2_total]
        """
        n_trios = trio_indices.shape[0]
        n_variants = gt_matrix.shape[0]
        results = np.zeros((n_trios, 4), dtype=np.int32)

        # Process trios sequentially within each process (no prange)
        for t_idx in range(n_trios):
            idx_i = trio_indices[t_idx, 0]
            idx_j = trio_indices[t_idx, 1]
            idx_k = trio_indices[t_idx, 2]

            t1_hits = 0
            t1_total = 0
            t2_hits = 0
            t2_total = 0

            # Vectorized over variants
            for v in range(n_variants):
                g1 = gt_matrix[v, idx_i]
                g2 = gt_matrix[v, idx_j]
                g3 = gt_matrix[v, idx_k]

                # Skip variants with a missing genotype in any of the trio
                if g1 == -1 or g2 == -1 or g3 == -1:
                    continue

                # T1 score: AA + BB = AB
                if (g1 == 0 and g2 == 2) or (g1 == 2 and g2 == 0):
                    t1_total += 1
                    if g3 == 1:
                        t1_hits += 1

                # T2 score: BB + BB = BB
                if g1 == 2 and g2 == 2:
                    t2_total += 1
                    if g3 == 2:
                        t2_hits += 1

            results[t_idx, 0] = t1_hits
            results[t_idx, 1] = t1_total
            results[t_idx, 2] = t2_hits
            results[t_idx, 3] = t2_total

        return results

    @jit(nopython=True, fastmath=True, cache=True)
    def calculate_pair_scores_batch(gt_matrix, pair_indices):
        """
        Batch process multiple pairs with Numba JIT (single-threaded).
        Parallelism: ONLY at process-level to avoid oversubscription.

        pair_indices: shape (n_pairs, 2) array of sample indices
        Returns: shape (n_pairs, 4) array of [neither_aa, ibs2, either_bb, ibs_gt0]
        """
        n_pairs = pair_indices.shape[0]
        n_variants = gt_matrix.shape[0]
        results = np.zeros((n_pairs, 4), dtype=np.int32)

        # Process pairs sequentially within each process (no prange)
        for p_idx in range(n_pairs):
            idx_i = pair_indices[p_idx, 0]
            idx_j = pair_indices[p_idx, 1]

            neither_aa = 0
            ibs2 = 0
            either_bb = 0
            ibs_gt0 = 0

            for v in range(n_variants):
                g1 = gt_matrix[v, idx_i]
                g2 = gt_matrix[v, idx_j]

                # Skip variants with a missing genotype in either sample
                if g1 == -1 or g2 == -1:
                    continue

                # MZ score calculations
                if g1 != 0 and g2 != 0:
                    neither_aa += 1
                    if g1 == g2:
                        ibs2 += 1

                # PO score calculations
                if g1 == 2 or g2 == 2:
                    either_bb += 1
                    # IBS > 0 means not IBS=0 (which is AA vs BB or BB vs AA)
                    if not ((g1 == 0 and g2 == 2) or (g1 == 2 and g2 == 0)):
                        ibs_gt0 += 1

            results[p_idx, 0] = neither_aa
            results[p_idx, 1] = ibs2
            results[p_idx, 2] = either_bb
            results[p_idx, 3] = ibs_gt0

        return results
else:
    # Fallback non-Numba versions
    def calculate_trio_scores_batch(gt_matrix, trio_indices):
        """Non-Numba fallback"""
        n_trios = trio_indices.shape[0]
        results = np.zeros((n_trios, 4), dtype=np.int32)
        
        for t_idx in range(n_trios):
            idx_i, idx_j, idx_k = trio_indices[t_idx]
            g1 = gt_matrix[:, idx_i]
            g2 = gt_matrix[:, idx_j]
            g3 = gt_matrix[:, idx_k]

            # Exclude variants with a missing genotype in any of the trio
            present = (g1 != -1) & (g2 != -1) & (g3 != -1)

            t1_valid = (((g1 == 0) & (g2 == 2)) | ((g1 == 2) & (g2 == 0))) & present
            t1_total = np.sum(t1_valid)
            t1_hits = np.sum(t1_valid & (g3 == 1))

            t2_valid = (g1 == 2) & (g2 == 2) & present
            t2_total = np.sum(t2_valid)
            t2_hits = np.sum(t2_valid & (g3 == 2))

            results[t_idx] = [t1_hits, t1_total, t2_hits, t2_total]

        return results

    def calculate_pair_scores_batch(gt_matrix, pair_indices):
        """Non-Numba fallback"""
        n_pairs = pair_indices.shape[0]
        results = np.zeros((n_pairs, 4), dtype=np.int32)

        for p_idx in range(n_pairs):
            idx_i, idx_j = pair_indices[p_idx]
            g1 = gt_matrix[:, idx_i]
            g2 = gt_matrix[:, idx_j]

            # Exclude variants with a missing genotype in either sample
            present = (g1 != -1) & (g2 != -1)

            neither_aa = (g1 != 0) & (g2 != 0) & present
            ibs2 = neither_aa & (g1 == g2)

            either_bb = ((g1 == 2) | (g2 == 2)) & present
            ibs0 = ((g1 == 0) & (g2 == 2)) | ((g1 == 2) & (g2 == 0))
            ibs_gt0 = either_bb & ~ibs0

            results[p_idx] = [np.sum(neither_aa), np.sum(ibs2),
                            np.sum(either_bb), np.sum(ibs_gt0)]

        return results

def process_pairs_batch(args):
    """
    Worker function for batch processing pairs.
    Processes multiple pairs at once for better efficiency.
    """
    shm_name, shape, dtype, pair_batch, percentile, MZ_thresh, PO_thresh = args
    
    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    gt_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    # Convert to numpy array for batch processing
    pair_indices = np.array(pair_batch, dtype=np.int32)
    
    # Batch compute all pairs
    batch_results = calculate_pair_scores_batch(gt_matrix, pair_indices)
    
    # Format results
    results = []
    for i, pair in enumerate(pair_batch):
        neither_aa, ibs2, either_bb, ibs_gt0 = batch_results[i]
        
        MZp = 100.0 * ibs2 / neither_aa if neither_aa > 0 else 0
        POp = 100.0 * ibs_gt0 / either_bb if either_bb > 0 else 0
        
        if neither_aa >= 100 and MZp >= MZ_thresh:
            verdict = 'MZ twins'
        elif either_bb >= 100 and POp >= PO_thresh:
            verdict = 'Parent-child' 
        elif either_bb >= 100 and POp < PO_thresh:
            verdict = 'Other/unrelated'
        else:
            verdict = 'na'
        
        results.append(dict(
            pair=pair, percentile=percentile, autos=shape[0],
            neitherAA=int(neither_aa), IBS2=int(ibs2), MZp=float(MZp),
            eitherBB=int(either_bb), IBS12=int(ibs_gt0), POp=float(POp),
            verdict=verdict
        ))
    
    shm.close()
    return results

def process_trios_batch(args):
    """
    Worker function for batch processing trios.
    Processes multiple trios at once for better efficiency.
    """
    shm_name, shape, dtype, trio_batch, orders, percentile, T1_thresh, T2_thresh = args
    
    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    gt_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    results = []
    
    for triple in trio_batch:
        for order in orders:
            pivot_pos = order.index(2)
            pivot = triple[pivot_pos]
            others = [triple[i] for i in range(3) if i != pivot_pos]
            idx_array = np.array([others + [pivot]], dtype=np.int32)

            # Compute scores
            batch_results = calculate_trio_scores_batch(gt_matrix, idx_array)
            t1_hits, t1_total, t2_hits, t2_total = batch_results[0]
            
            AA_BB_ABp = round(100.0 * t1_hits / t1_total, 2) if t1_total > 0 else 0
            BB_BB_BBp = round(100.0 * t2_hits / t2_total, 2) if t2_total > 0 else 0
            
            if t1_total >= 100:
                if AA_BB_ABp >= T1_thresh:
                    verdict = 'Regular trio' if BB_BB_BBp >= T2_thresh else 'Inverted/generational trio'
                else:
                    verdict = 'Not trio'
            else:
                verdict = 'na'
            
            results.append(dict(
                triple=triple, order=order, pivot=pivot, percentile=percentile,
                verdict=verdict, autos=shape[0], tot_AA_BB=int(t1_total),
                AA_BB_AB=int(t1_hits), AA_BB_ABp=float(AA_BB_ABp),
                tot_BB_BB=int(t2_total), BB_BB_BB=int(t2_hits), BB_BB_BBp=float(BB_BB_BBp)
            ))
    
    shm.close()
    return results

def inferGenders(XCHRfilt, formatinfo, percentile, FEMALE_thresh, MALE_thresh):
    """Unchanged from original"""
    nSamples = formatinfo['nSamples']
    Xdistribs = [collections.Counter() for _ in range(nSamples)]
    Xtot = 0
    for gtcomb, lines in XCHRfilt.items():
        nvars = len(lines)
        Xtot += nvars
        for i, gt in enumerate(gtcomb):
            Xdistribs[i][gt] += nvars 
    
    common = dict(percentile=percentile, Xtot=Xtot)
    def dist2gender(Xdist):
        denom = Xdist['AB']+Xdist['BB']
        Xhetp = 100.0 * Xdist['AB']/denom if denom > 0 else 0
        if denom >=25:
            if Xhetp <= MALE_thresh:
                gender = 'Male' 
            elif Xhetp < FEMALE_thresh:
                gender = '?'
            else:
                gender = 'Female' 
        else: gender = 'na'
        return dict(Xhetp=Xhetp, gender=gender)
    
    res = []
    for i, Xdist in enumerate(Xdistribs):
        r = dist2gender(Xdist)
        r.update(dict(sample=i, AA=Xdist['AA'], AB=Xdist['AB'], BB=Xdist['BB'], **common))
        res.append(r)
    
    return res

def _trio_row(d):
    """Format one trio result dict as a plain row list — factored out of
    trio_table() so both the materializing table builder and the streaming
    selection path (_stream_best_trios()) share one formatting definition."""
    triple_txt = '%d,%d,%d' % tuple(p+1 for p in d['triple'])
    pivot_out = d['pivot'] + 1
    test1 = '%.1f' % d['AA_BB_ABp']
    test2 = '%.1f' % d['BB_BB_BBp']
    return [triple_txt, pivot_out, d['percentile'], d['autos'], d['tot_AA_BB'], d['AA_BB_AB'],
            test1, d['tot_BB_BB'], d['BB_BB_BB'], test2, d['verdict']]

def _pair_row(d):
    """Format one pair result dict as a plain row list — factored out of
    pairwise_table() so both the materializing table builder and the
    streaming selection path (_stream_best_pairs()) share one formatting
    definition."""
    pair_txt = '%d,%d' % tuple(p+1 for p in d['pair'])
    MZscore = '%.1f' % d['MZp']
    POscore = '%.1f' % d['POp']
    return [pair_txt, d['percentile'], d['autos'], d['neitherAA'], d['IBS2'], MZscore, d['eitherBB'], d['IBS12'], POscore, d['verdict']]

def trio_table(TRIORES):
    """Unchanged from original"""
    table = [['Triple', 'Pivot', 'Perc', 'Autos', 'AA+BB', '=AB', 'Test1', 'BB+BB', '=BB', 'Test2', 'Verdict']]
    for d in TRIORES:
        table.append(_trio_row(d))
    return table

def pairwise_table(PAIRRES):
    """Unchanged from original"""
    table = [['Pair', 'Perc', 'Autos', 'NeitherAA', 'IBS2', 'MZscore', 'EitherBB', 'IBS>0', 'POscore', 'Verdict']]
    for d in PAIRRES:
        table.append(_pair_row(d))
    return table

def gender_table(GENDERRES):
    """Unchanged from original"""
    table = [['Sample', 'Perc', 'Xvars', 'AA', 'AB', 'BB', 'Xhet', 'Gender']]
    for d in GENDERRES:
        XHET = '%.1f' % d['Xhetp']
        printdat = [d['sample']+1, d['percentile'], d['Xtot'], d['AA'], d['AB'], d['BB'], XHET, d['gender']]
        table.append(printdat)
    return table

def _stream_best_trios(triodata, reportall):
    """
    Streaming core of bestTrios(): yields one formatted row (no header) per
    selected triple instead of accumulating a full 'best' list and then
    running trio_table() over it. Selection/tie-break logic is byte-for-byte
    identical to bestTrios() below — this is now the single source of truth
    for that logic; bestTrios() is a thin materializing wrapper around it.
    """
    for k,v in itertools.groupby(triodata, key=lambda x: x['triple']):
        allcalls = list(v)
        nonNA = [r for r in allcalls if r['verdict'] != 'na']
        if not nonNA:
            if reportall:
                yield _trio_row(allcalls[0])
            continue
        nonNA.sort(key=lambda x: (-round(x['AA_BB_ABp'], 1), x['percentile']))
        b = nonNA[0]
        if b['verdict'][:3] in ('Reg', 'Inv')  or reportall:
            yield _trio_row(b)

def _stream_best_pairs(pairdata, reportall):
    """
    Streaming core of bestPairs(): yields one formatted row (no header) per
    selected pair instead of accumulating a full 'best' list and then
    running pairwise_table() over it. Selection/tie-break logic is
    byte-for-byte identical to bestPairs() below — this is now the single
    source of truth for that logic; bestPairs() is a thin materializing
    wrapper around it.
    """
    for k,v in itertools.groupby(pairdata, key=lambda x: x['pair']):
        allcalls = list(v)
        nonNA = [r for r in allcalls if r['verdict'] != 'na']
        if not nonNA:
            if reportall:
                yield _pair_row(allcalls[0])
            continue
        verdict = nonNA[-1]['verdict']
        if verdict == 'MZ twins':
            nonNA.sort(key=lambda x: (-round(x['MZp'], 1), x['percentile']))
        elif verdict == 'Parent-child' or reportall:
            nonNA.sort(key=lambda x: (-round(x['POp'], 1), x['percentile']))
        else:
            continue
        yield _pair_row(nonNA[0])

def bestTrios(triodata, reportall):
    """
    Materializing wrapper around _stream_best_trios() — used by the
    checkped()/pedfile path, which needs list/zip semantics on a tiny
    (exactly 3-sample) dataset. The large-scale output path in vcfped()
    uses _stream_best_trios() directly instead of this, to avoid
    materializing the full result set under --all.
    """
    header = ['Triple', 'Pivot', 'Perc', 'Autos', 'AA+BB', '=AB', 'Test1', 'BB+BB', '=BB', 'Test2', 'Verdict']
    return [header] + list(_stream_best_trios(triodata, reportall))

def bestPairs(pairdata, reportall):
    """
    Materializing wrapper around _stream_best_pairs() — used by the
    checkped()/pedfile path, which needs list/zip semantics on a tiny
    (exactly 3-sample) dataset. The large-scale output path in vcfped()
    uses _stream_best_pairs() directly instead of this, to avoid
    materializing the full result set under --all.
    """
    header = ['Pair', 'Perc', 'Autos', 'NeitherAA', 'IBS2', 'MZscore', 'EitherBB', 'IBS>0', 'POscore', 'Verdict']
    return [header] + list(_stream_best_pairs(pairdata, reportall))

def bestGenders(genderdata):
    """Unchanged from original"""
    best = []
    sortfun = lambda x: (round(x['Xhetp'], 1), x['percentile'])
    for k,v in itertools.groupby(genderdata, key=lambda x: x['sample']):
        allcalls = list(v)
        noNA = [r for r in allcalls if r['gender'] != 'na']
        if not noNA: 
            best.append(allcalls[0])
            continue
        verdicts = {r['gender'] for r in noNA}
        if 'Male' in verdicts and 'Female' in verdicts:
            noQ = [r for r in noNA if r['gender'] != '?']
            best.append(noQ[-1])
            continue
        if verdicts == {'?'}:
            verd = '?'
        else:
            verd = list(verdicts.difference({'?'}))[0]
        use = sorted([r for r in noNA if r['gender'] == verd], key=sortfun)
        best.append(use[0])
    
    return gender_table(best)

def _is_better_pair_result(new, current):
    """
    Determine if new pair result is better than current best.
    Better means: more confident verdict or higher scores.

    TODO(known discrepancy, --all only): this online best-tracking collapses
    each pair down to ONE row across all percentiles as results stream in
    (ties for 'Other/unrelated'/'na' broken by LOWEST percentile), before
    bestPairs() ever runs. bestPairs() itself picks its sort strategy from
    nonNA[-1]['verdict'] (the HIGHEST-percentile row) and, for
    'Other/unrelated' + reportall, sorts by HIGHEST POp instead — a
    different rule. vcfped.py (the reference engine) never does this early
    per-percentile collapse, so it always gives bestPairs() the full
    3-percentile picture. Net effect: with --all, vcfped_optimized.py can
    report a different percentile (score) for 'Other/unrelated' pairs than
    vcfped.py — everything else (MZ twins, Parent-child, and all non-`--all`
    output) is unaffected, since those verdicts are excluded from output by
    default and this only surfaces per-pair when --all reveals them. Not
    fixed here; needs the two selection algorithms reconciled (see
    _is_better_trio_result() below for the equivalent trio-side TODO —
    'Not trio' has the same structural risk though not yet observed).
    """
    # Verdict priority: MZ twins > Parent-child > Other > na
    verdict_priority = {'MZ twins': 3, 'Parent-child': 2, 'Other/unrelated': 1, 'na': 0}
    
    new_priority = verdict_priority.get(new['verdict'], 0)
    current_priority = verdict_priority.get(current['verdict'], 0)
    
    if new_priority > current_priority:
        return True
    elif new_priority < current_priority:
        return False
    
    # Same verdict, compare scores
    if new['verdict'] == 'MZ twins':
        return new['MZp'] > current['MZp']
    elif new['verdict'] == 'Parent-child':
        return new['POp'] > current['POp']
    else:
        # For 'na' or 'Other', prefer lower percentile (less filtering)
        return new['percentile'] < current['percentile']

def _is_better_trio_result(new, current):
    """
    Determine if new trio result is better than current best.
    Better means: more confident verdict or higher scores.

    TODO(known discrepancy risk, --all only): same structural issue as
    _is_better_pair_result()'s TODO above, applied to trios — this collapses
    each (triple, pivot) to one row across percentiles before bestTrios()
    runs, using its own tie-break rule ('Not trio'/'na' ties broken by
    higher AA_BB_ABp, then lower percentile). bestTrios() additionally picks
    the best PIVOT per triple from nonNA sorted by (-round(AA_BB_ABp,1),
    percentile), which is consistent with this function's primary tie-break
    — but for the 'Not trio' verdict specifically, the two collapse points
    (per-percentile here, per-pivot in bestTrios()) are not guaranteed to
    agree with what a single non-collapsing pass over the full percentile x
    pivot grid (as vcfped.py does) would pick. Not observed in testing yet
    (unlike the pair-side 'Other/unrelated' case, which is confirmed via
    diff against vcfped.py under --all) — flagging as the same class of risk
    pending a dedicated test case that exercises it.
    """
    # Verdict priority: Regular trio > Inverted trio > Not trio > na
    verdict_priority = {'Regular trio': 3, 'Inverted/generational trio': 2, 'Not trio': 1, 'na': 0}
    
    new_priority = verdict_priority.get(new['verdict'], 0)
    current_priority = verdict_priority.get(current['verdict'], 0)
    
    if new_priority > current_priority:
        return True
    elif new_priority < current_priority:
        return False
    
    # Same verdict, compare T1 score (most important)
    if new['AA_BB_ABp'] > current['AA_BB_ABp']:
        return True
    elif new['AA_BB_ABp'] < current['AA_BB_ABp']:
        return False
    
    # Same T1 score, prefer lower percentile (less filtering)
    return new['percentile'] < current['percentile']

# ---------------------------------------------------------------------------
# SQLite-backed result accumulation (replaces in-memory BEST_PAIRS/BEST_TRIOS
# dicts).
#
# The online best-tracking dicts retain one entry per pair (O(N^2)) and one
# entry per (triple, pivot) (O(3 * N^3)) for the LIFETIME of the run,
# regardless of --all/reportall — that flag only controls what gets written
# out afterward, not what gets accumulated during the parallel phase. At
# cohort scale (N in the thousands) this alone can reach hundreds of GB to
# TB before any output formatting even starts.
#
# Backing this with SQLite instead: moves accumulation to disk (or OS page
# cache) rather than the Python heap, replaces the final Python sorted()
# calls with index-backed ORDER BY (the primary key IS the sort key, so no
# separate sort pass is needed), and lets the final "best per group"
# selection in bestPairs()/bestTrios() stream row-by-row via a cursor
# instead of holding every combination in RAM as a materialized list —
# those two functions are otherwise completely unchanged, since
# itertools.groupby() works the same over a generator as over a list.
# ---------------------------------------------------------------------------

_PAIR_VERDICT_PRIORITY = {'MZ twins': 3, 'Parent-child': 2, 'Other/unrelated': 1, 'na': 0}
_TRIO_VERDICT_PRIORITY = {'Regular trio': 3, 'Inverted/generational trio': 2, 'Not trio': 1, 'na': 0}


def _open_results_db():
    """Create a scratch SQLite DB for online best-tracking. Returns (conn, path)."""
    fd, path = tempfile.mkstemp(prefix='vcfped_', suffix='.sqlite3')
    os.close(fd)
    conn = sqlite3.connect(path)
    # Disposable scratch file for a single run: durability is not needed,
    # only throughput, so relax the usual crash-safety guarantees.
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA temp_store=MEMORY')
    conn.execute('''
        CREATE TABLE best_pairs (
            pair_a INTEGER NOT NULL, pair_b INTEGER NOT NULL,
            percentile INTEGER NOT NULL, autos INTEGER NOT NULL,
            neitherAA INTEGER NOT NULL, IBS2 INTEGER NOT NULL, MZp REAL NOT NULL,
            eitherBB INTEGER NOT NULL, IBS12 INTEGER NOT NULL, POp REAL NOT NULL,
            verdict TEXT NOT NULL, verdict_priority INTEGER NOT NULL,
            PRIMARY KEY (pair_a, pair_b)
        ) WITHOUT ROWID
    ''')
    conn.execute('''
        CREATE TABLE best_trios (
            t_a INTEGER NOT NULL, t_b INTEGER NOT NULL, t_c INTEGER NOT NULL,
            pivot INTEGER NOT NULL, percentile INTEGER NOT NULL, autos INTEGER NOT NULL,
            tot_AA_BB INTEGER NOT NULL, AA_BB_AB INTEGER NOT NULL, AA_BB_ABp REAL NOT NULL,
            tot_BB_BB INTEGER NOT NULL, BB_BB_BB INTEGER NOT NULL, BB_BB_BBp REAL NOT NULL,
            verdict TEXT NOT NULL, verdict_priority INTEGER NOT NULL,
            PRIMARY KEY (t_a, t_b, t_c, pivot)
        ) WITHOUT ROWID
    ''')
    return conn, path


def _close_results_db(conn, path):
    conn.close()
    try:
        os.remove(path)
    except OSError:
        pass


def _upsert_pairs(conn, results):
    """
    Batched online best-tracking for pair results, replicating
    _is_better_pair_result()'s tie-break rules as a SQL upsert: higher
    verdict priority wins; for ties, MZ twins compares by MZp, Parent-child
    by POp, and na/Other by lower percentile.
    """
    if not results:
        return
    rows = []
    for r in results:
        a, b = r['pair']
        vp = _PAIR_VERDICT_PRIORITY.get(r['verdict'], 0)
        rows.append((a, b, r['percentile'], r['autos'], r['neitherAA'], r['IBS2'],
                     r['MZp'], r['eitherBB'], r['IBS12'], r['POp'], r['verdict'], vp))
    with conn:
        conn.executemany('''
            INSERT INTO best_pairs
                (pair_a, pair_b, percentile, autos, neitherAA, IBS2, MZp, eitherBB, IBS12, POp, verdict, verdict_priority)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(pair_a, pair_b) DO UPDATE SET
                percentile=excluded.percentile, autos=excluded.autos, neitherAA=excluded.neitherAA,
                IBS2=excluded.IBS2, MZp=excluded.MZp, eitherBB=excluded.eitherBB, IBS12=excluded.IBS12,
                POp=excluded.POp, verdict=excluded.verdict, verdict_priority=excluded.verdict_priority
            WHERE excluded.verdict_priority > best_pairs.verdict_priority
               OR (excluded.verdict_priority = best_pairs.verdict_priority AND excluded.verdict_priority = 3
                   AND excluded.MZp > best_pairs.MZp)
               OR (excluded.verdict_priority = best_pairs.verdict_priority AND excluded.verdict_priority = 2
                   AND excluded.POp > best_pairs.POp)
               OR (excluded.verdict_priority = best_pairs.verdict_priority AND excluded.verdict_priority NOT IN (2, 3)
                   AND excluded.percentile < best_pairs.percentile)
        ''', rows)


def _upsert_trios(conn, results):
    """
    Batched online best-tracking for trio results, replicating
    _is_better_trio_result()'s tie-break rules as a SQL upsert: higher
    verdict priority wins; ties broken by higher AA_BB_ABp (T1), then by
    lower percentile.
    """
    if not results:
        return
    rows = []
    for r in results:
        a, b, c = r['triple']
        vp = _TRIO_VERDICT_PRIORITY.get(r['verdict'], 0)
        rows.append((a, b, c, r['pivot'], r['percentile'], r['autos'], r['tot_AA_BB'],
                     r['AA_BB_AB'], r['AA_BB_ABp'], r['tot_BB_BB'], r['BB_BB_BB'],
                     r['BB_BB_BBp'], r['verdict'], vp))
    with conn:
        conn.executemany('''
            INSERT INTO best_trios
                (t_a, t_b, t_c, pivot, percentile, autos, tot_AA_BB, AA_BB_AB, AA_BB_ABp,
                 tot_BB_BB, BB_BB_BB, BB_BB_BBp, verdict, verdict_priority)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(t_a, t_b, t_c, pivot) DO UPDATE SET
                percentile=excluded.percentile, autos=excluded.autos, tot_AA_BB=excluded.tot_AA_BB,
                AA_BB_AB=excluded.AA_BB_AB, AA_BB_ABp=excluded.AA_BB_ABp, tot_BB_BB=excluded.tot_BB_BB,
                BB_BB_BB=excluded.BB_BB_BB, BB_BB_BBp=excluded.BB_BB_BBp, verdict=excluded.verdict,
                verdict_priority=excluded.verdict_priority
            WHERE excluded.verdict_priority > best_trios.verdict_priority
               OR (excluded.verdict_priority = best_trios.verdict_priority
                   AND excluded.AA_BB_ABp > best_trios.AA_BB_ABp)
               OR (excluded.verdict_priority = best_trios.verdict_priority
                   AND excluded.AA_BB_ABp = best_trios.AA_BB_ABp
                   AND excluded.percentile < best_trios.percentile)
        ''', rows)


def _iter_pair_rows(conn):
    """
    Stream best_pairs rows back out as the same dict shape bestPairs()/
    pairwise_table() already expect, ordered by pair (via the primary key
    index, so no separate sort pass) — lets those functions run unchanged
    against a cursor instead of a materialized list.
    """
    cur = conn.execute('''
        SELECT pair_a, pair_b, percentile, autos, neitherAA, IBS2, MZp, eitherBB, IBS12, POp, verdict
        FROM best_pairs ORDER BY pair_a, pair_b
    ''')
    for a, b, percentile, autos, neitherAA, IBS2, MZp, eitherBB, IBS12, POp, verdict in cur:
        yield dict(pair=(a, b), percentile=percentile, autos=autos, neitherAA=neitherAA,
                   IBS2=IBS2, MZp=MZp, eitherBB=eitherBB, IBS12=IBS12, POp=POp, verdict=verdict)


def _iter_trio_rows(conn):
    """
    Stream best_trios rows back out as the same dict shape bestTrios()/
    trio_table() already expect, ordered by (triple, pivot) (via the
    primary key index) — lets those functions run unchanged against a
    cursor instead of a materialized list.
    """
    cur = conn.execute('''
        SELECT t_a, t_b, t_c, pivot, percentile, autos, tot_AA_BB, AA_BB_AB, AA_BB_ABp,
               tot_BB_BB, BB_BB_BB, BB_BB_BBp, verdict
        FROM best_trios ORDER BY t_a, t_b, t_c, pivot
    ''')
    for (a, b, c, pivot, percentile, autos, tot_AA_BB, AA_BB_AB, AA_BB_ABp,
         tot_BB_BB, BB_BB_BB, BB_BB_BBp, verdict) in cur:
        yield dict(triple=(a, b, c), pivot=pivot, percentile=percentile, autos=autos,
                   tot_AA_BB=tot_AA_BB, AA_BB_AB=AA_BB_AB, AA_BB_ABp=AA_BB_ABp,
                   tot_BB_BB=tot_BB_BB, BB_BB_BB=BB_BB_BB, BB_BB_BBp=BB_BB_BBp, verdict=verdict)


def _stream_complete_pairs_to_log(conn, log, outfile, has_data):
    """
    Write every stored pair result to the log file as plain TSV, streaming
    row-by-row instead of building one tabulate()-aligned string in memory.
    tabulate() needs the full table materialized to compute column
    alignment, which does not scale at cohort size — this is log-only
    output (not part of any tested .pair/.trio file), so the plain-TSV
    layout here does not need to match tabulate's aligned columns.
    """
    if not log:
        return
    print("\n====PAIRWISE RELATIONS====", file=log)
    if not has_data:
        print("Not enough individuals in the file", file=log)
    else:
        print('\t'.join(['Pair', 'Perc', 'Autos', 'NeitherAA', 'IBS2', 'MZscore',
                         'EitherBB', 'IBS>0', 'POscore', 'Verdict']), file=log)
        for r in _iter_pair_rows(conn):
            pair_txt = '%d,%d' % (r['pair'][0] + 1, r['pair'][1] + 1)
            print('\t'.join(str(x) for x in [
                pair_txt, r['percentile'], r['autos'], r['neitherAA'], r['IBS2'],
                '%.1f' % r['MZp'], r['eitherBB'], r['IBS12'], '%.1f' % r['POp'], r['verdict']
            ]), file=log)
    print(f"\nMost confident pairwise calls written to {outfile}", file=log)


def _stream_complete_trios_to_log(conn, log, outfile, has_data):
    """Trio counterpart of _stream_complete_pairs_to_log() — see that docstring."""
    if not log:
        return
    print("\n====TRIO RELATIONS====", file=log)
    if not has_data:
        print("Not enough individuals in the file", file=log)
    else:
        print('\t'.join(['Triple', 'Pivot', 'Perc', 'Autos', 'AA+BB', '=AB', 'Test1',
                         'BB+BB', '=BB', 'Test2', 'Verdict']), file=log)
        for r in _iter_trio_rows(conn):
            triple_txt = '%d,%d,%d' % tuple(p + 1 for p in r['triple'])
            print('\t'.join(str(x) for x in [
                triple_txt, r['pivot'] + 1, r['percentile'], r['autos'], r['tot_AA_BB'],
                r['AA_BB_AB'], '%.1f' % r['AA_BB_ABp'], r['tot_BB_BB'], r['BB_BB_BB'],
                '%.1f' % r['BB_BB_BBp'], r['verdict']
            ]), file=log)
    print(f"\nMost confident trio calls written to {outfile}", file=log)


def _stream_and_write_best_pairs(conn, reportall, outfile, quiet, has_data):
    """
    Single streaming pass over the "best per pair" selection: for each row
    _stream_best_pairs() yields, writes it to the .pair file (plain TSV,
    byte-for-byte matching what '\\n'.join(pairwise_table(bestP)) used to
    produce) and, unless quiet, prints the same row to stdout.

    Replaces materializing bestP as a full Python list before writing or
    tabulate()-formatting it — bestPairs() keeps essentially every row when
    reportall=True (--all), so at cohort scale that list does not fit
    comfortably in memory alongside the accumulation-phase data.

    has_data mirrors the original `if PAIRRES else "Not enough individuals
    in the file"` check: whether ANY pair was ever scored at all (cohort
    large enough to form pairs), independent of whether any of them were
    CONFIRMED relationships (a cohort with zero confirmed pairs but >=2
    samples still gets a header-only table here, matching prior behavior).
    """
    header = ['Pair', 'Perc', 'Autos', 'NeitherAA', 'IBS2', 'MZscore', 'EitherBB', 'IBS>0', 'POscore', 'Verdict']
    if not quiet:
        print("\n====PAIRWISE RELATIONS====")
        if not has_data:
            print("Not enough individuals in the file")
        else:
            print('\t'.join(map(str, header)))

    with open(outfile, 'w') as out:
        out.write('\t'.join(map(str, header)))
        for row in _stream_best_pairs(_iter_pair_rows(conn), reportall):
            out.write('\n' + '\t'.join(map(str, row)))
            if not quiet:
                print('\t'.join(map(str, row)))

    if not quiet:
        print(f"\nMost confident pairwise calls written to {outfile}")


def _stream_and_write_best_trios(conn, reportall, outfile, quiet, has_data):
    """Trio counterpart of _stream_and_write_best_pairs() — see that docstring."""
    header = ['Triple', 'Pivot', 'Perc', 'Autos', 'AA+BB', '=AB', 'Test1', 'BB+BB', '=BB', 'Test2', 'Verdict']
    if not quiet:
        print("\n====TRIO RELATIONS====")
        if not has_data:
            print("Not enough individuals in the file")
        else:
            print('\t'.join(map(str, header)))

    with open(outfile, 'w') as out:
        out.write('\t'.join(map(str, header)))
        for row in _stream_best_trios(_iter_trio_rows(conn), reportall):
            out.write('\n' + '\t'.join(map(str, row)))
            if not quiet:
                print('\t'.join(map(str, row)))

    if not quiet:
        print(f"\nMost confident trio calls written to {outfile}")


def vcfped(file, quiet=True, reportall=False, prefix=None, variables=['QUAL','DP','GQ','AD'], percentiles=[10,30,50],
           T1_thresh=90, T2_thresh=95, MZ_thresh=95, PO_thresh=99, MALE_thresh=5, FEMALE_thresh=25, pedfile=None,
           nogender=False, nopairwise=False, notrio=False, exactmax=150000, samplesize=10000, samplesizeAABB=500,
           n_jobs=None, batch_size=None, max_trio_batches=None):
    """
    Optimized VCF relationship detection for large datasets.

    Additional parameters compared to the reference implementation:
        n_jobs (int): Number of parallel processes. Default: all available cores,
            except on machines with more than 8 cores, where the default is
            capped at 25% of the core count (see _default_n_jobs()).
        batch_size (int): Number of pairs/trios per batch. Default: auto-calculated
        max_trio_batches (int): Max number of trio batches to process (None = all).
                                Set this to limit memory for huge datasets.
    """
    
    log = None if prefix is None or prefix in ("", "STDOUT") else open('%s.log'%prefix, 'w')
    callGenders, callPairs, callTrios = not nogender, not nopairwise, not notrio
    ynG, ynP, ynT = tuple('YES' if b else 'NO' for b in (callGenders, callPairs, callTrios))
    
    # Determine optimal number of processes (NO threading - pure process parallelism)
    total_cores = cpu_count()
    n_jobs = min(_default_n_jobs(n_jobs, total_cores), total_cores)

    form = '{:>40} : {}'
    _writeout('VCFped %s (Optimized)\n'%__version__, log, quiet)
    _writeout('Options in effect:', log, quiet)
    _writeout(form.format('Input file', file), log, quiet)
    _writeout(form.format('Output file prefix', prefix), log, quiet)
    _writeout(form.format('Analysis:', 'Genders [%s], pairwise [%s], trios [%s]' %(ynG, ynP, ynT)), log, quiet)
    _writeout(form.format('Compare with pedfile', pedfile), log, quiet)
    _writeout(form.format('Include in output:', 'All pairs/triples' if reportall else 'Only inferred pairs/triples'), log, quiet)
    _writeout(form.format('Trio thresholds', 'T1 (AA+BB=AB) [%d%%], T2 (BB+BB=BB) [%d%%]'%(T1_thresh,T2_thresh)), log, quiet)
    _writeout(form.format('Pairwise thresholds', 'MZ (IBS=2) [%d%%], PO (IBS>0) [%d%%]'%(MZ_thresh,PO_thresh)), log, quiet)
    _writeout(form.format('Gender thresholds', 'Male (Xhet) [<%d%%], female (Xhet) [>%d%%]' %(MALE_thresh, FEMALE_thresh)), log, quiet)
    _writeout(form.format('Parallel configuration', f'{n_jobs} processes (single-threaded, Numba: {NUMBA_AVAILABLE})'), log, quiet)
    _writeout(form.format('Optimization mode', 'Shared memory + multiprocessing + JIT + online best-tracking'), log, quiet)
    _writeout(form.format('Sample if approx. count exceeds', exactmax), log, quiet)
    _writeout(form.format('If sampling: Minimum # variants', samplesize), log, quiet)
    _writeout(form.format('If sampling: Minimum with AA and BB', samplesizeAABB), log, quiet)
    _writeout(form.format('Filtering variables', ' '.join(variables)), log, quiet)
    _writeout(form.format('Filtering percentile ranks', ' '.join(map(str, percentiles))), log, quiet)
    st = time.time()
    
    info = inspectFile(file, log=log, quiet=quiet)
    nSamples = info['nSamples']
    samples = info['samples']
    linecount = info['linecount']
    
    badvars = [v for v in variables if v not in ['QUAL','DP','GQ','AD']]
    if badvars:
        raise RuntimeError("Unknown quality variables: %s." % ', '.join(badvars))
    
    if file.endswith('.gz') or linecount <= exactmax:
        AUTOSOM, XCHR = all_variants(file, info=info, log=log, quiet=quiet)
    else:
        AUTOSOM, XCHR = sample_variants(file, maxsize=samplesize*2, AABBsize=samplesizeAABB, Xsize=100, info=info, log=log, quiet=quiet)
    
    qualdist = qualityDistrib(AUTOSOM, info, variables)
    qualityPercentile(qualdist, percentiles, log=log, quiet=quiet)

    autosom_filt = AUTOSOM.copy()
    x_filt = XCHR.copy()
    
    pairs = list(itertools.combinations(range(nSamples), 2))
    triples = list(itertools.combinations(range(nSamples), 3))
    
    _writeout(f"\nDataset size: {nSamples} samples", log, quiet)
    _writeout(f"  Pairs to analyze: {len(pairs):,}", log, quiet)
    _writeout(f"  Trios to analyze: {len(triples):,}", log, quiet)
    
    # Calculate optimal batch sizes for process-level parallelism
    # Key: Larger batches reduce inter-process communication overhead
    # but should be small enough to keep all processes fed with work
    if batch_size is None:
        # Target: 3-6 batches per core for good load balancing
        # For 64 cores: 192-384 total batches
        batches_per_core = 5

        # For pairs: Larger batches since computation per item is lighter
        if len(pairs) > 0:
            pair_batch_size = max(1000, min(10000, len(pairs) // (n_jobs * batches_per_core)))
        else:
            pair_batch_size = 1000

        # For trios: Adjust based on dataset size
        if len(triples) > 0:
            if len(triples) < 1000000:
                trio_batch_size = max(500, min(5000, len(triples) // (n_jobs * batches_per_core)))
            else:
                trio_batch_size = max(1000, min(10000, len(triples) // (n_jobs * batches_per_core)))
        else:
            trio_batch_size = 500
    else:
        pair_batch_size = batch_size
        trio_batch_size = batch_size
    
    _writeout(f"Batch sizes (for load balancing): pairs={pair_batch_size}, trios={trio_batch_size}", log, quiet=True)
    _writeout(f"  Estimated batches: pairs={len(pairs)//pair_batch_size}, trios={len(triples)//trio_batch_size if callTrios else 0}", log, quiet=True)
    
    # Online best-tracking, backed by a scratch SQLite DB (see comment above
    # _open_results_db()) instead of in-memory dicts.
    conn, db_path = _open_results_db()
    GENDERRES = []

    # Start the resource tracker BEFORE the workers fork. It starts lazily on
    # the first SharedMemory registration, which here happens after the Pool
    # exists; workers forked earlier then each start a private tracker when they
    # attach to the segment, and at exit those trackers report it as leaked even
    # though the parent unlinked it ("leaked shared_memory objects" / Errno 2).
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()
    pool = Pool(processes=n_jobs)
    try:
        for p in percentiles:
            _writeout(f"\n====Processing percentile {p}====", log, quiet)
            qualperc = qualityPercentile(qualdist, p, log=None, quiet=True)
            filterargs = {key+'min':val for key,val in qualperc.items()}
            filterObject = Filter(formatinfo=info, PASS=True, **filterargs)
            filterFUN = filterObject.filter()
            
            autosom_filt = {k:list(filter(filterFUN, v)) for k,v in autosom_filt.items()}
            x_filt = {k:list(filter(filterFUN, v)) for k,v in x_filt.items()}
            
            N = sum(len(v) for v in autosom_filt.values()) 
            
            if N == 0: 
                break
    
            # Gender inference
            if callGenders:
                genders = inferGenders(x_filt, info, percentile=p, FEMALE_thresh=FEMALE_thresh, MALE_thresh=MALE_thresh)
                GENDERRES.extend(genders)
            
            # Build genotype matrix
            if callPairs or callTrios:
                _writeout(f"  Building genotype matrix...", log, quiet=True)
                gt_matrix, _ = build_genotype_matrix_optimized(autosom_filt, nSamples)
                _writeout(f"  Matrix shape: {gt_matrix.shape} ({gt_matrix.nbytes / 1024**2:.1f} MB)", log, quiet)
                
                # Create shared memory for the matrix
                shm = shared_memory.SharedMemory(create=True, size=gt_matrix.nbytes)
                shared_array = np.ndarray(gt_matrix.shape, dtype=gt_matrix.dtype, buffer=shm.buf)
                np.copyto(shared_array, gt_matrix)
                
                try:
                    # Pairwise analysis with batch processing
                    if callPairs:
                        pair_start = time.time()
                        
                        # Create batches first
                        pair_batches = [pairs[i:i+pair_batch_size] for i in range(0, len(pairs), pair_batch_size)]
                        
                        _writeout(f"  Processing {len(pairs):,} pairs in {len(pair_batches):,} batches...", log, quiet)
                        
                        # Prepare arguments for workers
                        worker_args = [
                            (shm.name, gt_matrix.shape, gt_matrix.dtype, batch, p, MZ_thresh, PO_thresh)
                            for batch in pair_batches
                        ]
                        
                        if TQDM_AVAILABLE and not quiet:
                            pbar = tqdm(total=len(pair_batches), desc=f"Pair batches (p={p})")
                        
                        # Use adaptive chunksize for better throughput
                        # Larger chunks reduce overhead but we want enough granularity for load balancing
                        chunksize = max(1, len(pair_batches) // (n_jobs * 3))
                        for batch_results in pool.imap_unordered(process_pairs_batch, worker_args, chunksize=chunksize):
                            # Online best-tracking: upsert keeps only the best result per
                            # pair (see _upsert_pairs() / _is_better_pair_result() parity note).
                            _upsert_pairs(conn, batch_results)

                            if TQDM_AVAILABLE and not quiet:
                                pbar.update(1)
                        
                        if TQDM_AVAILABLE and not quiet:
                            pbar.close()
                        
                        pair_time = time.time() - pair_start
                        _writeout(f"  Pair analysis completed in {pair_time:.1f}s ({len(pairs)/pair_time:.0f} pairs/sec)", log, quiet=True)
                    
                    # Trio analysis with batch processing
                    if callTrios:
                        trio_start = time.time()
                        
                        # For very large trio sets, optionally limit the number of batches
                        if max_trio_batches and len(triples) > max_trio_batches * trio_batch_size:
                            _writeout(f"  WARNING: Limiting trio analysis to {max_trio_batches} batches " 
                                    f"(~{max_trio_batches * trio_batch_size:,} trios) due to size", log, quiet)
                            triples_to_process = triples[:max_trio_batches * trio_batch_size]
                        else:
                            triples_to_process = triples
                        
                        # Create batches
                        trio_batches = [triples_to_process[i:i+trio_batch_size] 
                                      for i in range(0, len(triples_to_process), trio_batch_size)]
                        
                        _writeout(f"  Processing {len(triples_to_process):,} trios in {len(trio_batches):,} batches...", log, quiet)
                        
                        # Prepare arguments
                        orders = [(2,0,1), (1,2,0), (0,1,2)]
                        worker_args = [
                            (shm.name, gt_matrix.shape, gt_matrix.dtype, batch, orders, p, T1_thresh, T2_thresh)
                            for batch in trio_batches
                        ]
                        
                        if TQDM_AVAILABLE and not quiet:
                            pbar = tqdm(total=len(trio_batches), desc=f"Trio batches (p={p})")
                        
                        # Use adaptive chunksize for better throughput
                        # Larger chunks reduce overhead but we want enough granularity for load balancing
                        chunksize = max(1, len(trio_batches) // (n_jobs * 3))
                        for batch_results in pool.imap_unordered(process_trios_batch, worker_args, chunksize=chunksize):
                            # Online best-tracking: upsert keeps only the best result per
                            # (triple, pivot) (see _upsert_trios() / _is_better_trio_result()
                            # parity note).
                            _upsert_trios(conn, batch_results)

                            if TQDM_AVAILABLE and not quiet:
                                pbar.update(1)
                        
                        if TQDM_AVAILABLE and not quiet:
                            pbar.close()
                        
                        trio_time = time.time() - trio_start
                        _writeout(f"  Trio analysis completed in {trio_time:.1f}s ({len(triples_to_process)/trio_time:.0f} trios/sec)", log, quiet=True)
                    
                finally:
                    # Clean up shared memory
                    shm.close()
                    shm.unlink()

                # Force garbage collection after each percentile
            del gt_matrix
            gc.collect()
    except Exception:
        # Abnormal exit: kill workers immediately rather than risk hanging
        # on tasks that may never complete.
        pool.terminate()
        pool.join()
        _close_results_db(conn, db_path)
        raise
    else:
        # Normal completion: all imap_unordered calls above already drained
        # every submitted task, so workers are idle here. close()+join() lets
        # them exit cleanly instead of being SIGTERM'd mid-housekeeping,
        # which is what caused sporadic "leaked shared_memory objects"
        # warnings at high -j under the old unconditional terminate().
        pool.close()
        pool.join()

    GENDERRES.sort(key=lambda x: (x['sample'], x['percentile']))

    # Whether ANY pair/triple was ever scored at all (distinct from whether
    # any of them were CONFIRMED relationships) — matches the old
    # `if PAIRRES`/`if TRIORES` truthiness checks, which guarded the
    # "Not enough individuals in the file" fallback message rather than the
    # (separately-filtered) best-call selection below.
    n_pairs_stored = conn.execute('SELECT COUNT(*) FROM best_pairs').fetchone()[0]
    n_trios_stored = conn.execute('SELECT COUNT(*) FROM best_trios').fetchone()[0]

    # Genders are O(N), never a scale concern — kept as a materialized list.
    bestG = bestGenders(GENDERRES)

    # Pairs/trios: no bestP/bestT materialization here. bestPairs()/bestTrios()
    # keep essentially every row when reportall=True (--all), so at cohort
    # scale that list does not fit comfortably in memory. The callPairs/
    # callTrios blocks below stream selection, formatting, file-writing and
    # display in one pass instead (_stream_and_write_best_pairs/trios()).
    # bestP/bestT are only materialized further down, and only if pedfile is
    # set — checkped() needs list/zip semantics, but that path is inherently
    # tiny (exactly 3 samples), so materializing there is free.

    _writeout('\nAll analysis finished.\nTime used: %.2f seconds' % (time.time()-st), log, quiet)

    TEMP = "\n===={heading}====\n{table}\n\nMost confident {analysis} calls written to {outfile}"

    # Write outputs
    if callGenders:
        outfile = "%s.gender" % prefix
        complete = tabulate(gender_table(GENDERRES), tablefmt="plain")
        _writeout(TEMP.format(heading="GENDERS", table=complete, analysis="gender", outfile=outfile), log, quiet=True)
        best_pretty = tabulate(bestG, tablefmt="plain")
        _writeout(TEMP.format(heading="GENDERS", table=best_pretty, analysis="gender", outfile=outfile), log=None, quiet=quiet)
        best_tsv = '\n'.join('\t'.join(map(str, line)) for line in bestG)
        with open(outfile, 'w') as out:
            out.write(best_tsv)

    if callPairs:
        outfile = "%s.pair" % prefix
        # "Best" (filtered) selection FIRST: this is the file downstream
        # tools actually consume, and it's bounded by unique-pair count —
        # far smaller than the full unfiltered accumulation below. Writing
        # it first means it's on disk and usable immediately, rather than
        # waiting on the log-only "complete" dump that nothing downstream
        # reads. Single streaming pass, writes the .pair file and (unless
        # quiet) prints to stdout — see _stream_and_write_best_pairs().
        _stream_and_write_best_pairs(conn, reportall, outfile, quiet, has_data=bool(n_pairs_stored))
        if not quiet:
            print(f"  ({outfile} is complete and safe to read now — "
                  f"the unfiltered --all dump to the log file below can take much longer.)")
        # "Complete" (unfiltered) dump streams directly to the log file
        # instead of going through tabulate() — see _stream_complete_pairs_to_log().
        # Runs LAST, after the useful output already exists on disk.
        _stream_complete_pairs_to_log(conn, log, outfile, has_data=bool(n_pairs_stored))

    if callTrios:
        outfile = "%s.trio" % prefix
        # "Best" (filtered) selection FIRST — see the callPairs comment
        # above for why this ordering matters; same reasoning applies here,
        # and trios are where the raw/filtered size gap (and therefore the
        # wait) is largest (3x per-pivot multiplier under --all).
        _stream_and_write_best_trios(conn, reportall, outfile, quiet, has_data=bool(n_trios_stored))
        if not quiet:
            print(f"  ({outfile} is complete and safe to read now — "
                  f"the unfiltered --all dump to the log file below can take much longer.)")
        # "Complete" (unfiltered) dump streams directly to the log file
        # instead of going through tabulate() — see _stream_complete_trios_to_log().
        # Runs LAST, after the useful output already exists on disk.
        _stream_complete_trios_to_log(conn, log, outfile, has_data=bool(n_trios_stored))

    if pedfile:
        # checkped() needs list/zip semantics; this path only ever runs on
        # an exactly-3-sample dataset (see checkped.checkped()'s own
        # len(samples) != 3 guard), so materializing here is negligible —
        # unlike the general output path above, which avoids this for
        # cohorts where it would not be.
        bestP = bestPairs(_iter_pair_rows(conn), reportall=reportall)
        bestT = bestTrios(_iter_trio_rows(conn), reportall=reportall)
        pedcomp = "OK"
        try:
            checkped.checkped(pedfile, bestG, bestP, bestT, info)
        except RuntimeError as e:
            pedcomp = "FAIL!\n%s" %e
        _writeout("\n====COMPARISON WITH PEDIGREE FILE====\n%s\n" % pedcomp, log, quiet=quiet)

    _close_results_db(conn, db_path)

    if log:
        log.close()
    
    if pedfile:
        return pedcomp=="OK"
    else:
        return None

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to VCF file")
    parser.add_argument("--quiet", help="do not print information on the screen", action='store_true')
    parser.add_argument('--version', action='version', version='VCFped %s'%__version__)
    parser.add_argument("--no-gender", help="skip gender analysis (use for non-human samples or VCFs without X-linked variants)", dest='nogender', action='store_true')
    parser.add_argument("--no-pairwise", help="skip pairwise analysis", dest='nopairwise', action='store_true')
    parser.add_argument("--no-trio", help="skip trio analysis", dest='notrio', action='store_true')
    parser.add_argument("--all", help="show results for all pairs/triples (not only the inferred). "
                        "KNOWN ISSUE: scoring for low-confidence rows (Not trio/Other) can differ "
                        "from vcfped.py under --all (see _is_better_pair_result()/"
                        "_is_better_trio_result() docstrings) -- for a small candidate-subset rerun, "
                        "prefer vcfped.py instead.", dest="reportall", action='store_true')
    parser.add_argument("-j", "--jobs", dest="n_jobs", type=int, help="number of parallel processes (default: all cores, or 25%% of cores if >8 available)", default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="batch size for processing (default: auto)", default=None)
    parser.add_argument("--max-trio-batches", dest="max_trio_batches", type=int, help="max trio batches to limit memory (default: unlimited)", default=None)
    parser.add_argument("-o", help="prefix for output files", dest="prefix")
    parser.add_argument("-ped", help="path to trio pedfile", dest="pedfile")
    parser.add_argument("-v", dest="variables", nargs='+', help="quality variables to be used for filtering", default=['QUAL', 'DP', 'GQ', 'AD'])
    parser.add_argument("-p", dest="percentiles", nargs='+', type=int, help="filtering percentile ranks", default=[10,30,50])
    parser.add_argument("-e", dest='exactmax', type=int, help="if approx. line count exceeds this, apply random sampling", default=150000)
    parser.add_argument("-s", dest='samplesize', type=int, help="sample at least this many variant lines (if sampling)", default=10000)
    parser.add_argument("-d", dest='samplesizeAABB', type=int, help="sample at least this many lines where both 0/0 and 1/1 occur as genotypes (if sampling)", default=500)
    parser.add_argument("-t1", dest="T1_thresh", type=int, help="threshold (%%) for T1 score (AA + BB = AB)", default=90)
    parser.add_argument("-t2", dest="T2_thresh", type=int, help="threshold (%%) for T2 score (BB + BB = BB)", default=95)
    parser.add_argument("-mz", dest="MZ_thresh", type=int, help="threshold (%%) for MZ score (IBS=2 | neither is AA)", default=95)
    parser.add_argument("-po", dest="PO_thresh", type=int, help="threshold (%%) for PO score (IBS>0 | either is BB)", default=99)
    parser.add_argument("-female", dest="FEMALE_thresh", type=int, help="lower limit (%%) for female heterozygosity on X", default=25)
    parser.add_argument("-male", dest="MALE_thresh", type=int, help="upper limit (%%) for male heterozygosity on X", default=5)
    
    args = parser.parse_args(sys.argv[1:])  
    if args.pedfile and args.notrio:
        parser.error("argument --no-trio: Trio analysis cannot be switched off when a pedfile is indicated.")
    if args.pedfile and args.nogender:
        parser.error("argument --no-gender: Gender analysis cannot be switched off when a pedfile is indicated.")
    for pp in args.percentiles:
        if not 0 <= pp < 100: parser.error("argument -p: invalid percentile: %d" %pp)
    thresh_error = "argument -%s/--%s: invalid threshold (must be in [0, 100]): %d"
    if not 0<= args.T1_thresh <=100: parser.error(thresh_error % ('t1','T1_thresh', args.T1_thresh))
    if not 0<= args.T2_thresh <=100: parser.error(thresh_error % ('t2','T2_thresh', args.T2_thresh))
    if not 0<= args.MZ_thresh <=100: parser.error(thresh_error % ('mz','MZ_thresh', args.MZ_thresh))
    if not 0<= args.PO_thresh <=100: parser.error(thresh_error % ('po','PO_thresh', args.PO_thresh))
    if not 0<= args.FEMALE_thresh <=100: parser.error(thresh_error % ('female','FEMALE_thresh', args.FEMALE_thresh))
    if not 0<= args.MALE_thresh <=100: parser.error(thresh_error % ('male','MALE_thresh', args.MALE_thresh))
    if args.MALE_thresh > args.FEMALE_thresh: 
        parser.error("argument -male/--MALE_thresh: invalid threshold (must be below female threshold which is %d): %d" %(args.FEMALE_thresh, args.MALE_thresh))
    
    if args.prefix is None: args.prefix = os.path.splitext(os.path.basename(args.file))[0]
    vcfped(**vars(args))
    
if __name__ == '__main__':
    main()
