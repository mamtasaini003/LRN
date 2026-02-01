"""
Run all LNO vs LRN-FNO comparisons.

This script runs the comparison on all three benchmarks:
1. Burgers 2D
2. Darcy Flow
3. Navier-Stokes 2D

Usage:
    python run_all_comparisons.py
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_all_comparisons(epochs=150):
    """Run all benchmark comparisons."""
    print("=" * 80)
    print("LNO vs LRN-FNO COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print("\nThis will run comparisons on 3 benchmarks:")
    print("  1. 2D Burgers Equation")
    print("  2. Darcy Flow")
    print("  3. 2D Navier-Stokes")
    print("\n" + "=" * 80)
    
    all_results = {}
    total_start = time.time()
    
    # Import and run each comparison
    from lno_burgers2d_demo import compare_burgers2d
    from lno_darcy_demo import compare_darcy
    from lno_ns_demo import compare_navier_stokes
    
    # 1. Burgers 2D
    print("\n" + "=" * 80)
    print("BENCHMARK 1: 2D BURGERS EQUATION")
    print("=" * 80 + "\n")
    all_results['burgers2d'] = compare_burgers2d(epochs=epochs)
    
    # 2. Darcy Flow
    print("\n" + "=" * 80)
    print("BENCHMARK 2: DARCY FLOW")
    print("=" * 80 + "\n")
    all_results['darcy'] = compare_darcy(epochs=epochs)
    
    # 3. Navier-Stokes
    print("\n" + "=" * 80)
    print("BENCHMARK 3: 2D NAVIER-STOKES")
    print("=" * 80 + "\n")
    all_results['navier_stokes'] = compare_navier_stokes(epochs=epochs)
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL BENCHMARKS")
    print("=" * 80)
    
    print(f"\n{'Benchmark':<20} {'FNO':<12} {'LRN-FNO':<12} {'LNO':<12} {'Best':<12}")
    print("-" * 70)
    
    for bench, results in all_results.items():
        fno_err = results['fno']['error']
        lrn_err = results['lrn']['error']
        lno_err = results['lno']['error']
        
        best = 'LRN-FNO' if lrn_err < lno_err else 'LNO'
        
        print(f"{bench:<20} {fno_err:.4f}       {lrn_err:.4f}       {lno_err:.4f}       {best}")
    
    print("\n" + "-" * 70)
    print("\nImprovement Summary (vs FNO baseline):")
    print("-" * 70)
    
    for bench, results in all_results.items():
        fno_err = results['fno']['error']
        lrn_imp = (fno_err - results['lrn']['error']) / fno_err * 100
        lno_imp = (fno_err - results['lno']['error']) / fno_err * 100
        print(f"{bench:<20} LRN-FNO: {lrn_imp:+.2f}%    LNO: {lno_imp:+.2f}%")
    
    print("\n" + "-" * 70)
    print("\nLRN-FNO vs LNO Head-to-Head:")
    print("-" * 70)
    
    lrn_wins = 0
    lno_wins = 0
    
    for bench, results in all_results.items():
        lrn_err = results['lrn']['error']
        lno_err = results['lno']['error']
        diff = (lno_err - lrn_err) / lno_err * 100
        
        if diff > 0:
            print(f"{bench:<20} LRN-FNO wins by {diff:.2f}%")
            lrn_wins += 1
        else:
            print(f"{bench:<20} LNO wins by {-diff:.2f}%")
            lno_wins += 1
    
    print(f"\nOverall: LRN-FNO wins {lrn_wins}/3, LNO wins {lno_wins}/3")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    # Save comprehensive results
    os.makedirs('lno_checkpoints', exist_ok=True)
    
    results_serializable = {}
    for bench, results in all_results.items():
        results_serializable[bench] = {}
        for model, res in results.items():
            results_serializable[bench][model] = {
                'error': float(res['error']),
                'std': float(res['std']),
                'time': float(res['time']),
                'params': int(res['params'])
            }
    
    with open('lno_checkpoints/all_comparison_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\nAll results saved to lno_checkpoints/all_comparison_results.json")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to run per benchmark')
    args = parser.parse_args()
    
    run_all_comparisons(epochs=args.epochs)
