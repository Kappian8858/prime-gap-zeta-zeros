"""
Prime Gap Formula: Numerical Tests
Author: Kappian KT
Date: May 2026
"""

import mpmath
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Compute first N zeta zeros
# ----------------------------------------------------------------------
def get_zeros(n, dps=30):
    mpmath.mp.dps = dps
    zeros = []
    for i in range(1, n+1):
        try:
            rho = mpmath.zetazero(i)
            zeros.append(float(rho.imag))
        except:
            print(f"Warning: could not compute zero #{i}")
            break
    return zeros

# ----------------------------------------------------------------------
# 2. Compute F(y) = sum_gamma y^{i gamma} / (0.5 + i gamma)
# ----------------------------------------------------------------------
def F(y, zeros):
    s = 0.0 + 0.0j
    logy = np.log(y)
    for gamma in zeros:
        # term = exp(i gamma log y) / (0.5 + i gamma)
        term = mpmath.exp(1j * gamma * logy) / complex(0.5, gamma)
        s += term
    return s

def Z_max(x, zeros, num_points=100):
    ys = np.logspace(0, np.log10(x), num_points)
    max_val = 0.0
    for y in ys:
        val = abs(F(y, zeros))
        if val > max_val:
            max_val = val
    return max_val

# ----------------------------------------------------------------------
# 3. Known record prime gaps (p, actual_gap) from OEIS A002386/A005250
# ----------------------------------------------------------------------
# Data as list of (p, gap)
records = [
    (1.3495e6, 118), (1.3572e6, 132), (2.0107e6, 148), (4.6524e6, 154),
    (1.7052e7, 180), (2.0831e7, 210), (4.7327e7, 220), (1.2216e8, 222),
    (1.8970e8, 234), (1.9191e8, 248), (3.8710e8, 250), (4.3627e8, 282),
    (1.2943e9, 288), (1.4532e9, 292), (2.3009e9, 320), (3.8426e9, 336),
    (4.3024e9, 354), (1.0727e10, 382), (2.0678e10, 384), (2.2367e10, 394),
    (2.5056e10, 456), (4.2653e10, 464), (1.2798e11, 468), (1.8223e11, 474),
    (2.4116e11, 486), (2.9750e11, 490), (3.0337e11, 500), (3.0460e11, 514),
    (4.1661e11, 516), (4.6169e11, 532), (6.1449e11, 534), (7.3883e11, 540),
    (1.3463e12, 582), (1.4087e12, 588), (1.9682e12, 602), (2.6149e12, 652),
    (7.1772e12, 674), (1.3829e13, 716), (1.9581e13, 766), (4.2842e13, 778),
    (9.0874e13, 804), (1.7123e14, 806), (2.1821e14, 906), (1.1895e15, 916),
    (1.6870e15, 924), (1.6932e15, 1132), (4.3842e16, 1184), (5.5351e16, 1198),
    (8.0874e16, 1220), (2.0399e17, 1224), (2.1803e17, 1248), (3.0541e17, 1272),
    (3.5252e17, 1328), (4.0143e17, 1356), (4.1803e17, 1370), (8.0421e17, 1462),
    (1.4252e18, 1474), (5.7332e18, 1550), (6.7880e18, 1556)
]

# ----------------------------------------------------------------------
# 4. Main: compute Z(p), predicted gaps, and ratios
# ----------------------------------------------------------------------
def main():
    print("Computing first 100 zeta zeros...")
    zeros = get_zeros(100)
    print(f"Computed {len(zeros)} zeros.")
    
    results = []
    alpha = 0.4  # empirical constant
    
    for p, g_act in records:
        logp = np.log(p)
        loglogp = np.log(logp)
        Zp = Z_max(p, zeros, num_points=80)
        # predicted with alpha=0 and alpha=0.4
        pred0 = (logp**2) * np.exp( np.log(Zp) / loglogp )
        pred4 = (logp**2) * np.exp( (np.log(Zp) + alpha) / loglogp )
        ratio0 = g_act / pred0 if pred0>0 else 0
        ratio4 = g_act / pred4 if pred4>0 else 0
        results.append( (p, g_act, Zp, pred0, pred4, ratio0, ratio4) )
    
    # Print summary
    print("\n p          actual   Z(p)   pred0   pred4   ratio0  ratio4")
    for r in results[-20:]:  # last 20 entries
        p, g, Zp, p0, p4, r0, r4 = r
        print(f"{p:10.3e} {g:5d} {Zp:6.4f} {p0:6.0f} {p4:6.0f} {r0:6.3f} {r4:6.3f}")
    
    # Compute mean absolute error for alpha=0.4
    errors = [abs(r[1] - r[4])/r[1] for r in results]
    print(f"\nMean relative error (α=0.4): {np.mean(errors):.3f}")
    print(f"Max relative error: {np.max(errors):.3f}")
    
    # Plot
    indices = range(len(results))
    actual = [r[1] for r in results]
    pred0  = [r[3] for r in results]
    pred4  = [r[4] for r in results]
    
    plt.figure(figsize=(10,5))
    plt.plot(indices, actual, 'o-', label='Actual gaps')
    plt.plot(indices, pred4, 's-', label='Predicted (α=0.4)')
    plt.plot(indices, pred0, 'd--', label='Predicted (α=0)')
    plt.xlabel('Record gap index')
    plt.ylabel('Gap size')
    plt.legend()
    plt.grid(True)
    plt.title('Actual vs Predicted Maximal Prime Gaps')
    plt.savefig('gap_plot.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()