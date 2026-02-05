# geoopt Compatibility Fix

## Problem

The repository required `geoopt==0.5.0` which is incompatible with scipy versions >= 1.15.0. The incompatibility was caused by scipy removing deprecated functions (`scalar_search_wolfe2` and `scalar_search_armijo`) that geoopt 0.5.0's line search optimizer relied on.

## Solution

Updated `geoopt` from version 0.5.0 to 0.5.1 in `HyperCore/requirements.txt`. Version 0.5.1 is compatible with modern scipy versions (tested with scipy 1.17.0).

## Changes Made

1. **HyperCore/requirements.txt**: Updated `geoopt==0.5.0` to `geoopt==0.5.1`
2. **.gitignore**: Added file to ignore `__pycache__/` and `*.pyc` files
3. **test_geoopt_compatibility.py**: Added comprehensive test script to verify geoopt functionality

## Verification

All geoopt imports and functionality used throughout the repository have been tested and verified:

- ✓ `geoopt.manifolds.lorentz.math.expmap0` - Used in hyperbolic_mapping and hyperbolic_fully
- ✓ `geoopt.optim.RiemannianAdam` - Used in tokenizer
- ✓ `geoopt.optim.RiemannianSGD` - Used in tokenizer  
- ✓ `geoopt.manifolds.Lorentz` - Used in tokenizer
- ✓ `geoopt.ManifoldParameter` - Used in HyperCore and hyperbolic_fully
- ✓ `geoopt.Stereographic` - Used in HyperCore
- ✓ `geoopt.manifolds.PoincareBall` - Used in HyperCore

## Testing

Run the compatibility test script to verify geoopt is working:

```bash
python test_geoopt_compatibility.py
```

Expected output:
```
======================================================================
Testing geoopt 0.5.1 compatibility
======================================================================
...
✓ All tests passed! geoopt 0.5.1 is working correctly.
======================================================================
```

## Installation

To install the correct version of geoopt:

```bash
pip install geoopt==0.5.1
```

Or install all HyperCore requirements:

```bash
cd HyperCore
pip install -r requirements.txt
```

## Compatibility

- **geoopt**: 0.5.1
- **scipy**: >= 1.17.0 (tested)
- **torch**: >= 2.0.1 (tested with 2.10.0)
- **Python**: 3.9+

## References

- geoopt repository: https://github.com/geoopt/geoopt
- scipy changelog: https://scipy.github.io/devdocs/release/
