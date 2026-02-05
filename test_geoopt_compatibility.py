#!/usr/bin/env python3
"""
Test script to verify geoopt 0.5.1 compatibility with the repository.

This script tests all geoopt imports and functionality used across the codebase.
Run this script after installing dependencies to verify geoopt is working correctly.

Usage:
    python test_geoopt_compatibility.py
"""

import sys


def test_geoopt_imports():
    """Test that all required geoopt imports work correctly."""
    print("=" * 70)
    print("Testing geoopt 0.5.1 compatibility")
    print("=" * 70)
    
    # Test 1: Basic geoopt import
    print("\n[1/9] Testing basic geoopt import...")
    try:
        import geoopt
        print(f"    ✓ Successfully imported geoopt version {geoopt.__version__}")
    except ImportError as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 2: Lorentz manifold math functions
    print("\n[2/9] Testing Lorentz math functions...")
    try:
        from geoopt.manifolds.lorentz.math import expmap0
        print("    ✓ Successfully imported expmap0")
    except ImportError as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 3: Riemannian optimizers
    print("\n[3/9] Testing Riemannian optimizers...")
    try:
        from geoopt.optim import RiemannianSGD, RiemannianAdam
        print("    ✓ Successfully imported RiemannianSGD and RiemannianAdam")
    except ImportError as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 4: Manifold classes
    print("\n[4/9] Testing manifold classes...")
    try:
        from geoopt.manifolds import Lorentz as GeooptLorentz
        from geoopt import Stereographic
        from geoopt.manifolds import PoincareBall
        print("    ✓ Successfully imported Lorentz, Stereographic, and PoincareBall")
    except ImportError as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 5: Tensor and Parameter classes
    print("\n[5/9] Testing ManifoldParameter and ManifoldTensor...")
    try:
        from geoopt import ManifoldParameter, ManifoldTensor
        print("    ✓ Successfully imported ManifoldParameter and ManifoldTensor")
    except ImportError as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 6: Create Lorentz manifold
    print("\n[6/9] Testing Lorentz manifold creation...")
    try:
        import torch
        manifold = GeooptLorentz()
        print(f"    ✓ Created Lorentz manifold with curvature k={manifold.k}")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 7: Test manifold operations
    print("\n[7/9] Testing manifold projection operations...")
    try:
        import torch
        manifold = GeooptLorentz()
        x = torch.randn(5, 10)
        x_proj = manifold.projx(x)
        print(f"    ✓ Successfully projected tensor: {x.shape} -> {x_proj.shape}")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 8: Test expmap0 function
    print("\n[8/9] Testing expmap0 function...")
    try:
        import torch
        from geoopt.manifolds.lorentz.math import expmap0
        manifold = GeooptLorentz()
        v = torch.randn(5, 9)
        result = expmap0(v, k=manifold.k)
        print(f"    ✓ expmap0 works: {v.shape} -> {result.shape}")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 9: Test optimizer creation
    print("\n[9/9] Testing RiemannianAdam optimizer creation...")
    try:
        import torch
        import torch.nn as nn
        from geoopt import ManifoldParameter
        from geoopt.manifolds import Lorentz as GeooptLorentz
        from geoopt.optim import RiemannianAdam
        
        manifold = GeooptLorentz()
        tensor = torch.randn(10, 11)
        tensor = manifold.projx(tensor)
        param = ManifoldParameter(tensor, manifold=manifold, requires_grad=True)
        
        model = nn.Module()
        model.register_parameter('hyp_param', param)
        
        optimizer = RiemannianAdam(model.parameters(), lr=0.001)
        print(f"    ✓ Created optimizer with {sum(1 for _ in model.parameters())} parameter(s)")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    return True


def main():
    """Run all tests and report results."""
    success = test_geoopt_imports()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ All tests passed! geoopt 0.5.1 is working correctly.")
        print("=" * 70)
        return 0
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
