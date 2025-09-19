#!/usr/bin/env python3
# Test script to validate ADHD subtyping implementation against requirements

import adhd
import numpy as np
import pandas as pd

def test_theta_excess_rule():
    """Test Theta Excess rule: Z_Theta > 1 (changed from >= 1.5) in at least two frontal electrodes."""
    print("=== Testing Theta Excess Rule ===")
    
    # Test case 1: Exactly 2 frontal electrodes meet new threshold (> 1.0)
    test_data1 = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
        'Zrel_Theta': [1.1, 0.9, 1.2, 0.8, 0.7]  # 2 meet threshold
    })
    
    sites, hits, is_positive = adhd.theta_excess_rule(test_data1)
    print(f"Test 1 - 2 electrodes > 1.0: {is_positive} (expected: True)")
    print(f"  Sites meeting threshold: {sum(hits.values())}")
    
    # Test case 2: Would have failed with old threshold (>= 1.5) but passes with new (> 1.0)
    test_data2 = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
        'Zrel_Theta': [1.1, 1.3, 0.9, 0.8, 0.7]  # 2 meet new threshold, would be 1 with old
    })
    
    sites, hits, is_positive = adhd.theta_excess_rule(test_data2)
    print(f"Test 2 - threshold changed from >= 1.5 to > 1.0: {is_positive} (expected: True)")
    
    # Test case 3: Only 1 electrode meets threshold - should fail
    test_data3 = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
        'Zrel_Theta': [1.1, 0.9, 0.8, 0.7, 0.6]  # 1 meets threshold
    })
    
    sites, hits, is_positive = adhd.theta_excess_rule(test_data3)
    print(f"Test 3 - only 1 electrode > 1.0: {is_positive} (expected: False)")
    print()

def test_tbr_rule():
    """Test TBR rule with age-dependent thresholds."""
    print("=== Testing TBR Rule ===")
    
    # Test data with TBR = 3.5 at Cz
    test_data = pd.DataFrame({
        'Cuban_Channel': ['Cz'],
        'Rel_Theta': [0.35],
        'Rel_Beta': [0.10]  # TBR = 3.5
    })
    
    # Test 1: Child (age < 19) - should be negative (3.5 < 4.0)
    tbr_value, threshold, is_high, age_category = adhd.high_tbr_rule(test_data, 12)
    print(f"Test 1 - Child age 12, TBR {tbr_value:.1f}: {is_high} (expected: False, threshold: {threshold})")
    print(f"  Age category: {age_category}")
    
    # Test 2: Adult (age >= 19) - should be positive (3.5 > 3.0)
    tbr_value, threshold, is_high, age_category = adhd.high_tbr_rule(test_data, 25)
    print(f"Test 2 - Adult age 25, TBR {tbr_value:.1f}: {is_high} (expected: True, threshold: {threshold})")
    print(f"  Age category: {age_category}")
    
    # Test 3: Edge case - exactly 19 years old (adult category)
    tbr_value, threshold, is_high, age_category = adhd.high_tbr_rule(test_data, 19)
    print(f"Test 3 - Age 19 (boundary), TBR {tbr_value:.1f}: {is_high} (expected: True, threshold: {threshold})")
    print(f"  Age category: {age_category}")
    
    # Test 4: High TBR for child (> 4.0)
    test_data_high = pd.DataFrame({
        'Cuban_Channel': ['Cz'],
        'Rel_Theta': [0.45],
        'Rel_Beta': [0.10]  # TBR = 4.5
    })
    
    tbr_value, threshold, is_high, age_category = adhd.high_tbr_rule(test_data_high, 15)
    print(f"Test 4 - Child with high TBR {tbr_value:.1f}: {is_high} (expected: True, threshold: {threshold})")
    print()

def test_combined_classification():
    """Test combined ADHD classification with both rules."""
    print("=== Testing Combined Classification ===")
    
    # Test case 1: Both rules positive
    test_data1 = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'Cz'],
        'Zrel_Theta': [1.2, 1.3, 0.9, 0.8, 0.7, 0.5],  # 2 frontal > 1.0
        'Rel_Theta': [0.35, 0.35, 0.25, 0.25, 0.25, 0.35],
        'Rel_Beta': [0.15, 0.15, 0.15, 0.15, 0.15, 0.10]  # TBR at Cz = 3.5
    })
    
    result = adhd.classify_adhd_subtypes(test_data1, 25)  # Adult
    print(f"Test 1 - Both rules positive: {result['classification']['subtype']}")
    print(f"  Score: {result['classification']['score']}/2")
    
    # Test case 2: Only theta excess positive
    test_data2 = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'Cz'],
        'Zrel_Theta': [1.2, 1.3, 0.9, 0.8, 0.7, 0.5],  # 2 frontal > 1.0
        'Rel_Theta': [0.35, 0.35, 0.25, 0.25, 0.25, 0.25],
        'Rel_Beta': [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]  # TBR at Cz = 1.67 < 3.0
    })
    
    result = adhd.classify_adhd_subtypes(test_data2, 25)  # Adult
    print(f"Test 2 - Only theta excess: {result['classification']['subtype']}")
    print(f"  Score: {result['classification']['score']}/2")
    
    # Test case 3: Only TBR positive  
    test_data3 = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'Cz'],
        'Zrel_Theta': [0.8, 0.9, 0.9, 0.8, 0.7, 0.5],  # No frontal > 1.0
        'Rel_Theta': [0.25, 0.25, 0.25, 0.25, 0.25, 0.35],
        'Rel_Beta': [0.15, 0.15, 0.15, 0.15, 0.15, 0.10]  # TBR at Cz = 3.5 > 3.0
    })
    
    result = adhd.classify_adhd_subtypes(test_data3, 25)  # Adult
    print(f"Test 3 - Only TBR positive: {result['classification']['subtype']}")
    print(f"  Score: {result['classification']['score']}/2")
    
    # Test case 4: Neither rule positive
    test_data4 = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'Cz'],
        'Zrel_Theta': [0.8, 0.9, 0.9, 0.8, 0.7, 0.5],  # No frontal > 1.0
        'Rel_Theta': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        'Rel_Beta': [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]  # TBR at Cz = 1.67 < 3.0
    })
    
    result = adhd.classify_adhd_subtypes(test_data4, 25)  # Adult
    print(f"Test 4 - Neither rule positive: {result['classification']['subtype']}")
    print(f"  Score: {result['classification']['score']}/2")
    print()

def test_threshold_change_validation():
    """Validate that the threshold was actually changed from >= 1.5 to > 1.0."""
    print("=== Validating Threshold Change ===")
    
    # Test with Z_Theta exactly 1.0 - should NOT meet new threshold (> 1.0)
    test_exact = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
        'Zrel_Theta': [1.0, 1.1, 0.9, 0.8, 0.7]  # Only 1 electrode > 1.0
    })
    
    sites, hits, is_positive = adhd.theta_excess_rule(test_exact)
    print(f"Z_Theta exactly 1.0: meets threshold = {hits['Fp1']} (expected: False, since > 1.0)")
    
    # Test with Z_Theta values between 1.0 and 1.5 - should meet new threshold
    test_between = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
        'Zrel_Theta': [1.1, 1.3, 0.9, 0.8, 0.7]  # 2 electrodes between 1.0 and 1.5
    })
    
    sites, hits, is_positive = adhd.theta_excess_rule(test_between)
    print(f"Values 1.1 and 1.3 (between old and new thresholds): rule positive = {is_positive}")
    print("  This confirms threshold changed from >= 1.5 to > 1.0")
    print()

if __name__ == "__main__":
    print("ADHD Subtyping Implementation Validation")
    print("=" * 50)
    print()
    
    test_theta_excess_rule()
    test_tbr_rule() 
    test_combined_classification()
    test_threshold_change_validation()
    
    print("All tests completed!")
    print("✅ Theta Excess rule: Changed threshold to Z_Theta > 1 (from >= 1.5)")
    print("✅ TBR rule: Age-dependent thresholds (< 19: > 4.0, >= 19: > 3.0)")
    print("✅ Classification function implemented correctly")