#!/usr/bin/env python3
# Demo script showing ADHD subtyping analysis with realistic example data

import adhd
import numpy as np
import pandas as pd

def create_demo_data():
    """Create realistic demo EEG data for ADHD analysis."""
    # Simulate a complete EEG channel set with realistic Z-scores and relative power values
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'C3', 'C4', 'Cz', 'P3', 'P4', 'Pz', 'O1', 'O2']
    
    # Create example data for a case with ADHD markers
    demo_data = pd.DataFrame({
        'Cuban_Channel': channels,
        # Z-scores for Theta band - elevated in frontal regions (ADHD pattern)
        'Zrel_Theta': [1.3, 1.1, 1.4, 0.9, 1.2, 0.6, 0.7, 0.8, 0.5, 0.4, 0.6, 0.3, 0.2],
        # Relative power values for Theta (higher in ADHD)
        'Rel_Theta': [0.32, 0.30, 0.34, 0.27, 0.31, 0.24, 0.25, 0.40, 0.22, 0.21, 0.23, 0.18, 0.16],
        # Relative power values for Beta (reduced in ADHD, especially centrally)
        'Rel_Beta': [0.18, 0.19, 0.17, 0.20, 0.18, 0.16, 0.17, 0.12, 0.15, 0.16, 0.14, 0.13, 0.14]
    })
    
    return demo_data

def demonstrate_adhd_analysis():
    """Demonstrate ADHD subtyping analysis with example data."""
    print("ADHD Subtyping Analysis - Clinical Demonstration")
    print("=" * 55)
    print()
    
    # Create demo data
    demo_data = create_demo_data()
    
    print("Example EEG Data (Simulated):")
    print("-" * 30)
    print("Frontal electrodes Z_Theta values:")
    frontal_electrodes = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']
    for electrode in frontal_electrodes:
        z_theta = demo_data[demo_data['Cuban_Channel'] == electrode]['Zrel_Theta'].iloc[0]
        meets_threshold = "✓" if z_theta > 1.0 else "✗"
        print(f"  {electrode}: {z_theta:.1f} {meets_threshold}")
    
    cz_theta = demo_data[demo_data['Cuban_Channel'] == 'Cz']['Rel_Theta'].iloc[0]
    cz_beta = demo_data[demo_data['Cuban_Channel'] == 'Cz']['Rel_Beta'].iloc[0]
    tbr_cz = cz_theta / cz_beta
    print(f"\nTBR at Cz: {cz_theta:.2f} / {cz_beta:.2f} = {tbr_cz:.2f}")
    print()
    
    # Test with different ages
    ages_to_test = [8, 15, 19, 25, 45]
    
    for age in ages_to_test:
        print(f"Analysis for Age {age} years:")
        print("-" * 25)
        
        result = adhd.classify_adhd_subtypes(demo_data, age)
        
        # Theta Excess Rule Results
        theta_result = result['theta_excess']
        print(f"1. Theta Excess Rule:")
        print(f"   Frontal sites with Z_Theta > 1.0: {theta_result['count']}/5")
        print(f"   Result: {'POSITIVE' if theta_result['is_positive'] else 'NEGATIVE'}")
        
        # TBR Rule Results  
        tbr_result = result['tbr']
        print(f"2. TBR Rule:")
        print(f"   Age category: {tbr_result['age_category']}")
        print(f"   TBR threshold: > {tbr_result['threshold']:.1f}")
        print(f"   TBR value: {tbr_result['value']:.2f}")
        print(f"   Result: {'POSITIVE' if tbr_result['is_high'] else 'NEGATIVE'}")
        
        # Final Classification
        classification = result['classification']
        print(f"3. Final Classification:")
        print(f"   Subtype: {classification['subtype']}")
        print(f"   EEG Marker Score: {classification['score']}/2")
        
        if classification['score'] == 2:
            confidence = "HIGH"
        elif classification['score'] == 1:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        print(f"   Confidence: {confidence}")
        print()

def compare_old_vs_new_thresholds():
    """Demonstrate the impact of changing theta excess threshold."""
    print("Impact of Updated Theta Excess Threshold")
    print("=" * 42)
    print()
    
    # Create test case where old threshold (>= 1.5) would miss cases
    test_data = pd.DataFrame({
        'Cuban_Channel': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
        'Zrel_Theta': [1.1, 1.3, 0.9, 0.8, 0.7]  # 2 electrodes between 1.0-1.5
    })
    
    # Simulate old threshold behavior (>= 1.5)
    old_hits = {ch: (val >= 1.5) for ch, val in zip(test_data['Cuban_Channel'], test_data['Zrel_Theta'])}
    old_count = sum(old_hits.values())
    old_positive = old_count >= 2
    
    # New threshold behavior (> 1.0)
    sites, hits, new_positive = adhd.theta_excess_rule(test_data)
    new_count = sum(hits.values())
    
    print("Example case with Z_Theta values: [1.1, 1.3, 0.9, 0.8, 0.7]")
    print()
    print("Old threshold (>= 1.5):")
    print(f"  Sites meeting threshold: {old_count}/5")
    print(f"  Rule result: {'POSITIVE' if old_positive else 'NEGATIVE'}")
    print()
    print("New threshold (> 1.0):")
    print(f"  Sites meeting threshold: {new_count}/5") 
    print(f"  Rule result: {'POSITIVE' if new_positive else 'NEGATIVE'}")
    print()
    print("Impact: The updated threshold increases sensitivity by detecting")
    print("        more subtle theta excess patterns that may be clinically relevant.")
    print()

if __name__ == "__main__":
    demonstrate_adhd_analysis()
    compare_old_vs_new_thresholds()
    
    print("Key Implementation Features:")
    print("✅ Theta Excess threshold changed from >= 1.5 to > 1.0")
    print("✅ Age-dependent TBR thresholds (Child: >4.0, Adult: >3.0)")
    print("✅ Comprehensive classification with confidence scoring")
    print("✅ Compatible with existing qEEG analysis framework")