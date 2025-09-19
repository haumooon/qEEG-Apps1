#!/usr/bin/env python3
# ADHD Subtyping Streamlit App
# Implements research-based ADHD subtyping with updated thresholds

import streamlit as st
import io
import os
import numpy as np
import pandas as pd
import tempfile

# Import the ADHD module functions
import adhd

st.set_page_config(page_title="qEEG: ADHD Subtyping", layout="wide")

def main():
    st.title("qEEG ADHD Subtyping Analysis")
    st.markdown("Upload your EDF file and enter patient info. Implements updated research-based ADHD subtyping rules.")

    # Sidebar with rule explanations
    with st.sidebar:
        st.header("ADHD Subtyping Rules")
        st.markdown("**Updated Thresholds:**")
        st.markdown("1. **Theta Excess Rule**")
        st.markdown("   - Z_Theta > 1.0 (changed from ≥1.5)")
        st.markdown("   - At least 2 frontal electrodes")
        st.markdown("")
        st.markdown("2. **TBR Rule (at Cz)**")
        st.markdown("   - Children/Adolescents (<19): TBR > 4.0")
        st.markdown("   - Adults (≥19): TBR > 3.0")

    # Main inputs
    edf_file = st.file_uploader("Select EDF file", type=["edf"])
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=30, step=1)
    p2p = st.number_input("Artifact threshold (µV)", min_value=50.0, max_value=500.0, value=150.0, step=1.0)
    out_docx_name = st.text_input("Output Word report file name (optional)", value="ADHD_Subtyping_Report.docx")
    run_btn = st.button("Run ADHD Analysis", type="primary")

    # Try to load norms file
    try:
        norms_all = pd.read_csv("cuban_norms.csv")
    except Exception:
        try:
            norms_all = pd.read_csv("norms.csv")
        except Exception as e:
            st.error(f"Could not read norms CSV file: {e}")
            st.error("Please ensure 'cuban_norms.csv' or 'norms.csv' is in the app folder.")
            return

    st.write("**Cuban Norms CSV Preview:**")
    st.dataframe(norms_all.head())

    if run_btn and edf_file and age:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(edf_file.read())
                edf_path = tmp_file.name

            with st.spinner("Processing EEG data..."):
                # Check channel mapping
                raw0 = adhd.mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                unmatched = adhd.check_channel_mapping(raw0, norms_all)

                # Get age-appropriate norms
                norms_age = norms_all[norms_all["Age"] == age].copy()
                if norms_age.empty:
                    available_ages = sorted(norms_all["Age"].unique())
                    st.error(f"No norms found for age {age}. Available ages: {available_ages}")
                    return

                # Compute power spectral data
                df_raw = adhd.compute_raw_powers(edf_path)
                df_clean, kept = adhd.compute_clean_powers(edf_path, p2p)
                dfz = adhd.add_zrel_zabs(df_clean, norms_age)

            st.success(f"Analysis completed! Used {kept} clean epochs (artifact threshold: {p2p:.0f} µV)")

            # Show channel mapping issues if any
            if unmatched:
                st.warning("**Channel Mapping Issues:**")
                for orig, mapped in unmatched:
                    st.write(f"EDF: {orig} -> Normalized: {mapped} (not found in norms)")
            else:
                st.success("All EDF channels matched Cuban norms.")

            # Display topographic maps
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RAW Relative Power Maps")
                for b in ["Delta", "Theta", "Alpha", "Beta"]:
                    png = adhd._topomap_png(df_raw, f"Rel_{b}", f"RAW Relative – {b}")
                    st.image(png, caption=f"RAW Relative – {b}", use_column_width=True)

            with col2:
                st.subheader("Z-score Maps (Cuban Norms)")
                for b in ["Delta", "Theta", "Alpha", "Beta"]:
                    png = adhd._topomap_png(dfz, f"Zrel_{b}", f"Z – {b} (Age {age})")
                    st.image(png, caption=f"Z (Cuban) – {b}", use_column_width=True)

            # ADHD Subtyping Analysis
            st.header("ADHD Subtyping Analysis")
            
            adhd_results = adhd.classify_adhd_subtypes(dfz, age)

            # Theta Excess Rule Results
            st.subheader("1. Theta Excess Rule")
            theta_res = adhd_results["theta_excess"]
            
            st.write(f"**Rule:** Z_Theta > 1.0 in at least 2 frontal electrodes")
            st.write(f"**Frontal electrodes:** {', '.join(adhd.REGIONS['Frontal'])}")
            
            # Create a DataFrame for better display
            frontal_data = []
            for ch in adhd.REGIONS["Frontal"]:
                z_val = theta_res["sites"][ch]
                hit = theta_res["hits"][ch]
                if np.isfinite(z_val):
                    frontal_data.append({
                        "Electrode": ch,
                        "Z_Theta": f"{z_val:.2f}",
                        "Meets Threshold": "✓" if hit else "✗"
                    })
                else:
                    frontal_data.append({
                        "Electrode": ch,
                        "Z_Theta": "N/A",
                        "Meets Threshold": "N/A"
                    })
            
            frontal_df = pd.DataFrame(frontal_data)
            st.dataframe(frontal_df, use_container_width=True)
            
            st.write(f"**Electrodes meeting threshold:** {theta_res['count']}/5")
            
            if theta_res['is_positive']:
                st.success("✅ **POSITIVE** for Theta Excess")
            else:
                st.error("❌ **NEGATIVE** for Theta Excess")

            # TBR Rule Results
            st.subheader("2. Theta/Beta Ratio (TBR) Rule")
            tbr_res = adhd_results["tbr"]
            
            st.write(f"**Age category:** {tbr_res['age_category']} (Age: {age})")
            st.write(f"**Threshold:** TBR > {tbr_res['threshold']:.1f} at Cz")
            
            if np.isfinite(tbr_res["value"]):
                st.write(f"**TBR at Cz:** {tbr_res['value']:.2f}")
                
                # Show TBR components
                theta_cz, beta_cz, _ = adhd.compute_tbr_at_cz(dfz)
                st.write(f"  - Theta at Cz: {theta_cz:.4f}")
                st.write(f"  - Beta at Cz: {beta_cz:.4f}")
            else:
                st.write("**TBR at Cz:** N/A (insufficient data)")
            
            if tbr_res['is_high']:
                st.success("✅ **POSITIVE** for High TBR")
            else:
                st.error("❌ **NEGATIVE** for High TBR")

            # Final Classification
            st.subheader("🎯 ADHD Subtype Classification")
            classification = adhd_results["classification"]
            
            st.markdown(f"### {classification['subtype']}")
            st.write(f"**EEG Marker Score:** {classification['score']}/2")
            
            # Progress bar for score
            progress_value = classification['score'] / 2
            st.progress(progress_value)
            
            # Color-coded result
            if classification['score'] == 2:
                st.success("🔴 **High confidence:** Both EEG markers present")
            elif classification['score'] == 1:
                st.warning("🟡 **Moderate confidence:** One EEG marker present")
            else:
                st.info("⚪ **Low confidence:** No EEG markers detected")

            # Clinical Notes
            st.subheader("📋 Clinical Notes")
            st.info("""
            **Important:** These EEG markers are research-based indicators and should be interpreted 
            in conjunction with comprehensive clinical assessment. The updated thresholds reflect 
            recent research findings:
            
            - **Theta Excess threshold lowered** from Z ≥ 1.5 to Z > 1.0 for increased sensitivity
            - **Age-dependent TBR thresholds** account for developmental changes in EEG patterns
            """)

            # Generate Word Report
            st.subheader("📄 Export Report")
            
            with st.spinner("Generating Word report..."):
                # Create Word document
                doc = adhd.Document()
                doc.add_heading("qEEG ADHD Subtyping Report", level=0)
                doc.add_paragraph(f"Patient Age: {age}")
                doc.add_paragraph(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                doc.add_paragraph(f"Artifact threshold: {p2p:.0f} µV | Clean epochs used: {kept}")
                doc.add_paragraph("")

                # Add classification results
                doc.add_heading("ADHD Subtype Classification", level=1)
                doc.add_paragraph(f"Final Classification: {classification['subtype']}")
                doc.add_paragraph(f"EEG Marker Score: {classification['score']}/2")
                doc.add_paragraph("")

                # Theta Excess details
                doc.add_heading("1. Theta Excess Rule", level=2)
                doc.add_paragraph("Rule: Z_Theta > 1.0 in at least 2 frontal electrodes")
                for ch in adhd.REGIONS["Frontal"]:
                    z_val = theta_res["sites"][ch]
                    hit = theta_res["hits"][ch]
                    if np.isfinite(z_val):
                        doc.add_paragraph(f"  {ch}: {z_val:.2f} {'✓' if hit else '✗'}")
                    else:
                        doc.add_paragraph(f"  {ch}: N/A")
                doc.add_paragraph(f"Result: {'POSITIVE' if theta_res['is_positive'] else 'NEGATIVE'}")
                doc.add_paragraph("")

                # TBR details  
                doc.add_heading("2. Theta/Beta Ratio (TBR) Rule", level=2)
                doc.add_paragraph(f"Age category: {tbr_res['age_category']} (Age: {age})")
                doc.add_paragraph(f"Threshold: TBR > {tbr_res['threshold']:.1f} at Cz")
                if np.isfinite(tbr_res["value"]):
                    doc.add_paragraph(f"TBR at Cz: {tbr_res['value']:.2f}")
                else:
                    doc.add_paragraph("TBR at Cz: N/A")
                doc.add_paragraph(f"Result: {'POSITIVE' if tbr_res['is_high'] else 'NEGATIVE'}")

                # Add brain maps to document
                doc.add_heading("EEG Topographic Maps", level=1)
                
                # Raw maps
                doc.add_heading("Raw Relative Power", level=2)
                raw_items = [(f"RAW – {b}", adhd._topomap_png(df_raw, f"Rel_{b}", f"RAW – {b}")) 
                            for b in ["Delta","Theta","Alpha","Beta"]]
                adhd.make_two_per_row_section(doc, raw_items)
                
                # Z maps
                doc.add_heading("Z-score Maps (Cuban Norms)", level=2)
                z_items = [(f"Z – {b}", adhd._topomap_png(dfz, f"Zrel_{b}", f"Z – {b} (Age {age})")) 
                          for b in ["Delta","Theta","Alpha","Beta"]]
                adhd.make_two_per_row_section(doc, z_items)

                # Save to bytes buffer
                buf = io.BytesIO()
                doc.save(buf)
                buf.seek(0)

            # Download button
            st.download_button(
                label="📥 Download ADHD Report (Word)",
                data=buf.getvalue(),
                file_name=out_docx_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            # Clean up temporary file
            try:
                os.unlink(edf_path)
            except:
                pass

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            # Clean up temporary file on error
            try:
                os.unlink(edf_path)
            except:
                pass

if __name__ == "__main__":
    main()