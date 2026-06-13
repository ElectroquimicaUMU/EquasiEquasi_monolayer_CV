import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

from main import CVsim, FRT

st.set_page_config(layout="wide")
st.title("Cyclic Voltammetry — EirreEirre / EquasiEquasi")

with st.sidebar:
    plot_selection = st.selectbox("Plot", ["Both", "MH only", "BV only"])
    surface_model = st.selectbox("Surface excess model", ["MH", "BV"])
    mechanism = st.selectbox("Mechanism", ["EirreEirre", "EquasiEquasi"])

    lambda1 = st.number_input("λ (eV)", value=0.5, step=0.01)
    k01 = st.number_input("k01 (s⁻¹)", value=0.1, step=0.01)
    k02 = st.number_input("k02 (s⁻¹)", value=0.1, step=0.01)
    E02 = st.number_input("E02 (V)", value=-0.25, step=0.001)
    alpha = st.number_input("α", value=0.5, step=0.01)
    Es = st.number_input("Es (V)", value=1e-3, step=1e-4, format="%.4f")
    Ein = st.number_input("Ein (V)", value=0.25, step=0.001)
    Efin = st.number_input("Efin (V)", value=-0.7, step=0.001)
    rate = st.number_input("Scan rate (V/s)", value=0.1, step=0.01)

res = CVsim(
    lambda1 * FRT,
    k01,
    k02,
    E02,
    alpha,
    Es,
    Ein,
    Efin,
    rate,
    surface_model=surface_model,
    mechanism=mechanism,
)

E = res.Pot[1:]
show_mh = plot_selection in ["Both", "MH only"]
show_bv = plot_selection in ["Both", "BV only"]

st.subheader("Ψ response")
fig, ax = plt.subplots()
if show_mh:
    ax.plot(E, res.IntMH[1:], label="MH")
if show_bv:
    ax.plot(E, res.IntBV[1:], "--", label="BV")
ax.set_xlabel("E / V")
ax.set_ylabel("Ψ")
ax.legend()
st.pyplot(fig)

st.subheader(f"Surface excesses ({surface_model}, {mechanism})")
fig, ax = plt.subplots()
ax.plot(E, res.fO[1:], label="fO")
ax.plot(E, res.fR[1:], label="fR")
ax.plot(E, res.fI[1:], label="fI")
ax.set_xlabel("E / V")
ax.set_ylabel("Surface excess")
ax.legend()
st.pyplot(fig)

st.subheader("Rate constants")
fig, ax = plt.subplots()
if show_mh:
    ax.semilogy(E, res.kMH1red_s[1:], label="MH k_red 1")
    ax.semilogy(E, res.kMH2red_s[1:], label="MH k_red 2")
    ax.semilogy(E, res.kMH1ox_s[1:], label="MH k_ox 1")
    ax.semilogy(E, res.kMH2ox_s[1:], label="MH k_ox 2")
if show_bv:
    ax.semilogy(E, res.kBV1red_s[1:], "--", label="BV k_red 1")
    ax.semilogy(E, res.kBV2red_s[1:], "--", label="BV k_red 2")
    ax.semilogy(E, res.kBV1ox_s[1:], "--", label="BV k_ox 1")
    ax.semilogy(E, res.kBV2ox_s[1:], "--", label="BV k_ox 2")
ax.set_xlabel("E / V")
ax.set_ylabel("k (s⁻¹)")
ax.legend()
st.pyplot(fig)

st.subheader("Detected peak coordinates")
rows = []
for mdl, peaks in res.peaks.items():
    if (mdl == "MH" and not show_mh) or (mdl == "BV" and not show_bv):
        continue
    for n in [1, 2]:
        if f"E_peak{n}" in peaks:
            rows.append(
                {
                    "Model": mdl,
                    "Peak": n,
                    "E (V)": peaks[f"E_peak{n}"],
                    "I": peaks[f"I_peak{n}"],
                }
            )

if rows:
    st.dataframe(pd.DataFrame(rows))
else:
    st.info("No local peaks detected.")

st.subheader("Download results as .txt")


def download_txt(label, filename, header, data):
    buf = io.StringIO()
    np.savetxt(buf, data, header=header)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/plain")


if show_mh:
    download_txt(
        "Download MH voltammogram",
        "MH_curve.txt",
        "E (V)\tPsi_MH",
        np.column_stack((E, res.IntMH[1:])),
    )

if show_bv:
    download_txt(
        "Download BV voltammogram",
        "BV_curve.txt",
        "E (V)\tPsi_BV",
        np.column_stack((E, res.IntBV[1:])),
    )

download_txt(
    "Download surface excesses",
    f"surface_excess_{surface_model}_{mechanism}.txt",
    "E (V)\tfO\tfR\tfI",
    np.column_stack((E, res.fO[1:], res.fR[1:], res.fI[1:])),
)

if show_mh:
    download_txt(
        "Download MH rate constants",
        "MH_rates.txt",
        "E (V)\tk_red1\tk_red2\tk_ox1\tk_ox2",
        np.column_stack(
            (
                E,
                res.kMH1red_s[1:],
                res.kMH2red_s[1:],
                res.kMH1ox_s[1:],
                res.kMH2ox_s[1:],
            )
        ),
    )

if show_bv:
    download_txt(
        "Download BV rate constants",
        "BV_rates.txt",
        "E (V)\tk_red1\tk_red2\tk_ox1\tk_ox2",
        np.column_stack(
            (
                E,
                res.kBV1red_s[1:],
                res.kBV2red_s[1:],
                res.kBV1ox_s[1:],
                res.kBV2ox_s[1:],
            )
        ),
    )
