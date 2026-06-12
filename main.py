import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from collections import namedtuple

FRT = 38.923074
INTEGRATION_LIMIT = 200

CVResult = namedtuple(
    "CVResult",
    [
        "Pot",
        "IntMH",
        "IntBV",
        "kMH1red_s",
        "kMH2red_s",
        "kMH1ox_s",
        "kMH2ox_s",
        "kBV1red_s",
        "kBV2red_s",
        "kBV1ox_s",
        "kBV2ox_s",
        "fO",
        "fR",
        "fI",
        "peaks",
    ],
)


def _find_peaks(E, I):
    E = np.asarray(E)
    I = np.asarray(I)

    idx_max, _ = find_peaks(I)
    idx_min, _ = find_peaks(-I)

    idx_all = np.unique(np.concatenate([idx_max, idx_min]))
    if idx_all.size == 0:
        return {}

    order = np.argsort(np.abs(I[idx_all]))[::-1]
    idx_all = idx_all[order]

    peaks = {}
    for n, idx in enumerate(idx_all[:2], start=1):
        peaks[f"E_peak{n}"] = E[idx]
        peaks[f"I_peak{n}"] = I[idx]

    return peaks


def _mh_integral(lmbda, nu):
    return integrate.quad(
        lambda x: np.exp(-lmbda / 4 * (1 + (nu + x) / lmbda) ** 2)
        / (1 + np.exp(-x)),
        -INTEGRATION_LIMIT,
        INTEGRATION_LIMIT,
    )[0]


def CVsim(
    lambda1,
    k01,
    k02,
    E02,
    alpha,
    Es,
    Ein,
    Efin,
    rate,
    surface_model="MH",
    mechanism="EirreEirre",
):
    lambda2 = lambda1
    tau = Es / rate

    n_half = int(round(abs(Efin - Ein) / Es))
    nt = 2 * n_half + 1

    Pot = [Ein] * nt

    kMH1red = [0.0] * nt
    kMH2red = [0.0] * nt
    kMH1ox = [0.0] * nt
    kMH2ox = [0.0] * nt

    kBV1red = [0.0] * nt
    kBV2red = [0.0] * nt
    kBV1ox = [0.0] * nt
    kBV2ox = [0.0] * nt

    IntMH = [0.0] * nt
    IntBV = [0.0] * nt

    fO_MH, fI_MH, fR_MH = [1.0] * nt, [0.0] * nt, [0.0] * nt
    fO_BV, fI_BV, fR_BV = [1.0] * nt, [0.0] * nt, [0.0] * nt

    S01 = _mh_integral(lambda1, 0.0)
    S02 = _mh_integral(lambda2, 0.0)

    for i in range(1, nt):
        Pot[i] = Pot[i - 1] - Es if i <= n_half else Pot[i - 1] + Es

        nu1 = FRT * Pot[i]
        nu2 = FRT * (Pot[i] - E02)

        MH1 = _mh_integral(lambda1, nu1)
        MH2 = _mh_integral(lambda2, nu2)

        kMH1red[i] = k01 * tau * MH1 / S01
        kMH2red[i] = k02 * tau * MH2 / S02
        kMH1ox[i] = kMH1red[i] * np.exp(nu1)
        kMH2ox[i] = kMH2red[i] * np.exp(nu2)

        kBV1red[i] = k01 * tau * np.exp(-alpha * nu1)
        kBV2red[i] = k02 * tau * np.exp(-alpha * nu2)
        kBV1ox[i] = kBV1red[i] * np.exp(nu1)
        kBV2ox[i] = kBV2red[i] * np.exp(nu2)

        if mechanism == "EirreEirre":
            fO_MH[i] = fO_MH[i - 1] * np.exp(-kMH1red[i])
            den = kMH1red[i] - kMH2red[i] if abs(kMH1red[i] - kMH2red[i]) > 1e-15 else 1e-15
            fR_MH[i] = (
                1
                + (fR_MH[i - 1] - 1) * np.exp(-kMH2red[i])
                + fO_MH[i - 1]
                * (np.exp(-kMH1red[i]) - np.exp(-kMH2red[i]))
                * kMH2red[i]
                / den
            )
            fI_MH[i] = 1 - fO_MH[i] - fR_MH[i]
            IntMH[i] = (fO_MH[i] * kMH1red[i] + fI_MH[i] * kMH2red[i]) / Es / FRT

            fO_BV[i] = fO_BV[i - 1] * np.exp(-kBV1red[i])
            denBV = kBV1red[i] - kBV2red[i] if abs(kBV1red[i] - kBV2red[i]) > 1e-15 else 1e-15
            fR_BV[i] = (
                1
                + (fR_BV[i - 1] - 1) * np.exp(-kBV2red[i])
                + fO_BV[i - 1]
                * (np.exp(-kBV1red[i]) - np.exp(-kBV2red[i]))
                * kBV2red[i]
                / denBV
            )
            fI_BV[i] = 1 - fO_BV[i] - fR_BV[i]
            IntBV[i] = (fO_BV[i] * kBV1red[i] + fI_BV[i] * kBV2red[i]) / Es / FRT

        else:
            def mh_ode(_, y):
                fO, fR = y
                fI = 1 - fO - fR
                return [
                    -kMH1red[i] * fO + kMH1ox[i] * fI,
                    kMH2red[i] * fI - kMH2ox[i] * fR,
                ]

            sol_mh = solve_ivp(
                mh_ode,
                [0, 1],
                [fO_MH[i - 1], fR_MH[i - 1]],
                t_eval=[1],
            )

            fO_MH[i] = sol_mh.y[0, -1]
            fR_MH[i] = sol_mh.y[1, -1]
            fI_MH[i] = 1 - fO_MH[i] - fR_MH[i]

            IntMH[i] = (
                fO_MH[i] * kMH1red[i]
                + fI_MH[i] * kMH2red[i]
                - fI_MH[i] * kMH1ox[i]
                - fR_MH[i] * kMH2ox[i]
            ) / Es / FRT

            def bv_ode(_, y):
                fO, fR = y
                fI = 1 - fO - fR
                return [
                    -kBV1red[i] * fO + kBV1ox[i] * fI,
                    kBV2red[i] * fI - kBV2ox[i] * fR,
                ]

            sol_bv = solve_ivp(
                bv_ode,
                [0, 1],
                [fO_BV[i - 1], fR_BV[i - 1]],
                t_eval=[1],
            )

            fO_BV[i] = sol_bv.y[0, -1]
            fR_BV[i] = sol_bv.y[1, -1]
            fI_BV[i] = 1 - fO_BV[i] - fR_BV[i]

            IntBV[i] = (
                fO_BV[i] * kBV1red[i]
                + fI_BV[i] * kBV2red[i]
                - fI_BV[i] * kBV1ox[i]
                - fR_BV[i] * kBV2ox[i]
            ) / Es / FRT

    Pot = np.asarray(Pot)
    E = Pot[1:]

    kMH1red_s = np.asarray(kMH1red) / tau
    kMH2red_s = np.asarray(kMH2red) / tau
    kMH1ox_s = np.asarray(kMH1ox) / tau
    kMH2ox_s = np.asarray(kMH2ox) / tau

    kBV1red_s = np.asarray(kBV1red) / tau
    kBV2red_s = np.asarray(kBV2red) / tau
    kBV1ox_s = np.asarray(kBV1ox) / tau
    kBV2ox_s = np.asarray(kBV2ox) / tau

    peaks = {
        "MH": _find_peaks(E, IntMH[1:]),
        "BV": _find_peaks(E, IntBV[1:]),
    }

    if surface_model.upper() == "MH":
        fO, fR, fI = fO_MH, fR_MH, fI_MH
    else:
        fO, fR, fI = fO_BV, fR_BV, fI_BV

    return CVResult(
        Pot,
        np.asarray(IntMH),
        np.asarray(IntBV),
        kMH1red_s,
        kMH2red_s,
        kMH1ox_s,
        kMH2ox_s,
        kBV1red_s,
        kBV2red_s,
        kBV1ox_s,
        kBV2ox_s,
        np.asarray(fO),
        np.asarray(fR),
        np.asarray(fI),
        peaks,
    )