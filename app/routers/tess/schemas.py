from pydantic import BaseModel

class TessInferenceFeatures(BaseModel):
    # Stellar Proper Motion
    st_pmra: float  # PMRA [mas/yr]
    st_pmraerr1: float  # PMRA Upper Unc [mas/yr]
    st_pmraerr2: float  # PMRA Lower Unc [mas/yr]
    st_pmdec: float  # PMDec [mas/yr]
    st_pmdecerr1: float  # PMDec Upper Unc [mas/yr]
    st_pmdecerr2: float  # PMDec Lower Unc [mas/yr]
    
    # Orbital and Transit Parameters
    pl_tranmid: float  # Planet Transit Midpoint Value [BJD]
    pl_orbper: float  # Planet Orbital Period Value [days]
    pl_trandurh: float  # Planet Transit Duration Value [hours]
    pl_trandurherr1: float  # Planet Transit Duration Upper Unc [hours]
    pl_trandurherr2: float  # Planet Transit Duration Lower Unc [hours]
    pl_trandep: float  # Planet Transit Depth Value [ppm]
    pl_trandeperr1: float  # Planet Transit Depth Upper Unc [ppm]
    pl_trandeperr2: float  # Planet Transit Depth Lower Unc [ppm]
    
    # Planetary Properties
    pl_rade: float  # Planet Radius Value [R_Earth]
    pl_insol: float  # Planet Insolation Value [Earth flux]
    pl_eqt: float  # Planet Equilibrium Temperature Value [K]
    
    # Stellar Properties
    st_tmag: float  # TESS Magnitude
    st_tmagerr1: float  # TESS Magnitude Upper Unc
    st_tmagerr2: float  # TESS Magnitude Lower Unc
    st_dist: float  # Stellar Distance [pc]
    st_disterr1: float  # Stellar Distance Upper Unc [pc]
    st_disterr2: float  # Stellar Distance Lower Unc [pc]
    st_teff: float  # Stellar Effective Temperature Value [K]
    st_tefferr1: float  # Stellar Effective Temperature Upper Unc [K]
    st_tefferr2: float  # Stellar Effective Temperature Lower Unc [K]
    st_logg: float  # Stellar log(g) Value [cm/s**2]
    st_rad: float  # Stellar Radius Value [R_Sun]
