from pydantic import BaseModel

class K2InferenceFeatures(BaseModel):
    pl_orbper: float
    pl_tranmid: float
    pl_trandur: float
    pl_rade: float
    pl_radj: float
    pl_radjerr1: float
    pl_radjerr2: float
    pl_ratror: float
    st_rad: float
    st_raderr1: float
    st_raderr2: float
    sy_dist: float
    sy_disterr1: float
    sy_disterr2: float
    sy_plx: float
    sy_plxerr1: float
    sy_plxerr2: float
