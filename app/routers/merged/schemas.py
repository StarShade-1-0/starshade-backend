from pydantic import BaseModel

class MergedInferenceFeatures(BaseModel):
    orbital_period: float
    transit_duration: float
    transit_duration_err1: float
    transit_duration_err2: float
    transit_depth: float
    transit_depth_err1: float
    transit_depth_err2: float
    planet_radius: float
    planet_radius_err1: float
    planet_radius_err2: float
    equi_temp: float
    stellar_temp: float
    stellar_temp_err1: float
    stellar_temp_err2: float
    stellar_radius: float
    stellar_radius_err1: float
    stellar_radius_err2: float
