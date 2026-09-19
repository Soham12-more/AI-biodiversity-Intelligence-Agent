from __future__ import annotations

from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]


class SiteProfile(BaseModel):
    """Everything the system knows about one site. All fields optional: the dialogue fills them over turns."""

    # soil
    soc_pct: Optional[float] = Field(None, ge=0, le=60, description="Soil organic carbon, % by mass")
    ph: Optional[float] = Field(None, ge=2, le=11)
    soil_moisture_class: Optional[Level] = None
    # climate
    rainfall_mm: Optional[float] = Field(None, ge=0, le=12000, description="Mean annual rainfall")
    rainfall_class: Optional[Level] = None
    mean_temp_c: Optional[float] = Field(None, ge=-30, le=45)
    climate: Optional[Literal["arid", "semi-arid", "sub-humid", "humid", "temperate", "tropical"]] = None
    # land use / land cover
    land_use: Optional[Literal["cropland", "pasture", "forest", "degraded", "urban", "wetland"]] = None
    crop: Optional[str] = None
    crop_system: Optional[Literal["monoculture", "rotation", "intercrop"]] = None
    native_habitat_pct: Optional[float] = Field(None, ge=0, le=100)
    # biodiversity
    species_richness: Optional[int] = Field(None, ge=0)
    biodiversity_trend: Optional[Literal["declining", "stable", "increasing"]] = None
    fragmentation: Optional[Level] = None
    # human impact
    pesticide_use: Optional[Level] = None
    pollution: Optional[Level] = None
    deforestation: Optional[bool] = None
    near_water_body: Optional[bool] = None
    # spatial
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    region: Optional[str] = None
    # provenance: field -> "user" | "soilgrids" | "open-meteo"
    provenance: dict[str, str] = Field(default_factory=dict)

    # the fields that count as "environmental variables" for the >=3-variable rule
    CORE: ClassVar[tuple[str, ...]] = (
        "soc_pct", "ph", "soil_moisture_class", "rainfall_mm", "rainfall_class", "mean_temp_c",
        "climate", "land_use", "crop_system", "native_habitat_pct", "species_richness",
        "fragmentation", "pesticide_use", "pollution", "deforestation",
    )

    def known(self) -> list[str]:
        return [f for f in self.CORE if getattr(self, f) is not None]

    def variable_groups(self) -> set[str]:
        """Distinct variable families known (soil / water-climate / land use / biodiversity / human impact)."""
        groups = {
            "soil": ["soc_pct", "ph", "soil_moisture_class"],
            "climate": ["rainfall_mm", "rainfall_class", "mean_temp_c", "climate"],
            "land_use": ["land_use", "crop_system", "native_habitat_pct"],
            "biodiversity": ["species_richness", "fragmentation"],
            "human_impact": ["pesticide_use", "pollution", "deforestation"],
        }
        return {g for g, fs in groups.items() if any(getattr(self, f) is not None for f in fs)}

    def merge(self, other: "SiteProfile", source: str = "user") -> "SiteProfile":
        data = self.model_dump()
        for k, v in other.model_dump().items():
            if k == "provenance" or v is None:
                continue
            data[k] = v
            data["provenance"][k] = other.provenance.get(k, source)
        return SiteProfile(**data)


class Citation(BaseModel):
    id: str
    short: str
    doi: Optional[str] = None
    verification: str
    retrieval_score: Optional[float] = None


class Recommendation(BaseModel):
    id: str
    action: str
    why: str
    reasoning_chain: list[str]
    impacted_metrics: list[str]
    addresses: list[str]
    expected_effect: str
    time_horizon: Literal["short", "medium", "long"]
    confidence: Literal["low", "medium", "high"]
    confidence_reason: str
    caveats: list[str] = []
    citations: list[Citation]
    score: float


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    site: Optional[dict[str, Any]] = None  # structured JSON input


class ChatResponse(BaseModel):
    session_id: str
    kind: Literal["clarification", "recommendations", "answer", "explanation"]
    text: str
    profile: SiteProfile
    diagnosis: list[dict] = []
    recommendations: list[Recommendation] = []
    questions: list[str] = []
    retrieval_trace: list[dict] = []
    warnings: list[str] = []
