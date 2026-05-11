"""JD extraction schemas for learning backend."""

from pydantic import BaseModel, Field

class JDBasicInfo(BaseModel):
    """Basic job posting fields."""

    jobTitle: str = Field(default="")
    jobType: str = Field(default="")
    location: str = Field(default="")
    company: str = Field(default="")
    department: str = Field(default="")
    updateTime: str = Field(default="")


class JDRequirements(BaseModel):
    """Job requirements fields."""

    degree: str = Field(default="")
    experience: str = Field(default="")
    techStack: list[str] = Field(default_factory=list)
    mustHave: list[str] = Field(default_factory=list)
    niceToHave: list[str] = Field(default_factory=list)
    jobDuties: list[str] = Field(default_factory=list)


class JDData(BaseModel):
    """Full JD structured payload."""

    basicInfo: JDBasicInfo = Field(default_factory=JDBasicInfo)
    requirements: JDRequirements = Field(default_factory=JDRequirements)


class JDExtractRequest(BaseModel):
    """JD extraction request body."""

    text: str = Field(description="Raw JD text")


class JDExtractResponse(BaseModel):
    """JD extraction response body."""

    data: JDData
    elapsed_seconds: float
