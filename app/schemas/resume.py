"""Resume request/response schemas (frontend-compatible)."""

from pydantic import BaseModel, Field


class BasicInfo(BaseModel):
    """Basic personal information."""

    name: str = Field(default="")
    personalEmail: str = Field(default="")
    phoneNumber: str = Field(default="")
    age: str = Field(default="")
    born: str = Field(default="")
    gender: str = Field(default="")
    desiredPosition: str = Field(default="")
    desiredLocation: list[str] = Field(default_factory=list)
    currentLocation: str = Field(default="")
    placeOfOrigin: str = Field(default="")
    rewards: list[str] = Field(default_factory=list)


class EmploymentPeriod(BaseModel):
    startDate: str = Field(default="")
    endDate: str = Field(default="")


class WorkExperienceItem(BaseModel):
    companyName: str = Field(default="")
    employmentPeriod: EmploymentPeriod = Field(default_factory=EmploymentPeriod)
    title: str = Field(default="")
    position: str = Field(default="")
    internship: int = Field(default=0)
    jobDescription: str = Field(default="")


class EducationPeriod(BaseModel):
    startDate: str = Field(default="")
    endDate: str = Field(default="")


class EducationItem(BaseModel):
    degreeLevel: str = Field(default="")
    period: EducationPeriod = Field(default_factory=EducationPeriod)
    school: str = Field(default="")
    department: str = Field(default="")
    major: str = Field(default="")
    gpa: str = Field(default="")
    ranking: str = Field(default="")
    educationDescription: str = Field(default="")


class ProjectPeriod(BaseModel):
    startDate: str = Field(default="")
    endDate: str = Field(default="")


class ProjectItem(BaseModel):
    projectName: str = Field(default="")
    projectPeriod: ProjectPeriod = Field(default_factory=ProjectPeriod)
    role: str = Field(default="")
    companyOrOrganization: str = Field(default="")
    projectDescription: str = Field(default="")


class AcademicAchievementItem(BaseModel):
    type: str = Field(default="")
    title: str = Field(default="")
    date: str = Field(default="")
    venue: str = Field(default="")
    description: str = Field(default="")
    status: str = Field(default="")


class ResumeData(BaseModel):
    """Structured resume data in frontend expected shape."""

    basicInfo: BasicInfo = Field(default_factory=BasicInfo)
    workExperience: list[WorkExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    academicAchievements: list[AcademicAchievementItem] = Field(default_factory=list)


class ResumeParseRequest(BaseModel):
    """Input payload for parse-text endpoint."""

    text: str = Field(min_length=1, description="Raw resume text")


class ResumeParseResponse(BaseModel):
    """Output payload for parse-text endpoint."""

    data: ResumeData
    meta: "ParseMeta"


class ElapsedTime(BaseModel):
    """Elapsed time for parse stages."""

    ocr_seconds: float = Field(default=0.0)
    llm_seconds: float = Field(default=0.0)


class ParseMeta(BaseModel):
    """Metadata returned to frontend."""

    filename: str = Field(default="")
    extension: str = Field(default="")
    elapsed: ElapsedTime = Field(default_factory=ElapsedTime)
    guidance: str = Field(default="")


