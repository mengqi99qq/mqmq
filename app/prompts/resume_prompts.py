"""Prompts for LLM-based resume extraction."""

SYSTEM_PROMPT = (
    "你是简历解析助手。请严格根据用户提供的简历文本抽取信息，"
    "不要臆造不存在的内容；无法确定时返回空字符串或空数组。"
)

RESUME_EXTRACT_PROMPT = (
    "请抽取以下字段：\n"
    "1) basicInfo: name, personalEmail, phoneNumber, age, born, gender, desiredPosition, desiredLocation, currentLocation, placeOfOrigin, rewards\n"
    "2) workExperience: companyName, employmentPeriod(startDate,endDate), title, position, internship, jobDescription\n"
    "3) education: degreeLevel, period(startDate,endDate), school, department, major, gpa, ranking, educationDescription\n"
    "4) projects: projectName, projectPeriod(startDate,endDate), role, companyOrOrganization, projectDescription\n"
    "5) academicAchievements: type, title, date, venue, description, status\n"
    "要求：\n"
    "- 所有字段都必须返回。\n"
    "- 不能从文本外补充信息。\n"
    "- 不确定时返回空字符串或空数组，不要编造。\n"
    "- internship 仅输出 0 或 1。"
)

RESUME_FIELD_COMMON_RULES = (
    "要求：\n"
    "- 仅基于给定简历文本提取，不要编造。\n"
    "- 缺失字段返回空字符串或空数组。\n"
    "- 保持字段名与结构严格一致。"
)

BASIC_INFO_PROMPT = (
    "请只提取 basicInfo 字段："
    "name, personalEmail, phoneNumber, age, born, gender, desiredPosition, "
    "desiredLocation, currentLocation, placeOfOrigin, rewards。\n"
    f"{RESUME_FIELD_COMMON_RULES}"
)

WORK_EXPERIENCE_PROMPT = (
    "请只提取 workExperience 字段数组，"
    "每项包含 companyName, employmentPeriod(startDate,endDate), title, position, internship, jobDescription。\n"
    "- internship 仅输出 0 或 1。\n"
    f"{RESUME_FIELD_COMMON_RULES}"
)

EDUCATION_PROMPT = (
    "请只提取 education 字段数组，"
    "每项包含 degreeLevel, period(startDate,endDate), school, department, major, gpa, ranking, educationDescription。\n"
    f"{RESUME_FIELD_COMMON_RULES}"
)

PROJECT_PROMPT = (
    "请只提取 projects 字段数组，"
    "每项包含 projectName, projectPeriod(startDate,endDate), role, companyOrOrganization, projectDescription。\n"
    f"{RESUME_FIELD_COMMON_RULES}"
)

ACADEMIC_ACHIEVEMENTS_PROMPT = (
    "请只提取 academicAchievements 字段数组，"
    "每项包含 type, title, date, venue, description, status。\n"
    f"{RESUME_FIELD_COMMON_RULES}"
)
