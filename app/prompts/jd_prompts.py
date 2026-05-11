"""Prompts for JD extraction."""

JD_SYSTEM_PROMPT = (
    "你是职位描述（JD）结构化提取助手。"
    "请严格根据输入文本抽取，不要编造不存在信息。"
)

JD_VALIDITY_PROMPT = (
    "请判断输入是否是一份岗位 JD。"
    "如果是，返回 Yes；否则返回 No。"
)

JD_BASIC_INFO_PROMPT = (
    "请只提取 basicInfo 字段："
    "jobTitle, jobType, location, company, department, updateTime。"
)

JD_REQUIREMENTS_PROMPT = (
    "请只提取 requirements 字段："
    "degree, experience, techStack, mustHave, niceToHave, jobDuties。"
)
