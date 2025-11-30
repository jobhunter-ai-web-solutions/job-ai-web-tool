from textwrap import dedent
from typing import Mapping, Any
import google.generativeai as genai
import os

DEFAULT_STYLE_GUIDE = """\
- clear, professional tone
- 1 page max
"""

def _build_prompt(
    contacts: Mapping[str, Any] | None = None,
    sections: Mapping[str, str]  | None = None,
    tone: str = "professional",
    job_title: str | None = None,
    company: str | None = None,
    extras: str | None = None,
    job_description: str | None = None,
    job_board: str | None = None,
    candidate_name: str | None = None,
) -> str:
    contacts = contacts or {}
    sections = sections or {}
    
    name = candidate_name or contacts.get("name", "Candidate")
    skills = sections.get("skills","")
    projects = sections.get("projects", "")
    experience = sections.get("experience", "")
    education = sections.get("education", "")

    return dedent(f"""
    You are drafting a cover letter.
    
    Candidate: {name}
    Target Role: {job_title or "Not Specified"}
    Company: {company or "Not Specified"}
    Tone: {tone}
    Position Found On: {job_board or "Adzuna"}

    ###  Style Guide
    {DEFAULT_STYLE_GUIDE}
    {"- " + extras if extras else ""}

    ### Job Description
    {job_description}

    ### Candidate Resume Details
    Skills:
    {skills}

    Experience:
    {experience}

    Projects:
    {projects}

    Education:
    {education}

    ### Deliverable
    Write a cover letter for the candidate that contains the following:
    - A placeholder for the cover letter writer's address, and today's date at the top.
    - A placeholder for the recruiter's name, company name, and company address below that.
    - Address the recruiter with "Dear Mr./Ms. (Insert Name)"
    - The first paragraph should indicate what position you are interested in and how you heard about it. Use the names of contact persons, if appropriate, or references to your sources of information.
    - The second paragraph should relate your experience, skills and background for the position. Highlight the specific skills and competencies that could be useful to the company.
    - The third paragraph should indicate your plans for follow-up contact and that your resume is enclosed.
    - End the letter with "Sincerely, (Insert the cover letter writer's name)"
    Output only the letter text.
    """)

class CoverLetterGenerator:
    def __init__(self, model_name: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key.strip().strip('"').strip("'"))
        model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        self.model = genai.GenerativeModel(
            model_name,
            system_instruction="You are a professional writer who crafts clear, concise cover letters."
        )

    def generate_cover_letter(
        self,
        contacts: Mapping[str, Any] | None = None,
        sections: dict | None = None,
        tone: str = "professional",
        job_title: str | None = None,
        company: str | None = None,
        job_description: str | None = None,
        job_board: str | None = None,
        candidate_name: str | None = None,
    ) -> str:
        prompt = _build_prompt(
            contacts=contacts,
            sections=sections,
            tone=tone,
            job_title=job_title,
            company=company,
            job_description=job_description,
            job_board=job_board,
            candidate_name=candidate_name,
        )
        resp = self.model.generate_content(prompt)
        return (getattr(resp, "text", None) or "").strip()