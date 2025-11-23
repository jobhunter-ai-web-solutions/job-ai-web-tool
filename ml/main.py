# ml/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from .pre_llm_filter_functions import ParsingFunctionsPreLLM
from .cover_letter_generator import CoverLetterGenerator
import base64

app = FastAPI()

class ResumePayload(BaseModel):
    file_b64: str
    filename: str
    raw_text: str | None = None

class CoverLetterPayload(BaseModel):
    job_title: str
    company: str
    job_description: str
    contacts: dict | None = None
    sections: dict | None = None
    candidate_name: str | None = "Candidate"

@app.get("/health")
def health():
    return {"status": "ok", "service": "ml"}

@app.post("/parse-resume")
def parse_resume(payload: ResumePayload):
    content = base64.b64decode(payload.file_b64)

    parser = ParsingFunctionsPreLLM(path="<bytes>")
    if payload.raw_text:
        raw = payload.raw_text
    else:
        raw = parser.extract_text_from_pdf_bytes(content)

    clean = parser.clean_up_text(raw)
    sections = parser.define_sections(clean)
    contacts = parser.gather_contact_info_from_text(clean)

    return {
        "raw_text": raw,
        "cleaned_text": clean,
        "sections": sections,
        "contacts": contacts,
    }

@app.post("/generate-cover-letter")
def generate_cover_letter(payload: CoverLetterPayload):
    generator = CoverLetterGenerator()
    letter = generator.generate_cover_letter(
        job_title=payload.job_title,
        company=payload.company,
        job_description=payload.job_description,
        candidate_name=payload.candidate_name,
        contacts=payload.contacts,
        sections=payload.sections,
    )
    return {"cover_letter": letter}
