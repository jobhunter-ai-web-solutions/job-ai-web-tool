from ml.pre_llm_filter_functions import ParsingFunctionsPreLLM
from ml.cover_letter_generator import CoverLetterGenerator
from pprint import pprint
"""
How to run this file:

    .env variables needed:
        - GEMINI_API_KEY (required)
        - GEMINI_MODEL (optional, defaults to "gemini-2.5-flash")
    
    modifiable variables:
        - job_description, company, job_title, tone (editable on the current file)
        - edit the file path under variable parser to route to a different resume pdf.
"""
# MOCK DATA (FOR TESTING PURPOSES)
job_description = """
Job Responsibilities:
Design, develop, and maintain software applications
Collaborate with cross-functional teams to define, design, and ship new features
Write clean, scalable code using programming languages such as Java, C++, or Python
Test and debug software to ensure smooth and efficient functioning
Conduct code reviews and provide feedback to other team members
Stay updated on emerging technologies and best practices in the software development industry
Essential Qualifications:
Bachelor's degree in Computer Science, Engineering, or related field
Proven work experience as a Software Engineer or Software Developer
Strong knowledge of software development methodologies and best practices
Proficiency in one or more programming languages
Excellent problem-solving skills and attention to detail
Desired Experience:
Minimum of 1 year of experience in software development
Experience with database management and cloud technologies
Experience working in an Agile development environment
Familiarity with front-end development such as React or Angular
Salary & Benefits:
Minimum Salary: $98,000 per year

Maximum Salary: $135,000 per year

Currency: USD

Benefits: Competitive salary, health insurance, 401(k) matching, paid time off
"""
company = "Wallbanger Academy"
job_title = "Software Engineer"
tone = "friendly but professional"

if __name__ == "__main__":
    parser = ParsingFunctionsPreLLM(r"ml\mock_resumes\Computer Science CAREER sample_1.pdf")
    pre_llm_full_pipeline = parser.run_full_pipeline()

    contacts = pre_llm_full_pipeline['contacts']
    sections = pre_llm_full_pipeline['sections']

    generate_it = CoverLetterGenerator()
    letter = generate_it.generate_cover_letter(
        contacts = contacts,
        sections = sections,
        job_description = job_description,
        company = company,
        job_title = job_title,
        tone = tone
    )
    print(letter)