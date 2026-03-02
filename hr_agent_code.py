#REQUIRED IMPORTS
from __future__ import annotations
import subprocess
import sys
from docx import Document
from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass,field
from datetime import datetime,date,timedelta
from typing import List,Dict
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pandas as pd
import csv
import re
import numpy as np
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

#LOADING OPENAI KEY

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=api_key)


CONFIG = {
    "llm_model": "gpt-4o-mini"   # CONFIG
}

REFERENCE_DATE = date(2023, 12, 12)  # Fixed reference date for consistent leave processing due to dates in the dataset being in the past.

# ABSTRACT INTERFACES 

class ResumeScreener(ABC):
    @abstractmethod
    def extract_skills(self, resume_text: str) -> List[str]:
        """Extract skills from resume text."""
        pass

    def rank_candidates(self, candidates: List[Candidate], jd: JobDescription) -> List[Candidate]:
        """Rank candidates against job description. Returns sorted list."""
        pass


class InterviewScheduler(ABC):
    @abstractmethod
    def schedule_interview(self, candidate: Candidate, available_slots: List[InterviewSlot]) -> Optional[InterviewSlot]:
        """Find and book an interview slot. Returns booked slot or None."""
        pass

class QuestionnaireGenerator(ABC):
    @abstractmethod
    def generate_questions(self, jd, candidate=None):
        pass

from dataclasses import dataclass
@dataclass
class InterviewSlot:
    slot_id: str
    interviewer_id: str
    start_time: datetime
    end_time: datetime
    is_available: bool = True


@dataclass
class JobDescription:
    job_id: str
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience: float


@dataclass
class Candidate:
    candidate_id: str
    name: str
    email: str
    resume_text: str
    skills: List[str]
    experience_years: float
    match_score: float = 0.0
    status: str = "applied"

@dataclass
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str         # casual, sick, earned, etc.
    start_date: date
    end_date: date
    reason: str
    status: str = "pending"  # pending, approved, rejected
    policy_violations: List[str] = field(default_factory=list)

@dataclass
class LeavePolicy:
    leave_type: str
    annual_quota: int
    max_consecutive_days: int
    min_notice_days: int
    requires_document: bool = False  # e.g., medical certificate for sick leave

POLICIES = {
    "Sick Leave": LeavePolicy(
        leave_type="Sick Leave",
        annual_quota=28,
        max_consecutive_days=7,
        min_notice_days=1,
        requires_document=True
    ),
    "Casual Leave": LeavePolicy(
        leave_type="Casual Leave",
        annual_quota=12,
        max_consecutive_days=3,
        min_notice_days=2,
        requires_document=False
    ),
    "Maternity Leave": LeavePolicy(
        leave_type="Maternity Leave",
        annual_quota=90,
        max_consecutive_days=90,
        min_notice_days=30,
        requires_document=True
    )
}

DATE_FORMAT = "%d-%m-%Y"


#PIPELINE FLOW FOR WHOLE HR AGENT (RECRUITMENT + ONBOARDING + LEAVE MANAGEMENT + ESCALATION)

PIPELINE_FLOW = [
    "applied",
    "screened",
    "interview_scheduled",
    "interviewed",
    "offer_extended",
    "offer_accepted",
    "hired"
    "onboarding_initiated"
]


# CSV LOADER FOR CANDIDATES
#extracting data from csv and creating candidate objects with resume text and skills list for each candidate
#DATASET 1 - https://www.kaggle.com/datasets/sayyedfaizan95/resume-and-job-description

def load_candidates_from_csv(csv_path: str) -> List[Candidate]:
    df = pd.read_csv(csv_path)

    candidates = []

    for index, row in df.iterrows():
        skills = []
        if pd.notna(row["Skills"]):
            skills = [s.strip() for s in row["Skills"].split(",")]

        resume_text = f"""
Education: {row['Education_Level']} in {row['Field_of_Study']}
Experience: {row['Experience_Years']} years
Skills: {row['Skills']}
Certifications: {row['Certifications']}
"""

        candidate = Candidate(
            candidate_id=f"C{index+1}",
            name=row["Name"],
            email="not_provided@email.com",
            resume_text=resume_text,
            skills=skills,
            experience_years=float(row["Experience_Years"])
        )

        candidates.append(candidate)

    return candidates


# CREATE JOB DESCRIPTION FROM CSV ROW

def create_job_description_from_row(row, index: int) -> JobDescription:

    required_skills = []
    preferred_skills = []

    if pd.notna(row["Skills"]):
        extracted_skills = [s.strip() for s in row["Skills"].split(",")]

        # Simple split: first half required, rest preferred
        split_point = len(extracted_skills) // 2
        required_skills = extracted_skills[:split_point]
        preferred_skills = extracted_skills[split_point:]

    return JobDescription(
        job_id=f"JD{index+1}",
        title=row["Target_Job_Description"],
        description=row["Target_Job_Description"],
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience=float(row["Experience_Years"])
    )

def get_current_balance(
    leave_records: List[Dict],
    employee_id: str,
    leave_type: str,
    annual_quota: int
) -> int:

    total_taken = sum(
        record["days_taken"]
        for record in leave_records
        if record["employee_id"] == employee_id
        and record["leave_type"] == leave_type
    )

    return max(0, annual_quota - total_taken)

def get_employee_existing_leaves(
    leave_records: List[Dict],
    employee_id: str
) -> List[Dict]:

    return [
        {
            "start_date": record["start_date"],
            "end_date": record["end_date"]
        }
        for record in leave_records
        if record["employee_id"] == employee_id
    ]

class PipelineManager:

    def __init__(self, flow):
        self.flow = flow

    def move_to_next_stage(self, candidate: Candidate):
        try:
            current_index = self.flow.index(candidate.status)
            if current_index < len(self.flow) - 1:
                candidate.status = self.flow[current_index + 1]
        except ValueError:
            raise ValueError(f"Invalid pipeline status: {candidate.status}")






class LLMResumeScreener(ResumeScreener):
    """Resume screening with LLM (for skills) + LOCAL embeddings."""

    THRESHOLD = 0.60 #Minium Score to be considered for screening

    def __init__(self):
        # Load embedding model ONCE
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") #Using Local LLM using sentence transformer for embedding generation


    # Step 1 : Skill Extraction (LLM-based)

    def extract_skills(self, resume_text: str) -> List[str]:

        prompt = (
            "Extract all technical and soft skills from this resume:\n"
            f"{resume_text}\n"
            "Return as JSON list."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        try:
            skills = json.loads(content)
            if isinstance(skills, list):
                return [s.strip().lower() for s in skills if isinstance(s, str)]
        except Exception:
            pass

        return []

    #Step 2 : Candidate Ranking (Skills + Experience + Semantic Similarity)
    
    def rank_candidates(self, candidates: List[Candidate], jd: JobDescription) -> List[Candidate]:

        jd_skills = set([s.lower() for s in jd.required_skills])
        jd_embedding = self._embed_text(jd.description)

        for candidate in candidates:

            # Extract skills if missing
            if not candidate.skills:
                candidate.skills = self.extract_skills(candidate.resume_text)

            candidate_skills = set([s.lower() for s in candidate.skills])

            # 1 Skill Score
            skill_score = (
                len(candidate_skills & jd_skills) / len(jd_skills)
                if jd_skills else 0.0
            )

            # 2 Experience Score
            exp_score = (
                min(candidate.experience_years / jd.min_experience, 1.0)
                if jd.min_experience > 0 else 1.0
            )

            # 3 Semantic Similarity (LOCAL)
            resume_embedding = self._embed_text(candidate.resume_text)
            semantic_score = self._cosine_similarity(resume_embedding, jd_embedding)

            # Final Weighted Score
            candidate.match_score = (
                0.4 * skill_score +
                0.2 * exp_score +
                0.4 * semantic_score
            )


        top_score = max(c.match_score for c in candidates)
        dynamic_threshold = 0.8 * top_score

        for candidate in candidates:
            if candidate.match_score >= dynamic_threshold:
                candidate.status = "screened"

        #Return sorted
        return sorted(candidates, key=lambda c: c.match_score, reverse=True)
        

    # Step 3: Embedding Utility (LOCAL MODEL)

    def _embed_text(self, text: str):

        if not text or not text.strip():
            return np.zeros(384)  # MiniLM embedding size

        return self.embedding_model.encode(text)


    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))






class BasicInterviewScheduler(InterviewScheduler):
    """Greedy scheduler — assumes interview completes immediately."""

    def schedule_interview(
        self,
        candidate: Candidate,
        available_slots: List[InterviewSlot]
    ) -> Optional[InterviewSlot]:

        # Only screened candidates can be scheduled
        if candidate.status != "screened":
            return None

        for slot in available_slots:
            if slot.is_available:
                slot.is_available = False

                # Move pipeline manually
                candidate.status = "interview_scheduled"

                # Immediately assume interview is completed
                candidate.status = "interviewed"

                return slot

        return None






class LLMQuestionnaireGenerator(QuestionnaireGenerator):

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = CONFIG["llm_model"]


    # Generate Questions
    def generate_questions(
        self,
        jd: JobDescription,
        candidate: Optional[Candidate] = None
    ) -> List[Dict]:

        candidate_context = ""

        if candidate:
            candidate_context = f"""
Candidate Information:
Name: {candidate.name}
Skills: {candidate.skills}
Experience: {candidate.experience_years} years
Resume Summary:
{candidate.resume_text[:800]}
"""

        prompt = f"""
Generate EXACTLY 10 structured interview questions in JSON format.

Role:
Title: {jd.title}
Description: {jd.description}
Required Skills: {jd.required_skills}
Preferred Skills: {jd.preferred_skills}
Minimum Experience: {jd.min_experience}

{candidate_context}

Distribution:
- 3 Technical
- 3 Behavioral (STAR format)
- 2 Situational
- 2 Role-specific

Each question must include:
"question", "type", "category", "expected_answer_points"

Return ONLY a valid JSON array.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Return only valid JSON. No explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            content = response.choices[0].message.content.strip()

            questions = self._safe_json_parse(content)

            validated = self._validate_questions(questions)

            if len(validated) == 10:
                return validated

            return self._fallback_questions(jd)

        except Exception:
            return self._fallback_questions(jd)


    #  JSON Parser

    def _safe_json_parse(self, content: str) -> List[Dict]:
        try:
            # Remove ```json blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]

            return json.loads(content)

        except Exception:
            return []

  
    # Structure Validation
  
    def _validate_questions(self, questions: List[Dict]) -> List[Dict]:

        if not isinstance(questions, list):
            return []

        required_keys = {"question", "type", "category", "expected_answer_points"}

        cleaned = []

        for q in questions:
            if (
                isinstance(q, dict) and
                required_keys.issubset(q.keys()) and
                isinstance(q["expected_answer_points"], list)
            ):
                cleaned.append(q)

        return cleaned[:10]

   
    # Guaranteed Fallback (Strict Distribution)
   
    def _fallback_questions(self, jd: JobDescription) -> List[Dict]:

        questions = []

        # 3 Technical
        for skill in jd.required_skills[:3]:
            questions.append({
                "question": f"Explain your hands-on experience using {skill}.",
                "type": "technical",
                "category": skill,
                "expected_answer_points": [
                    "Real-world usage",
                    "Challenges faced",
                    "Optimization approach"
                ]
            })

        # 3 Behavioral
        behavioral_questions = [
            "Describe a challenging project and how you handled it.",
            "Tell me about a conflict in your team and how you resolved it.",
            "Describe a failure and what you learned from it."
        ]

        for q in behavioral_questions:
            questions.append({
                "question": q,
                "type": "behavioral",
                "category": "soft_skills",
                "expected_answer_points": ["Situation", "Task", "Action", "Result"]
            })

        # 2 Situational
        situational_questions = [
            "How would you handle a production outage?",
            "What would you do if a critical deadline was at risk?"
        ]

        for q in situational_questions:
            questions.append({
                "question": q,
                "type": "situational",
                "category": "decision_making",
                "expected_answer_points": [
                    "Problem assessment",
                    "Decision process",
                    "Communication"
                ]
            })

        # 2 Role-specific
        role_specific_questions = [
            "How do you ensure scalability in system design?",
            "How do you evaluate trade-offs in architecture decisions?"
        ]

        for q in role_specific_questions:
            questions.append({
                "question": q,
                "type": "role_specific",
                "category": "architecture",
                "expected_answer_points": [
                    "Design principles",
                    "Technical trade-offs",
                    "Long-term maintainability"
                ]
            })

        return questions[:10]






class PolicyLeaveManager:

    
    # Calculate Leave Duration
    
    def calculate_days(self, start_date: date, end_date: date) -> int:

        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")

        return (end_date - start_date).days + 1

    # Process Leave Request
    def process_leave_request(
        self,
        request: LeaveRequest,
        policy: LeavePolicy,
        current_balance: int
    ) -> Dict:

        violations = []

        # Calculate Days
        try:
            days_requested = self.calculate_days(
                request.start_date,
                request.end_date
            )
        except ValueError as e:
            return {
                "status": "denied",
                "approved": False,
                "reason": str(e),
                "violations": [str(e)],
                "days_requested": 0,
                "remaining_balance": current_balance
            }

        # Policy Type Check

        if request.leave_type != policy.leave_type:
            violations.append(
                f"Policy mismatch: expected {policy.leave_type}, got {request.leave_type}"
            )

        # Balance Check

        if days_requested > current_balance:
            violations.append(
                f"Insufficient balance: requested {days_requested}, available {current_balance}"
            )


        # Max Consecutive Days

        if days_requested > policy.max_consecutive_days:
            violations.append(
                f"Exceeds max consecutive days ({policy.max_consecutive_days})"
            )

        # Notice Period Check

        today = REFERENCE_DATE
        notice_days = (request.start_date - today).days

        if notice_days < policy.min_notice_days:
            violations.append(
                f"Insufficient notice: {notice_days} days (minimum required: {policy.min_notice_days})"
            )

 
        # Documentation Check

        if policy.requires_document and (
            not request.reason or request.reason.strip() == ""
        ):
            violations.append("Required documentation not provided")


        # Final Decision

        approved = len(violations) == 0

        return {
            "status": "approved" if approved else "denied",
            "approved": approved,
            "reason": "Approved" if approved else "Denied due to policy violations",
            "violations": violations,
            "days_requested": days_requested,
            "remaining_balance": (
                current_balance - days_requested if approved else current_balance
            )
        }






class RuleBasedEscalation:
    """
    Identifies complex HR queries requiring human HR intervention.
    """

    PRIORITY_ORDER = ["low", "medium", "high"]

    ESCALATION_PATTERNS = {
        "high": [
            "harassment",
            "discrimination",
            "termination",
            "legal",
            "grievance",
            "abuse",
            "complaint",
            "lawsuit",
            "sexual harassment"
        ],
        "medium": [
            "salary revision",
            "compensation",
            "policy exception",
            "promotion dispute",
            "transfer request",
            "conflict with manager"
        ],
        "low": [
            "general complaint",
            "workplace concern",
            "feedback"
        ],
    }

    def should_escalate(self, query: str, context=None):

        query_lower = query.lower()
        detected_keywords = []
        detected_priorities = []

        for priority, keywords in self.ESCALATION_PATTERNS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                    detected_keywords.append(kw)
                    detected_priorities.append(priority)

        if detected_keywords:
            # Determine highest priority
            highest_priority = max(
                detected_priorities,
                key=lambda p: self.PRIORITY_ORDER.index(p)
            )

            return (
                True,
                f"Matched keywords: {detected_keywords}",
                highest_priority
            )

        return False, "No escalation needed", "none"







class CompensationEngine:

    ROLE_SKILL_MAP = {
        "AI/ML Engineer": ["machine learning", "ml", "deep learning", "nlp", "tensorflow", "pytorch"],
        "Data Engineer": ["spark", "hadoop", "etl", "data pipeline", "big data"],
        "Backend Engineer": ["java", "spring", "node", "api", "microservices"],
        "Frontend Engineer": ["react", "angular", "javascript", "ui", "frontend"],
        "DevOps Engineer": ["docker", "kubernetes", "aws", "ci/cd", "devops"]
    }

    BASE_SALARY = {
        "AI/ML Engineer": 1500000,
        "Data Engineer": 1200000,
        "Backend Engineer": 1100000,
        "Frontend Engineer": 1000000,
        "DevOps Engineer": 1300000
    }

    @classmethod
    def determine_role(cls, candidate: Candidate):

        skill_set = [s.lower() for s in candidate.skills]

        best_match_role = "Software Engineer"
        best_score = 0

        for role, role_skills in cls.ROLE_SKILL_MAP.items():

            score = sum(
                1 for skill in skill_set
                if any(rs in skill for rs in role_skills)
            )

            if score > best_score:
                best_score = score
                best_match_role = role

        return best_match_role

    @classmethod
    def calculate_ctc(cls, candidate: Candidate, role: str):

        base = cls.BASE_SALARY.get(role, 900000)

        # Experience multiplier
        exp_multiplier = 1 + (0.1 * candidate.experience_years)

        # Skill density bonus
        skill_bonus = min(len(candidate.skills) * 20000, 300000)

        final_salary = int(base * exp_multiplier + skill_bonus)

        return f"₹{final_salary:,.0f}"


class OfferLetterGenerator:

    STORAGE_PATH = "generated_offers"

    OFFER_TEMPLATE = """
----------------------------------------------
              OFFER LETTER
----------------------------------------------

Date: {today_date}

To,
{name}

Subject: Offer of Employment

Dear {name},

We are pleased to offer you the position of {role} at Netrik Systems.

Based on your experience and interview performance, we are confident that you will be a valuable addition to our organization.

Employment Details:
-------------------
Position        : {role}
Annual CTC      : {salary}
Joining Date    : {joining_date}
Work Location   : {location}

Terms & Conditions:
-------------------
1. This offer is subject to verification of your documents.
2. You are expected to comply with company policies.
3. Either party may terminate employment as per company guidelines.

Please confirm your acceptance by replying to this offer.

We look forward to working with you.

Sincerely,

HR Team
Netrik Systems
----------------------------------------------
"""

    @classmethod
    def generate_offer(cls, candidate: Candidate):

        os.makedirs(cls.STORAGE_PATH, exist_ok=True)

        file_path = os.path.join(
            cls.STORAGE_PATH,
            f"offer_{candidate.candidate_id}.docx"
        )

        today_date = datetime.now().strftime("%d-%m-%Y")

        document = Document()


        role = CompensationEngine.determine_role(candidate)

 
        salary = CompensationEngine.calculate_ctc(candidate, role)

        offer_text = cls.OFFER_TEMPLATE.format(
            today_date=today_date,
            name=candidate.name,
            role=role,
            salary=salary,
            joining_date="01 January 2024",
            location="Bangalore"
        )

        
        for line in offer_text.split("\n"):
            document.add_paragraph(line)

        document.save(file_path)

        return file_path






class OnboardingManager:

    STORAGE_PATH = "onboarding_records"

    def __init__(self):
        os.makedirs(self.STORAGE_PATH, exist_ok=True)
        self.records = {}

    def initiate_onboarding(self, candidate: Candidate):

        if candidate.status != "hired":
            return None  # Only hired candidates can onboard

        employee_id = f"EMP_{candidate.candidate_id}"

        onboarding_data = {
            "employee_id": employee_id,
            "name": candidate.name,
            "department": "AI Engineering",
            "reporting_manager": "Engineering Manager",
            "joining_date": "01 January 2024",
            "checklist": [
                "Document Verification",
                "Laptop Allocation",
                "Email Creation",
                "HR Policy Briefing",
                "Team Introduction"
            ],
            "status": "onboarding_initiated"
        }

        self.records[candidate.candidate_id] = onboarding_data

        # Update pipeline stage correctly
        candidate.status = "onboarding_initiated"

        return onboarding_data


class HRAgent:
    """Central HR Orchestrator"""

    def __init__(self):
        self.screener = LLMResumeScreener()
        self.scheduler = BasicInterviewScheduler()
        self.questionnaire = LLMQuestionnaireGenerator()
        self.leave_mgr = PolicyLeaveManager()
        self.escalation = RuleBasedEscalation()
        self.pipeline = {}
        self.offers = {}
        self.onboarding_mgr = OnboardingManager()


    # Recruitment Flow

    def process_recruitment(self, candidates, jd, top_n, slots):

        ranked = self.screener.rank_candidates(candidates, jd)

        for candidate in ranked:
            self.pipeline[candidate.candidate_id] = candidate

        interviews = []

        for candidate in ranked[:top_n]:
            slot = self.scheduler.schedule_interview(candidate, slots)

            if slot:
                interviews.append({
                    "candidate_id": candidate.candidate_id,
                    "name": candidate.name,
                    "status": candidate.status,
                    "slot_id": slot.slot_id
                })

                if candidate.status == "interviewed":

                    candidate.status = "offer_extended"

                    if candidate.candidate_id not in self.offers:
                        offer_path = OfferLetterGenerator.generate_offer(candidate)
                        self.offers[candidate.candidate_id] = offer_path

                    candidate.status = "offer_accepted"
                    candidate.status = "hired"
                    onboarding_info = self.onboarding_mgr.initiate_onboarding(candidate)

        return ranked, interviews


    # Questionnaire
  
    def generate_questions(self, jd, candidate=None):
        return self.questionnaire.generate_questions(jd, candidate)

 
    # Leave Management
 
    def process_leave(self, leave_request, policy, balance):
        return self.leave_mgr.process_leave_request(
            leave_request,
            policy,
            balance
        )


    # Escalation Handling (Integrated)

    def handle_query(self, query: str, context=None):

        should_esc, reason, priority = self.escalation.should_escalate(
            query,
            context or {}
        )

        if should_esc:
            return {
                "escalated": True,
                "priority": priority,
                "reason": reason,
                "query_text": query
            }

        return {
            "escalated": False,
            "priority": "none",
            "reason": "Handled automatically",
            "query_text": query
        }


    # Export Pipeline

    def export_pipeline(self):

        return {
            cid: {
                "name": c.name,
                "status": c.status,
                "score": c.match_score
            }
            for cid, c in self.pipeline.items()
        }

if __name__ == "__main__":

    import json
    import pandas as pd

    print("=" * 60)
    print("HR AGENT EVALUATION RUN")
    print("=" * 60)

    team_id = "LogicX"
    track = "TRACK_2_HR_AGENT"

    agent = HRAgent()


    # PART 1: RESUME SCREENING + SCHEDULING


    resume_csv_path = r"C:\Users\Welcome\OneDrive\Desktop\datasets\resume_dataset_1200.csv"

    df = pd.read_csv(resume_csv_path)
    candidates = load_candidates_from_csv(resume_csv_path)

    jd = create_job_description_from_row(df.iloc[0], 0)

   

    now = datetime(2023, 12, 12, 0, 0, 0)
    
    slots = [
        InterviewSlot(
            slot_id="S1",
            interviewer_id="INT1",
            start_time=now,
            end_time=now + timedelta(hours=1),
            is_available=True
        ),
        InterviewSlot(
            slot_id="S2",
            interviewer_id="INT2",
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2),
            is_available=True
        ),
    ]
    ranked_candidates, interviews = agent.process_recruitment(
        candidates=candidates,
        jd=jd,
        top_n=5,
        slots=slots
    )

    ranked_output = []
    scores_output = []

    for c in ranked_candidates:
        ranked_output.append({
            "candidate_id": c.candidate_id,
            "name": c.name,
            "score": c.match_score
        })
        scores_output.append({
            "candidate_id": c.candidate_id,
            "score": c.match_score
        })

    interviews_scheduled = interviews
    schedule_conflicts = [] 


    # PART 2: QUESTIONNAIRE

    generated_questions = agent.generate_questions(jd, ranked_candidates[0])


    # PART 3: LEAVE MANAGEMENT

    leave_file_path = r"C:\Users\Welcome\OneDrive\Desktop\datasets\employee leave tracking data.xlsx"

    #leave_df = pd.read_excel(leave_file_path, nrows=10)
    leave_df = pd.read_excel(leave_file_path)
    leave_df["Start Date"] = pd.to_datetime(leave_df["Start Date"]).dt.date
    leave_df["End Date"] = pd.to_datetime(leave_df["End Date"]).dt.date

    leave_df = leave_df.rename(columns={
        "Employee Name": "employee_id",
        "Leave Type": "leave_type",
        "Start Date": "start_date",
        "End Date": "end_date",
        "Total Leave Entitlement": "annual_quota",
        "Days Taken": "days_taken"
    })

    processed_leave_requests = []

    for index, record in enumerate(leave_df.to_dict(orient="records")):

        policy = POLICIES.get(record["leave_type"])
        if not policy:
            continue

        leave_request = LeaveRequest(
            request_id=f"REQ_{index}",
            employee_id=record["employee_id"],
            leave_type=record["leave_type"],
            start_date=record["start_date"],
            end_date=record["end_date"],
            reason="Auto Processing"
        )

        current_balance = get_current_balance(
            leave_df.to_dict(orient="records"),
            employee_id=leave_request.employee_id,
            leave_type=leave_request.leave_type,
            annual_quota=policy.annual_quota
        )

        result = agent.process_leave(
            leave_request,
            policy,
            current_balance
        )

        processed_leave_requests.append({
            "employee_id": leave_request.employee_id,
            "leave_type": leave_request.leave_type,
            "status": result["status"]
        })

    # PART 4: ESCALATION TESTING


    test_queries = [
        "I want to file a harassment complaint.",
        "Can I request salary revision?",
        "I need information about leave policy.",
        "There is discrimination happening in my team."
    ]

    escalations = []

    for i, query in enumerate(test_queries, 1):
        result = agent.handle_query(query)

        if result["escalated"]:
            escalations.append({
                "query_id": f"Q_{i}",
                "priority": result["priority"],
                "reason": result["reason"],
                "query_text": query
            })


 
    # FINAL OUTPUT FORMAT (STRICTLY AS REQUIRED)

    final_output = {
        "team_id": team_id,
        "track": track,
        "results": {
            "resume_screening": {
                "ranked_candidates": ranked_output,
                "scores": scores_output
            },
            "scheduling": {
                "interviews_scheduled": interviews_scheduled,
                "conflicts": schedule_conflicts
            },
            "questionnaire": {
                "questions": generated_questions
            },
            "pipeline": {
                "candidates": {
                    cid: c.status
                    for cid, c in agent.pipeline.items()
                }
            },
            "leave_management": {
                "processed_requests": processed_leave_requests
            },
            "escalations": escalations,
            "offers": agent.offers,
           
            
        }
    }

    with open("results.json", "w") as f:
        json.dump(final_output, f, indent=4)
    print("Results exported to results.json")

    with open("onboarding.json", "w") as f:
      json.dump(agent.onboarding_mgr.records, f, indent=4)

    print("Onboarding records exported to onboarding.json")
   
