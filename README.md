# HR Agent System — README

Note: Please update the file paths in the code to match the location of the CSV files on your system before running.

## Project Title

HR Agent System — Intelligent Recruitment and HR Automation Pipeline

---

## Project Description

This project implements a modular HR Agent orchestration system that automates:

* Resume screening using LLM + embeddings
* Interview scheduling
* Interview questionnaire generation
* Leave policy validation
* HR query escalation detection
* Offer Letter Generation
* Interactive Onboarding Task Orchestration
* Structured pipeline export to JSON

The system combines AI reasoning with rule-based validation to ensure both intelligence and deterministic behavior.

## Overall Architecture

The system follows a Central Orchestrator Pattern.

Main controller:

HRAgent

Modules used:

* LLMResumeScreener
* BasicInterviewScheduler
* LLMQuestionnaireGenerator
* PolicyLeaveManager
* RuleBasedEscalation
* CompensationEngine
* OfferLetterGenerator
* OnboardingManager

Each module has a single responsibility.

---

## Recruitment Pipeline Logic

Pipeline stages:

applied
screened
interview_scheduled
interviewed
offer_extended
offer_accepted
hired

Each candidate starts with:

status = "applied"

Pipeline transitions are automatic.

---

## Resume Screening Logic

Implemented in LLMResumeScreener.

Candidate ranking uses weighted scoring.

Skill Match Score = 40%
Experience Score = 20%
Semantic Similarity = 40%

Final score:

0.4 * skill_score +
0.2 * experience_score +
0.4 * semantic_score

---

## Dynamic Shortlisting Logic (Important)

Shortlisting is based on relative performance.

After scoring all candidates:

top_score = max(candidate.match_score)

dynamic_threshold = 0.8 * top_score

Candidates are screened if:

candidate.match_score >= dynamic_threshold

This ensures only candidates close to the best performer are selected.

**Additional Rule:**  
We are considering the top **two applicants** from the selected list for further processing.

--- 

## Interview Scheduling Logic

Implemented in BasicInterviewScheduler.

Rules:

* Only screened candidates allowed
* First available slot selected
* Slot marked unavailable
* Candidate status updated

screened → interview_scheduled → interviewed

Greedy scheduling strategy used.

---

## Questionnaire Generation Logic

Implemented in LLMQuestionnaireGenerator.

Exactly 10 questions generated.

Distribution:

3 Technical
3 Behavioral (STAR)
2 Situational
2 Role-Specific

Each question contains:

question
type
category
expected_answer_points

If LLM output fails → fallback generator used.

---

## Leave Management Logic

Implemented in PolicyLeaveManager.

Checks performed:

1. Date validation
2. Leave type match
3. Balance availability
4. Max consecutive days
5. Notice period validation
6. Documentation check

If violations exist → Denied
Else → Approved

---

## Deterministic Notice Period Logic (Important)

Notice period uses a fixed reference date.

REFERENCE_DATE = 12-12-2023

Notice days calculated as:

notice_days = start_date − REFERENCE_DATE

This ensures:

* Deterministic results
* Reproducible evaluation
* Dataset-consistent validation
* No dependency on current system date

Reference date chosen to be before earliest dataset record.

---

## Escalation Detection Logic

Implemented in RuleBasedEscalation.

Keyword-based priority detection.

Priority levels:

high
medium
low

High priority keywords:

harassment
discrimination
legal
grievance
abuse

Medium:

salary revision
promotion dispute
policy exception

Low:

complaint
feedback

If keyword found → Escalated.

---

## Onboarding Records

The `onboarding.json` file stores onboarding records for candidates who were hired.  
Each record includes:

* Generated employee ID
* Employee name
* Department
* Reporting manager
* Joining date
* Onboarding checklist
* Current status (`onboarding_initiated`)

It serves as a structured handoff from recruitment to employee onboarding.

--- 

## Central Orchestrator

HRAgent integrates all modules.

Main functions:

process_recruitment()
generate_questions()
process_leave()
handle_query()
export_pipeline()

---

## Hybrid AI + Rule Based Design

AI Components:

* Skill extraction using LLM
* Embedding similarity
* Question generation

Rule Based Components:

* Leave validation
* Escalation detection
* Scheduling
* Pipeline transitions

Advantages:

* Intelligent ranking
* Deterministic validation
* Reliable output
* Modular design

---

## Output

System generates:

results.json

Contains:

* Ranked candidates
* Scores
* Interview schedule
* Questions
* Pipeline status
* Leave results
* Escalations

---

## Requirements

Python 3.10+

Install:

pip install openai python-dotenv pandas numpy openpyxl sentence-transformers

---

## Environment Variable

Create .env

OPENAI_API_KEY=your_key_here

---

## Run

From code folder:

python final_netrik_2.py

---

## Design Strengths

✔ Modular architecture
✔ Dynamic threshold shortlisting
✔ Embedding similarity scoring
✔ Deterministic leave validation
✔ Hybrid AI + rule system
✔ Structured JSON output
✔ Evaluation-ready pipeline

---

## Conclusion

This HR Agent demonstrates a production-style intelligent HR automation system combining:

* LLM reasoning
* Embedding similarity
* Rule-based compliance
* Deterministic validation
* Modular orchestration

to automate the complete HR workflow.

