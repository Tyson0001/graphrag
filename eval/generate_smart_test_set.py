# eval/generate_smart_test_set.py
import pandas as pd
import json
import random

df = pd.read_csv("profiles.csv")

# Remove rows with missing names
df = df[df['name'].notna() & (df['name'].str.strip() != '')]

test_cases = []

# Sample 15 diverse profiles
for _, row in df.sample(min(15, len(df)), random_state=42).iterrows():
    name = row['name'].strip()
    core_skills = row['core_skills']
    potential_roles = row['potential_roles']
    years_exp = row['years_of_experience']
    skill_summary = row['skill_summary']
    
    # Build ground truth from actual row content
    # This ensures RAGAS can match retrieved chunks
    ground_truth = f"{name} has core skills: {core_skills}. Years of experience: {years_exp}. Potential roles: {potential_roles}."
    
    # Generate 2 questions per profile
    test_cases.append({
        "question": f"What are the core skills of {name}?",
        "ground_truth": core_skills,
        "contexts": [skill_summary],  # Use skill_summary as expected context
    })
    
    test_cases.append({
        "question": f"What roles is {name} suitable for?",
        "ground_truth": potential_roles,
        "contexts": [skill_summary],
    })

# Add generic skill-based questions that match chunk content
skill_questions = [
    {
        "question": "Who has regulatory affairs experience?",
        "ground_truth": "Candidates with Regulatory Affairs skills include those with foundational to advanced beginner knowledge in regulatory compliance, FDA regulations, and legal research.",
        "contexts": [df.sample(1).iloc[0]['skill_summary']],
    },
    {
        "question": "Find candidates with brand marketing skills",
        "ground_truth": "Candidates with Brand Marketing skills at beginner to advanced beginner levels, with experience in advisory roles.",
        "contexts": [df.sample(1).iloc[0]['skill_summary']],
    },
    {
        "question": "Who can work as a Compliance Specialist?",
        "ground_truth": "Candidates suitable for Compliance Specialist roles include those with regulatory affairs and analytical thinking skills.",
        "contexts": [df[df['potential_roles'].str.contains('Compliance', na=False)].sample(1).iloc[0]['skill_summary']],
    },
]

test_cases.extend(skill_questions)

# Ensure we have exactly 30 test cases
test_cases = test_cases[:30]

# Save
with open("eval/smart_test_set.json", "w") as f:
    json.dump(test_cases, f, indent=2)

print(f"✅ Generated {len(test_cases)} test cases")
print(f"📊 Sample questions:")
for tc in test_cases[:3]:
    print(f"  - {tc['question']}")