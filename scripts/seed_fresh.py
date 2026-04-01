"""
Seed Script - Creates test users for demo/development.

Run from backend directory:
    python -m scripts.seed_fresh

PRESERVES: plan_definitions, point_costs, feature_limits, badges, titles
           (all system tables are untouched)
CLEARS:    profiles, organizations (cascades to all user data)

Creates:
  Organization: GenVerse Academy (Academic Year: 2025-2026, org_pro plan)

  All passwords: Test@123

  Admins (2):
    admin@genverse.dev   (Sakthi Kumar)
    admin2@genverse.dev  (Kavitha Rajan)

  Teachers (3):
    teacher1@genverse.dev  (Priya Sharma - Mathematics)
    teacher2@genverse.dev  (Rajesh Kumar - Science)
    teacher3@genverse.dev  (Meena Iyer - English)

  Students (10) - Grade 10, Section A, CBSE:
    student1@genverse.dev .. student10@genverse.dev

  Classes (5): Mathematics, Science, English, Social Studies, Hindi
  All students enrolled in all classes.
  teacher1 assigned as Class Teacher for Grade 10A.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import uuid
from datetime import datetime, timezone, timedelta, date
from sqlalchemy import text
from app.database import engine
from app.core.security import hash_password

PW_ALL = hash_password("Test@123")

ORG_ID = str(uuid.uuid4())
AY = "2025-2026"

ADMIN_IDS = [str(uuid.uuid4()), str(uuid.uuid4())]
TEACHER_IDS = [str(uuid.uuid4()) for _ in range(3)]
STUDENT_IDS = [str(uuid.uuid4()) for _ in range(10)]
CLASS_IDS = [str(uuid.uuid4()) for _ in range(5)]

ADMINS = [
    {"id": ADMIN_IDS[0], "name": "Sakthi Kumar", "email": "admin@genverse.dev"},
    {"id": ADMIN_IDS[1], "name": "Kavitha Rajan", "email": "admin2@genverse.dev"},
]

TEACHERS = [
    {"id": TEACHER_IDS[0], "name": "Priya Sharma",  "email": "teacher1@genverse.dev", "dept": "Mathematics", "emp": "EMP001", "qual": "M.Sc, B.Ed", "gender": "Female"},
    {"id": TEACHER_IDS[1], "name": "Rajesh Kumar",  "email": "teacher2@genverse.dev", "dept": "Science",     "emp": "EMP002", "qual": "M.Sc, B.Ed", "gender": "Male"},
    {"id": TEACHER_IDS[2], "name": "Meena Iyer",    "email": "teacher3@genverse.dev", "dept": "English",     "emp": "EMP003", "qual": "M.A, B.Ed",  "gender": "Female"},
]

STUDENTS = [
    {"id": STUDENT_IDS[0], "name": "Aarav Patel",    "email": "student1@genverse.dev",  "roll": "10A001", "gender": "Male",   "dob": "2011-03-15", "blood": "B+",  "parent": "Mahesh Patel",   "pph": "9876500001", "ec": "Mahesh Patel",   "ecph": "9876500001", "ecr": "Father"},
    {"id": STUDENT_IDS[1], "name": "Ananya Singh",   "email": "student2@genverse.dev",  "roll": "10A002", "gender": "Female", "dob": "2011-07-22", "blood": "O+",  "parent": "Ramesh Singh",   "pph": "9876500002", "ec": "Sunita Singh",   "ecph": "9876500012", "ecr": "Mother"},
    {"id": STUDENT_IDS[2], "name": "Arjun Nair",     "email": "student3@genverse.dev",  "roll": "10A003", "gender": "Male",   "dob": "2011-01-10", "blood": "A+",  "parent": "Vijay Nair",     "pph": "9876500003", "ec": "Vijay Nair",     "ecph": "9876500003", "ecr": "Father"},
    {"id": STUDENT_IDS[3], "name": "Diya Reddy",     "email": "student4@genverse.dev",  "roll": "10A004", "gender": "Female", "dob": "2011-11-05", "blood": "AB+", "parent": "Suresh Reddy",   "pph": "9876500004", "ec": "Lakshmi Reddy",  "ecph": "9876500014", "ecr": "Mother"},
    {"id": STUDENT_IDS[4], "name": "Karthik Raj",    "email": "student5@genverse.dev",  "roll": "10A005", "gender": "Male",   "dob": "2011-06-18", "blood": "O-",  "parent": "Suresh Raj",     "pph": "9876500005", "ec": "Suresh Raj",     "ecph": "9876500005", "ecr": "Father"},
    {"id": STUDENT_IDS[5], "name": "Meera Das",      "email": "student6@genverse.dev",  "roll": "10A006", "gender": "Female", "dob": "2011-09-25", "blood": "B-",  "parent": "Anil Das",       "pph": "9876500006", "ec": "Priya Das",      "ecph": "9876500016", "ecr": "Mother"},
    {"id": STUDENT_IDS[6], "name": "Rohan Gupta",    "email": "student7@genverse.dev",  "roll": "10A007", "gender": "Male",   "dob": "2011-02-14", "blood": "A-",  "parent": "Amit Gupta",     "pph": "9876500007", "ec": "Amit Gupta",     "ecph": "9876500007", "ecr": "Father"},
    {"id": STUDENT_IDS[7], "name": "Sanya Mehta",    "email": "student8@genverse.dev",  "roll": "10A008", "gender": "Female", "dob": "2011-04-30", "blood": "O+",  "parent": "Vikram Mehta",   "pph": "9876500008", "ec": "Rekha Mehta",    "ecph": "9876500018", "ecr": "Mother"},
    {"id": STUDENT_IDS[8], "name": "Vivaan Sharma",  "email": "student9@genverse.dev",  "roll": "10A009", "gender": "Male",   "dob": "2011-08-12", "blood": "AB-", "parent": "Deepak Sharma",  "pph": "9876500009", "ec": "Deepak Sharma",  "ecph": "9876500009", "ecr": "Father"},
    {"id": STUDENT_IDS[9], "name": "Ishita Joshi",   "email": "student10@genverse.dev", "roll": "10A010", "gender": "Female", "dob": "2011-12-03", "blood": "B+",  "parent": "Rakesh Joshi",   "pph": "9876500010", "ec": "Kavita Joshi",   "ecph": "9876500020", "ecr": "Mother"},
]

CLASSES = [
    {"id": CLASS_IDS[0], "name": "Mathematics - 10 - A",    "subject": "Mathematics",    "tid": TEACHER_IDS[0], "code": "MAT10A01", "color": "#6366f1"},
    {"id": CLASS_IDS[1], "name": "Science - 10 - A",        "subject": "Science",        "tid": TEACHER_IDS[1], "code": "SCI10A02", "color": "#14b8a6"},
    {"id": CLASS_IDS[2], "name": "English - 10 - A",        "subject": "English",        "tid": TEACHER_IDS[2], "code": "ENG10A03", "color": "#f59e0b"},
    {"id": CLASS_IDS[3], "name": "Social Studies - 10 - A", "subject": "Social Studies", "tid": TEACHER_IDS[0], "code": "SOC10A04", "color": "#ec4899"},
    {"id": CLASS_IDS[4], "name": "Hindi - 10 - A",          "subject": "Hindi",          "tid": TEACHER_IDS[2], "code": "HIN10A05", "color": "#8b5cf6"},
]


async def _insert_user(conn, uid, email, pw, name, role, now, pe, **extra):
    """Insert profile + user_role + org_member + personal subscription."""
    fields = "id, email, hashed_password, name, language, auth_provider, onboarding_completed, xp, streak, is_active, created_at, updated_at"
    vals = ":id, :email, :pw, :name, 'en', 'email', true, 0, 0, true, :now, :now"
    params = {"id": uid, "email": email, "pw": pw, "name": name, "now": now}
    for k, v in extra.items():
        if v is not None:
            fields += f", {k}"
            vals += f", :{k}"
            params[k] = v
    await conn.execute(text(f"INSERT INTO profiles ({fields}) VALUES ({vals})"), params)
    await conn.execute(text(
        "INSERT INTO user_roles (id, user_id, role, created_at) VALUES (:id, :uid, :role, :now)"
    ), {"id": str(uuid.uuid4()), "uid": uid, "role": role, "now": now})
    await conn.execute(text(
        "INSERT INTO org_members (id, org_id, user_id, role, status, joined_at) VALUES (:id, :oid, :uid, :role, 'active', :now)"
    ), {"id": str(uuid.uuid4()), "oid": ORG_ID, "uid": uid, "role": role, "now": now})
    await conn.execute(text(
        "INSERT INTO subscriptions (id, user_id, plan, status, workspace_type, points_balance, points_monthly_quota, storage_limit_mb, current_period_start, current_period_end, auto_renew) "
        "VALUES (:id, :uid, 'free', 'active', 'individual', 100, 100, 100, :s, :e, false)"
    ), {"id": str(uuid.uuid4()), "uid": uid, "s": now, "e": pe})


async def run():
    async with engine.begin() as conn:
        print("=== Clearing user data (system tables preserved) ===")
        await conn.execute(text("TRUNCATE TABLE profiles CASCADE"))
        await conn.execute(text("TRUNCATE TABLE organizations CASCADE"))

        now = datetime.now(timezone.utc)
        pe = now + timedelta(days=365)

        # Organization
        print("\n--- Organization ---")
        await conn.execute(text(
            "INSERT INTO organizations (id, name, product_type, has_genverse, has_evaluation, enforce_academic_context, default_theme, current_academic_year, allowed_boards, created_at, updated_at) "
            "VALUES (:id, 'GenVerse Academy', 'genverse_evaluation', true, true, false, 'classic', :ay, '[\"CBSE\",\"ICSE\",\"IGCSE\",\"IB\",\"Cambridge\",\"State Board\"]', :now, :now)"
        ), {"id": ORG_ID, "ay": AY, "now": now})
        await conn.execute(text(
            "INSERT INTO subscriptions (id, org_id, plan, status, workspace_type, points_balance, points_monthly_quota, storage_limit_mb, max_seats, current_period_start, current_period_end, auto_renew) "
            "VALUES (:id, :oid, 'org_pro', 'active', 'organization', 50000, 50000, 51200, 1000, :s, :e, false)"
        ), {"id": str(uuid.uuid4()), "oid": ORG_ID, "s": now, "e": pe})
        print("  GenVerse Academy (org_pro, 50k pts)")

        # Admins
        print("\n--- Admins ---")
        for a in ADMINS:
            await _insert_user(conn, a["id"], a["email"], PW_ALL, a["name"], "org_admin", now, pe)
            print(f"  {a['email']} / Test@123")

        # Teachers
        print("\n--- Teachers ---")
        for t in TEACHERS:
            await _insert_user(conn, t["id"], t["email"], PW_ALL, t["name"], "teacher", now, pe,
                               employee_id=t["emp"], department=t["dept"], qualification=t["qual"], gender=t["gender"])
            print(f"  {t['email']} / Test@123  ({t['name']})")

        # Students
        print("\n--- Students (Grade 10, Section A, CBSE) ---")
        for s in STUDENTS:
            await _insert_user(conn, s["id"], s["email"], PW_ALL, s["name"], "student", now, pe,
                               grade=10, section="A", board_preference="CBSE", roll_number=s["roll"],
                               gender=s["gender"], date_of_birth=date.fromisoformat(s["dob"]),
                               blood_group=s["blood"], parent_name=s["parent"], parent_phone=s["pph"],
                               emergency_contact_name=s["ec"], emergency_contact_phone=s["ecph"],
                               emergency_contact_relation=s["ecr"], city="Chennai", state="Tamil Nadu", pincode="600001")
            print(f"  {s['email']} / Test@123  ({s['name']} - {s['roll']})")

        # Classes
        print("\n--- Classes ---")
        for c in CLASSES:
            await conn.execute(text(
                "INSERT INTO classes (id, org_id, name, board, grade, subject, section, join_code, teacher_id, color, academic_year, is_active, created_at, updated_at) "
                "VALUES (:id, :oid, :name, 'CBSE', 10, :subject, 'A', :code, :tid, :color, :ay, true, :now, :now)"
            ), {**c, "oid": ORG_ID, "ay": AY, "now": now})
            tname = next(t["name"] for t in TEACHERS if t["id"] == c["tid"])
            print(f"  {c['subject']}  ({tname})")

        # Enrollments
        print("\n--- Enrollments ---")
        for c in CLASSES:
            for s in STUDENTS:
                await conn.execute(text(
                    "INSERT INTO class_students (id, class_id, student_id, roll_no, joined_at) VALUES (:id, :cid, :sid, :roll, :now)"
                ), {"id": str(uuid.uuid4()), "cid": c["id"], "sid": s["id"], "roll": s["roll"], "now": now})
        print(f"  {len(CLASSES) * len(STUDENTS)} enrollments")

        # Class Teacher
        print("\n--- Class Teacher ---")
        await conn.execute(text(
            "INSERT INTO grade_section_teachers (id, org_id, teacher_id, grade, section, board, academic_year, created_at) "
            "VALUES (:id, :oid, :tid, 10, 'A', 'CBSE', :ay, :now)"
        ), {"id": str(uuid.uuid4()), "oid": ORG_ID, "tid": TEACHER_IDS[0], "ay": AY, "now": now})
        print("  Priya Sharma -> Grade 10 Section A")

    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"\nOrg: GenVerse Academy | Year: {AY}")
    print("\nAll passwords: Test@123")
    print("\nAdmin 1:  admin@genverse.dev")
    print("Admin 2:  admin2@genverse.dev")
    print("Teacher1: teacher1@genverse.dev  (Class Teacher 10A)")
    print("Teacher2: teacher2@genverse.dev")
    print("Teacher3: teacher3@genverse.dev")
    print("Students: student1-10@genverse.dev")
    print(f"\nClasses: 5 | Enrollments: {len(CLASSES) * len(STUDENTS)}")


if __name__ == "__main__":
    asyncio.run(run())
