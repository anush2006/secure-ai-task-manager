from encrypted_storage import load_or_init_database, save_encrypted_db
from database import add, get_all

load_or_init_database()

sample_tasks = [
    ("Fix PDF export crash", "App crashes when exporting large PDF files. Requires stack trace analysis and memory profiling.", "2025-03-01", "bug", 8, "todo"),
    ("Implement JWT authentication", "Add secure login system with JWT tokens for mobile and web clients.", "2025-02-10", "feature", 6, "todo"),
    ("Improve image compression", "Optimize PNG and JPG compression pipeline. Reduce output size by at least 30%.", "2025-04-05", "maintenance", 5, "todo"),
    ("Add dark mode", "Frontend UI needs a toggle-based dark mode. Update CSS variables and component themes.", "2025-03-20", "feature", 7, "todo"),
    ("Research WebAssembly integration", "Investigate feasibility of moving critical CPU-heavy tasks to WebAssembly.", "2025-02-18", "research", 5, "todo"),
    ("Fix login timeout issue", "Users are logged out too quickly due to token refresh failure.", "2025-02-15", "bug", 9, "in-progress"),
    ("Write API documentation", "Document all endpoints, schemas, rate limits, and authentication flow.", "2025-02-25", "documentation", 4, "todo"),
    ("Set up CI/CD pipeline", "Configure GitHub Actions for testing, linting, and deployment to staging.", "2025-02-28", "devops", 8, "todo"),
    ("Database migration cleanup", "Remove deprecated tables, update schemas, and ensure backward compatibility.", "2025-03-09", "maintenance", 6, "todo"),
    ("UI/UX redesign for dashboard", "Revamp layout, spacing, and typography for better readability.", "2025-03-12", "feature", 5, "todo"),
    ("Implement rate limiting", "Protect API endpoints by adding per-IP rate limiting and abuse detection system.", "2025-04-01", "feature", 7, "todo"),
    ("Optimize SQL queries", "Several slow queries detected during monitoring. Apply indexing and query refactor.", "2025-03-22", "maintenance", 8, "todo"),
    ("Add multi-language support", "Support English, German, Spanish through i18n framework.", "2025-05-01", "feature", 4, "todo"),
    ("Fix Stripe payment bug", "Webhook handler fails for recurring payments. Investigate signature mismatch.", "2025-02-17", "bug", 9, "todo"),
    ("Improve test coverage", "Increase backend unit test coverage from 62% to at least 85%.", "2025-03-11", "maintenance", 4, "todo"),
    ("Integrate Redis caching", "Cache expensive queries and session data in Redis Cluster.", "2025-03-10", "feature", 7, "todo"),
    ("Add export to Excel", "Users need ability to export table data to Excel with custom formatting.", "2025-02-21", "feature", 6, "todo"),
    ("Fix logout redirect loop", "Logout sometimes loops back to login unintentionally.", "2025-02-14", "bug", 8, "todo"),
    ("Research AI-assisted code search", "Investigate techniques using static analysis and classical IR methods.", "2025-03-18", "research", 5, "todo"),
    ("Improve deployment logs", "Enhance logging for deployments to track container rollouts and health checks.", "2025-03-07", "devops", 3, "todo")
]

for title, desc, due, cat, pri, status in sample_tasks:
    add(title, desc, status=status, priority=pri, category=cat, due_date=due)

save_encrypted_db()

all_tasks = get_all()
print(f"Persisted {len(all_tasks)} tasks to encrypted storage (data/tasks.enc).")
print("Now start the UI with: streamlit run app.py")
