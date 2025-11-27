import streamlit as st #type: ignore
import pandas as pd
from database import add, get_all, get_by_id, update, delete
from encrypted_storage import save_encrypted_db
from ai_utils import summarize_text, build_tfidf_index, semantic_search, generate_subtasks, parse_text_spacy
from main import initialize
import tests.sample_tasks as sample_tasks
from database import add

st.set_page_config(page_title="Secure AI Task Manager", layout="wide")
if "db_loaded" not in st.session_state:
    initialize()
    st.session_state.db_loaded = True


if "tfidf_index" not in st.session_state:
    st.session_state["tfidf_index"] = None

def rebuild_index():
    tasks = get_all()
    st.session_state["tfidf_index"] = build_tfidf_index(tasks)

st.markdown(
    """
    <h1 style="text-align:center; font-size: 3rem;">
         Secure AI Task Manager
    </h1>
    <p style="text-align:center; font-size:1.2rem; color:gray;">
        Privacy-first, encrypted, local task intelligence
    </p>
    <br>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["📋 Tasks", "➕ Add Task", "🔍 Semantic Search", "🤖 AI Insights"])

with tabs[0]:
    st.subheader("All Tasks")
    tasks = get_all()
    if tasks:
        df = pd.DataFrame(tasks)
        st.dataframe(df, use_container_width=True, height=500)

        ids = [t["id"] for t in tasks]
        selected_id = st.selectbox("Select a task to modify:", ids)
        if selected_id:
            task = get_by_id(selected_id)

            title = st.text_input("Title", task["title"])
            description = st.text_area("Description", task["description"] or "", height=120)
            due = st.text_input("Due Date", task["due_date"] or "")
            category = st.text_input("Category", task["category"] or "")
            priority = st.number_input("Priority", value=task["priority"])
            status = st.text_input("Status", task["status"])

            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save Changes", use_container_width=True):
                    update(
                        selected_id,
                        title=title,
                        description=description,
                        due_date=due,
                        category=category,
                        priority=priority,
                        status=status
                    )
                    rebuild_index()
                    st.success("Task updated.")

            with col2:
                if st.button("🗑 Delete Task", use_container_width=True):
                    delete(selected_id)
                    rebuild_index()
                    st.success("Task deleted.")

    else:
        st.info("No tasks available.")

with tabs[1]:
    st.subheader("Add New Task")

    title = st.text_input("Title")
    description = st.text_area("Description", height=120)
    due = st.text_input("Due Date")
    category = st.text_input("Category")
    priority = st.number_input("Priority", value=10)
    status = st.text_input("Status", value="todo", key="add_status")


    if st.button("➕ Add Task"):
        add(title, description, status=status, priority=priority, category=category, due_date=due)
        rebuild_index()
        st.success("Task added.")

with tabs[2]:
    st.subheader("Semantic Search")

    query = st.text_input("Search Query")
    if st.button("🔍 Search"):
        if st.session_state["tfidf_index"] is None:
            rebuild_index()
        results = semantic_search(query, st.session_state["tfidf_index"])
        if results:
            for tid, score in results:
                task = get_by_id(tid)
                st.markdown(f"**{task['title']}** — ({score:.4f})")
        else:
            st.info("No matching tasks found.")

with tabs[3]:
    st.subheader("AI Insights")

    tasks = get_all()
    if tasks:
        ids = [t["id"] for t in tasks]
        selected_id = st.selectbox("Choose a task", ids)
        task = get_by_id(selected_id)

        st.markdown(f"### {task['title']}")
        st.write(task["description"])

        if st.button("📝 Generate Summary"):
            st.markdown("**Summary:**")
            st.write(summarize_text(task["description"]))

        if st.button("🧩 Generate Subtasks"):
            subs = generate_subtasks(task)
            if subs is None:
                st.error("Failed to generate subtasks.")
            elif not subs:
                st.info("No subtasks generated.")
            else:
                st.markdown(f"{subs}")
            try:
                for s in subs:
                    st.markdown(f"**{s['title']}** — {s['description']}")
            except:
                st.error("Error displaying subtasks.")

        if st.button("🔬 Show Parsed Features"):
            parsed = parse_text_spacy(task["description"] or "")
            st.json(parsed)
    else:
        st.info("No tasks found.")

if st.button("🔐 Save Encrypted Database Now"):
    save_encrypted_db()
    st.success("Encrypted DB saved.")
