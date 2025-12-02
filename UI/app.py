# app.py
import os
import re
import ast
import random

import streamlit as st
import requests
import pandas as pd

# ---------------------------------------------------------
# Make OPENAI_API_KEY available to ai_helpers via env var
# (ai_helpers.py will read from os.getenv or st.secrets)
# ---------------------------------------------------------
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

# 👇 import challenge stuff from challenges.py
from challenges import (
    init_tutor_state,
    tutor_sidebar,
    flashcards_ui,
    mcq_ui,
    fill_in_blank_ui,
    best_qa_match,
    match_answers_ui,
    timed_question_ui,
    scenario_ui,
    progress_dashboard_ui,   # ✅ dashboard
)

# --------------- GLOBAL CONFIG -----------------
st.set_page_config(page_title="Textbook Tutor", layout="wide")


# --------------- HELPERS FOR MULTI-BOOK RAG (API view) -----------------
def multi_book_rag_ui():
    st.title("Textbook Tutor — Multi-Book RAG (New Project)")

    api = st.secrets.get("API_URL", "http://localhost:8000")

    # ---- 1) Upload / ingest first ----
    st.subheader("1) Upload a textbook PDF")
    up = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=False, key="multi_pdf")
    title = st.text_input("Book title (for display)", value="Untitled Book", key="multi_title")
    if st.button("Ingest PDF", disabled=up is None, key="multi_ingest_btn"):
        if up:
            resp = requests.post(
                f"{api}/ingest",
                files={"file": (up.name, up, "application/pdf")},
                data={"book_title": title},
            )
            if resp.status_code == 200:
                st.success(resp.json())
            else:
                st.error(resp.text)

    # ---- Sidebar library (runs after ingest, so /books sees new data) ----
    with st.sidebar:
        st.header("Library")
        try:
            books = requests.get(f"{api}/books").json()
        except Exception as e:
            st.error(f"API not reachable at {api}. Start FastAPI first. Error: {e}")
            books = {}

        book_options = {f"{v['book_title']} ({b})": b for b, v in books.items()}
        book_choice = st.multiselect("Select book(s)", list(book_options.keys()))
        chosen_book_ids = [book_options[k] for k in book_choice]

        chapters = []
        for b in chosen_book_ids:
            chapters.extend(books[b]["chapters"])  # chapter IDs or names
        chapter_choice = st.multiselect("Filter by chapter IDs", chapters)

    # ---- 2) Ask a question ----
    st.subheader("2) Ask a question")
    q = st.text_input(
        "Your question",
        value="Explain pivot tables with a simple example.",
        key="multi_q",
    )
    k = st.slider("Top-k", 3, 15, 6, key="multi_k")
    if st.button("Ask", key="multi_ask_btn"):
        body = {
            "query": q,
            "k": k,
            "scope": {"book_ids": chosen_book_ids, "chapter_ids": chapter_choice},
        }
        r = requests.post(f"{api}/ask", json=body)
        if r.ok:
            data = r.json()
            st.write("**Answer**:", data.get("answer"))
            st.write("**Citations**:")
            for c in data.get("citations", []):
                st.caption(
                    f"book={c['book_id']} chapter={c['chapter_id']} page={c['page']} score={c['score']:.3f}"
                )
        else:
            st.error(r.text)

    # ---- 3) Story / Case / Quiz ----
    st.subheader("3) Story / Case / Quiz")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate Story", key="multi_story_btn"):
            body = {
                "scope": {"book_ids": chosen_book_ids, "chapter_ids": chapter_choice},
                "max_words": 160,
            }
            r = requests.post(f"{api}/story", json=body)
            if r.ok:
                st.write(r.json())
            else:
                st.error(r.text)

    with col2:
        if st.button("Generate Case", key="multi_case_btn"):
            body = {"scope": {"book_ids": chosen_book_ids, "chapter_ids": chapter_choice}}
            r = requests.post(f"{api}/case", json=body)
            if r.ok:
                st.write(r.json())
            else:
                st.error(r.text)

    with col3:
        n = st.number_input("# MCQs", min_value=3, max_value=15, value=5, step=1, key="multi_n_mcq")
        if st.button("Build Quiz", key="multi_quiz_btn"):
            body = {
                "scope": {"book_ids": chosen_book_ids, "chapter_ids": chapter_choice},
                "n_mcq": int(n),
            }
            r = requests.post(f"{api}/quiz", json=body)
            if r.ok:
                items = r.json().get("items", [])
                for i, it in enumerate(items, 1):
                    st.markdown(f"**Q{i}. {it.get('question','')}**")
                    opts = it.get("options", {})
                    for key, val in opts.items():
                        st.write(f"- {key}. {val}")
                    st.caption(
                        f"Answer: {it.get('correct','?')} | Evidence: {it.get('evidence','')}"
                    )
            else:
                st.error(r.text)


# --------------- SINGLE-TEXTBOOK INTERACTIVE TUTOR -----------------
DATA_URL = "https://raw.githubusercontent.com/sravani919/AI_Tutor_Interactive_learning/main/Merged_Chapter_Dataset.csv"

@st.cache_data(show_spinner="Loading chapter dataset...")
def load_chapter_data():
    df = pd.read_csv(DATA_URL)
    chapter_summaries = {}
    chapter_questions = {}
    chapter_answers = {}

    def clean_answer_from_question(question, answer):
        q_words = question.lower().split()
        a_words = answer.strip().split()
        q_set = set(w.strip(".,?") for w in q_words)
        start_index = 0
        for w in a_words:
            cw = w.lower().strip(".,?")
            if cw not in q_set:
                break
            start_index += 1
        trimmed = a_words[start_index:]
        cleaned = " ".join(trimmed).strip()
        if not cleaned or len(cleaned.split()) <= 3:
            cleaned = "It refers to " + " ".join(a_words)
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned.rstrip(". ")

    for _, row in df.iterrows():
        chapter = str(row["chapter"])
        chapter_content = str(row.get("Chapter Content", "")) or "No summary available."

        try:
            questions = ast.literal_eval(row["Questions"]) if isinstance(row["Questions"], str) else row["Questions"]
        except Exception:
            questions = []
        try:
            answers = ast.literal_eval(row["Answers"]) if isinstance(row["Answers"], str) else row["Answers"]
        except Exception:
            answers = []

        chapter_summaries[chapter] = chapter_content
        q5 = questions[:5] if questions else []
        chapter_questions[chapter] = q5
        cleaned_answers = [clean_answer_from_question(q, a) for q, a in zip(q5, answers[:5])] if questions and answers else []
        chapter_answers[chapter] = cleaned_answers

    return chapter_summaries, chapter_questions, chapter_answers


def clean_chapter_name(chapter_name: str) -> str:
    return re.sub(r"^\d+(\.\d+)?\s*", "", str(chapter_name)).strip()


def generate_business_case(chapter: str, chapter_summaries):
    cleaned = clean_chapter_name(chapter)
    summary = chapter_summaries.get(chapter, "")
    company = random.choice(
        ["AlphaCorp", "Beta Enterprises", "Gamma Solutions", "Delta Analytics", "Nova Systems"]
    )
    exec_summary = (
        f"{company} is struggling to properly apply {cleaned} in their day-to-day work. "
        f"Teams rely on manual processes and outdated tools, which slows decisions and increases errors. "
        f"This case explores how adopting the ideas from “{cleaned}” can improve accuracy, collaboration, "
        "and data-driven decision making."
    )
    problem = (
        f"Managers at {company} receive reports that are inconsistent, hard to interpret, "
        "and often delivered too late. There is no standard way to store, retrieve, or analyze information."
    )
    importance = (
        "If this continues, the company risks lost revenue, poor customer experience, "
        "and low trust in internal data. Modern, structured approaches are needed."
    )
    solution = (
        f"The operations team proposes a pilot project where a small group adopts the concepts from “{cleaned}”. "
        "They define clear data structures, automate repetitive steps, and train staff on best practices."
    )
    objectives = [
        f"Reduce manual work related to {cleaned} by at least 40%.",
        "Increase accuracy of internal reports and dashboards.",
        "Make it easier for non-technical staff to understand key results.",
        "Create a repeatable workflow that can be scaled to other teams.",
    ]
    financials = (
        "The pilot requires an up-front investment of $50,000 in training and tooling, "
        "but is expected to save around $80,000 per year by reducing rework and delays."
    )
    conclusion = (
        f"If the pilot succeeds, {company} can roll out the new approach across departments, "
        "building a culture of data-driven decision making."
    )

    return {
        "chapter": cleaned,
        "company": company,
        "summary": summary,
        "executive_summary": exec_summary,
        "problem": problem,
        "importance": importance,
        "solution": solution,
        "objectives": objectives,
        "financials": financials,
        "conclusion": conclusion,
    }


def generate_story(chapter: str, chapter_summaries):
    cleaned = clean_chapter_name(chapter)
    summary = chapter_summaries.get(chapter, "This chapter introduces key ideas you'll apply in practice.")
    protagonist = random.choice([
        ("Mia", "data analyst"),
        ("Alex", "junior accountant"),
        ("Jordan", "IT coordinator"),
        ("Sophia", "project manager"),
        ("Leo", "business intelligence intern"),
    ])
    name, role = protagonist

    story = f"""
{name} works as {role} at a mid-sized company. Recently, their manager asked them to improve how the team works with <b>{cleaned}</b>.
At first, {name} felt overwhelmed—there were long documents, spreadsheets everywhere, and no clear structure.

After reading the key ideas from this chapter, {name} starts with small steps:
• They identify where information lives today.<br>
• They apply one concept from the chapter to clean things up.<br>
• They build a simple example the whole team can understand.<br><br>

Soon, meetings become shorter and decisions become clearer. Instead of arguing about “who has the right file”, the team opens a shared view and focuses on the actual problem.

By the end of the week, {name} realizes that <b>{cleaned}</b> isn’t just theory—it’s a toolkit for making everyday work less chaotic.

<i>Reflection:</i> Based on this story and the chapter summary below, how could you apply one idea from this chapter in your own context?
"""
    return story, summary


# ---- Interactive Tutor main UI ----
def interactive_tutor_ui():
    init_tutor_state()
    state = st.session_state.tutor
    tutor_sidebar()

    st.title("📚 Interactive AI Tutor – Single Textbook Demo")

    # Learner name
    name = st.text_input("First things first — what should I call you?", value=state["name"])
    if name and name != state["name"]:
        state["name"] = name

    if not state["name"]:
        st.info("Enter your name to unlock the tutor experience ✨.")
        return

    chapter_summaries, chapter_questions, chapter_answers = load_chapter_data()
    chapters = sorted(chapter_summaries.keys())

    st.markdown(f"Hi **{state['name']}** 👋 — pick a chapter and how you’d like to learn it.")

    col_ch, col_mode = st.columns([2, 1])

    with col_ch:
        chapter = st.selectbox(
            "Choose a chapter",
            ["Select a chapter"] + chapters,
            index=chapters.index(state["chapter"]) + 1 if state["chapter"] in chapters else 0,
        )
        if chapter != "Select a chapter":
            state["chapter"] = chapter
    with col_mode:
        mode = st.radio(
            "Learning mode",
            [
                "Summary only",
                "📊 Business Case",
                "📖 Storytelling",
                "🎮 Challenges",
                "💬 Ask a Question",
                "📈 My Progress",   # ✅ new dashboard mode
            ],
            index=0,
        )
        state["mode"] = mode

    if not state["chapter"]:
        st.warning("Select a chapter to continue.")
        return

    cleaned = clean_chapter_name(state["chapter"])
    st.markdown(f"### 📘 {cleaned}")
    st.write(chapter_summaries.get(state["chapter"], "No summary available yet."))

    st.divider()

    # MODE RENDERING
    if mode == "Summary only":
        st.info("Use the controls above to explore business cases, storytelling, challenges, or Q&A for this chapter.")

    elif mode == "📊 Business Case":
        case = generate_business_case(state["chapter"], chapter_summaries)
        st.markdown("#### 📊 Real-World Business Scenario")
        st.write(case["executive_summary"])
        st.markdown("**Problem:**")
        st.write(case["problem"])
        st.markdown("**Why this matters:**")
        st.write(case["importance"])
        st.markdown("**Proposed approach:**")
        st.write(case["solution"])
        st.markdown("**Objectives:**")
        for obj in case["objectives"]:
            st.write("•", obj)
        st.markdown("**Financial snapshot:**")
        st.write(case["financials"])
        st.markdown("**Conclusion:**")
        st.write(case["conclusion"])

    elif mode == "📖 Storytelling":
        story, summary = generate_story(state["chapter"], chapter_summaries)
        st.markdown("#### 📖 Storytelling Module")
        st.markdown(story, unsafe_allow_html=True)
        with st.expander("See chapter summary again"):
            st.write(summary)

    elif mode == "🎮 Challenges":
        st.markdown("#### 🎮 Challenge Center")
        st.write("Choose a challenge type to practice this chapter.")

        challenge_type = st.selectbox(
            "Challenge type",
            [
                "Flashcards",
                "MCQ Quiz",
                "Fill in the Blank",
                "Match the Answers",
                "Timed Question",
                "Scenario-Based (with Hint)",
            ],
            key="challenge_type",
        )

        if challenge_type == "Flashcards":
            flashcards_ui(state["chapter"], chapter_questions, chapter_answers)
        elif challenge_type == "MCQ Quiz":
            mcq_ui(state["chapter"], chapter_questions, chapter_answers)
        elif challenge_type == "Fill in the Blank":
            fill_in_blank_ui(state["chapter"], chapter_answers)
        elif challenge_type == "Match the Answers":
            match_answers_ui(state["chapter"], chapter_questions, chapter_answers)
        elif challenge_type == "Timed Question":
            timed_question_ui(state["chapter"], chapter_questions, chapter_answers)
        elif challenge_type == "Scenario-Based (with Hint)":
            scenario_ui(state["chapter"], chapter_summaries, chapter_questions, chapter_answers)

    elif mode == "💬 Ask a Question":
        st.markdown("#### 💬 Chat with your AI-style Tutor")

        # --- session storage for chats (persist across reruns) ---
        if "chat_history" not in st.session_state:
            # { chapter_key: [ {role, content}, ... ] }
            st.session_state.chat_history = {}
        if "chat_archives" not in st.session_state:
            # { chapter_key: [ [msg,msg,...], [msg,msg,...], ... ] }
            st.session_state.chat_archives = {}

        chapter_key = state["chapter"]
        if chapter_key not in st.session_state.chat_history:
            st.session_state.chat_history[chapter_key] = []
        if chapter_key not in st.session_state.chat_archives:
            st.session_state.chat_archives[chapter_key] = []

        # --- controls row: history usage + new/clear buttons ---
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            use_ai = st.toggle("Use AI (ChatGPT) for this chapter", value=True)
        with c2:
            use_history_for_answer = st.toggle(
                "Use chat history", value=True, help="Turn OFF to ignore message history when answering."
            )
        with c3:
            if st.button("🆕 New chat", help="Archive current chat and start fresh (per chapter)."):
                current = st.session_state.chat_history[chapter_key]
                if current:
                    st.session_state.chat_archives[chapter_key].append(current.copy())
                st.session_state.chat_history[chapter_key] = []
                # no explicit rerun; button press already reruns the script
        with c4:
            if st.button("🗑️ Clear chat", help="Delete the current chat messages (per chapter)."):
                st.session_state.chat_history[chapter_key] = []
                # no explicit rerun needed

        # --- answer style ---
        sys_style = st.selectbox(
            "Answer style",
            ["Concise (default)", "Step-by-step (brief)", "Examples first"],
            index=0,
            help="Changes the AI's tone/instructions",
        )
        if sys_style == "Concise (default)":
            system_preamble = (
                "You are a helpful textbook tutor. "
                "Answer clearly and concisely in 4–8 sentences. "
                "Use the provided chapter context and (optionally) the conversation history. "
                "If the answer is not in the context, say you’re unsure and suggest where to look."
            )
        elif sys_style == "Step-by-step (brief)":
            system_preamble = (
                "You are a textbook tutor. Provide a brief, step-by-step explanation (3–6 steps) "
                "grounded in the chapter context (and history if enabled). "
                "If unsure, say so and point to a likely section."
            )
        else:  # Examples first
            system_preamble = (
                "You are a textbook tutor. Start with a simple real-world example, then explain the concept succinctly. "
                "Use the chapter context (and history if enabled). If unsure, say so."
            )

        # --- show current conversation ---
        st.markdown("##### Current conversation")
        for msg in st.session_state.chat_history[chapter_key]:
            st.chat_message(msg["role"]).write(msg["content"])

        # --- show archives (previous chats) with expanders ---
        archives = st.session_state.chat_archives[chapter_key]
        if archives:
            with st.expander(f"Previous chats for this chapter ({len(archives)})"):
                for idx, convo in enumerate(reversed(archives), start=1):
                    st.markdown(f"**Chat #{len(archives) - idx + 1}**")
                    for m in convo:
                        st.chat_message(m["role"]).write(m["content"])
                    st.markdown("---")

        # --- chat-style input ---
        user_q = st.chat_input("Ask your AI tutor something about this chapter…")

        if user_q:
            # display user message
            st.chat_message("user").write(user_q)

            # choose whether to include history in the model call
            history_for_model = st.session_state.chat_history[chapter_key] if use_history_for_answer else []

            ai_answer = None
            if use_ai:
                with st.spinner("Thinking with chapter context…"):
                    try:
                        from ai_helpers import answer_with_ai
                        ai_answer = answer_with_ai(
                            user_q=user_q,
                            chapter=state["chapter"],
                            chapter_summaries=chapter_summaries,
                            chapter_questions=chapter_questions,
                            chapter_answers=chapter_answers,
                            system_preamble=system_preamble,
                            temperature=0.4,
                            chat_history=history_for_model,
                        )
                        st.markdown("**AI-guided answer (grounded in this chapter):**")
                        st.write(ai_answer)
                    except Exception as e:
                        st.error(f"AI module not available yet. {e}")
            else:
                st.caption("AI is off. Toggle it on to get a grounded answer from ChatGPT.")

            # save to current chat
            st.session_state.chat_history[chapter_key].append({"role": "user", "content": user_q})
            if ai_answer:
                st.session_state.chat_history[chapter_key].append({"role": "assistant", "content": ai_answer})

    elif mode == "📈 My Progress":
        st.markdown("#### 📈 Your Progress & XP Overview")
        progress_dashboard_ui()


# --------------- MAIN ROUTER -----------------
def main():
    st.sidebar.title("App Mode")
    view = st.sidebar.radio(
        "Choose interface:",
        ["Interactive Tutor (Single Textbook)","Multi-Book RAG (API)"],
    )
    if view == "Multi-Book RAG (API)":
        multi_book_rag_ui()
    else:
        interactive_tutor_ui()


if __name__ == "__main__":
    main()
