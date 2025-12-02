# App/latestcoding.py
# Streamlit version of the Textbook-to-Interaction (T2I) AI Tutor
# Modes: Story, Business Case, RAG Chat, Challenges with XP

import os
import io
import contextlib
import time
import json
import random
import warnings
from typing import List, Dict, Tuple, Optional

import requests
import torch
import pandas as pd
import numpy as np
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils import logging as hf_logging
from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# -------------------------------------------------------------------
# Basic setup
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
hf_logging.set_verbosity_error()

st.set_page_config(
    page_title="Interactive AI Textbook Tutor (T2I)",
    layout="wide",
)

st.title("📚 Interactive AI Textbook Tutor – Textbook → Interaction (T2I)")

st.markdown(
    """
This app is a **Streamlit port** of your Google Colab AI Tutor prototype.

It follows the same idea:
- 📘 Select a **chapter** from your textbook (OpenStax Workplace Software & Skills)
- 🎭 Choose an **interaction mode** (Story, Business Case, Challenges, Chat)
- 🧠 Learn through **retrieval-augmented generation** and gamified activities

Use the sidebar to pick a chapter and see your **XP & Level**.
"""
)

# -------------------------------------------------------------------
# 1. Data loading & preprocessing
# -------------------------------------------------------------------

CSV_URL = (
    "https://raw.githubusercontent.com/sravani919/AI_Tutor_Interactive_learning/main/"
    "Merged_Chapter_Dataset.csv"
)

def clean_answer_from_question(question: str, answer: str) -> str:
    """
    Same spirit as your original clean_answer_from_question function.
    Removes repetition of question text in the answer.
    """
    q_words = str(question).lower().split()
    a_words = str(answer).strip().split()

    q_set = {w.strip(".,?") for w in q_words}

    start_index = 0
    for i, word in enumerate(a_words):
        clean_word = word.lower().strip(".,?")
        if clean_word not in q_set:
            break
        start_index += 1

    trimmed = a_words[start_index:]
    cleaned = " ".join(trimmed).strip()

    if not cleaned or len(cleaned.split()) <= 3:
        cleaned = "It refers to " + " ".join(a_words)

    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned.rstrip(". ")


@st.cache_data(show_spinner=True)
def load_chapter_dataframe() -> pd.DataFrame:
    st.write("⬇️ Loading chapter dataset from GitHub…")
    df = pd.read_csv(CSV_URL)
    return df


@st.cache_data(show_spinner=True)
def build_chapter_structures() -> Tuple[Dict, Dict, Dict, List[str]]:
    """
    Build chapter_summaries, chapter_questions, chapter_answers
    from the curated dataset (DeepSeek + manual cleaning).
    """
    df_grouped = load_chapter_dataframe()

    chapter_summaries: Dict[str, str] = {}
    chapter_questions: Dict[str, List[str]] = {}
    chapter_answers: Dict[str, List[str]] = {}

    for _, row in df_grouped.iterrows():
        chapter = str(row["chapter"])
        chapter_content = str(row.get("Chapter Content", ""))

        questions_raw = row.get("Questions", [])
        answers_raw = row.get("Answers", [])

        # Parse list-like strings if needed
        if isinstance(questions_raw, str):
            try:
                questions = eval(questions_raw)
            except Exception:
                questions = []
        else:
            questions = questions_raw or []

        if isinstance(answers_raw, str):
            try:
                answers = eval(answers_raw)
            except Exception:
                answers = []
        else:
            answers = answers_raw or []

        chapter_summaries[chapter] = (
            chapter_content if chapter_content else "No summary available."
        )
        chapter_questions[chapter] = questions[:5] if questions else []

        cleaned_answers = []
        if questions and answers:
            for q, a in zip(questions[:5], answers[:5]):
                cleaned_answers.append(clean_answer_from_question(q, a))

        chapter_answers[chapter] = cleaned_answers

    chapters_sorted = sorted(chapter_summaries.keys(), key=lambda x: str(x))
    return chapter_summaries, chapter_questions, chapter_answers, chapters_sorted


# -------------------------------------------------------------------
# 2. Embeddings + FAISS store (for retrieval)
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def build_retriever(chapter_summaries, chapter_questions, chapter_answers):
    """
    Build a semantic index of:
      - chapter summaries
      - Q/A pairs as small documents
    """
    docs: List[Document] = []

    for chap, summary in chapter_summaries.items():
        docs.append(
            Document(
                page_content=f"Chapter {chap} summary:\n{summary}",
                metadata={"chapter": chap, "type": "summary"},
            )
        )

        qs = chapter_questions.get(chap, [])
        ans = chapter_answers.get(chap, [])
        for q, a in zip(qs, ans):
            docs.append(
                Document(
                    page_content=f"Question: {q}\nAnswer: {a}",
                    metadata={"chapter": chap, "type": "qa"},
                )
            )

    if not docs:
        raise RuntimeError("No documents built from dataset; check CSV / columns.")

    st.write("📐 Building embeddings and FAISS index…")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 6})
    return retriever


# -------------------------------------------------------------------
# 3. Load local HF model (Falcon 1B) and pipeline
# -------------------------------------------------------------------

PRIVATE_MODEL_ID = "tiiuae/falcon-rw-1b"  # same as your Colab public tutor

@st.cache_resource(show_spinner=True)
def load_local_tutor_model():
    """
    Load Falcon-RW-1B as a HuggingFace text-generation pipeline.
    This mirrors your Colab deployment (no quantization here yet).
    """
    st.write("🤖 Loading Falcon-based AI tutor model…")
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        tokenizer = AutoTokenizer.from_pretrained(PRIVATE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(PRIVATE_MODEL_ID)

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
    )
    return gen_pipeline


def generate_tutor_answer(
    gen_pipeline,
    question: str,
    context_docs: List[Document],
    chapter: Optional[str] = None,
    mode: str = "qa",
) -> str:
    """
    Generic RAG prompt for chat / story / case.
    Mode can be: "qa", "story", "case".
    """
    if not context_docs:
        context_text = "No chapter context was retrieved. Answer based on general understanding."
    else:
        snippets = []
        for d in context_docs:
            prefix = f"[Chapter {d.metadata.get('chapter','?')} - {d.metadata.get('type','?')}]"
            snippets.append(f"{prefix}\n{d.page_content}")
        context_text = "\n\n".join(snippets[:6])

    chap_hint = f"Current chapter: {chapter}\n" if chapter else ""

    if mode == "story":
        system_hint = (
            "You are an instructional designer. Using ONLY the context, "
            "write a short workplace story (120–180 words) that illustrates 2–3 key ideas. "
            "Bold key terms on first mention. End with one reflective question."
        )
    elif mode == "case":
        system_hint = (
            "You are creating a brief business case scenario from a textbook chapter. "
            "Using ONLY the context, write a 120–180 word workplace scenario with a clear decision point "
            "and two options. Then suggest what a good choice would be and why."
        )
    else:  # "qa"
        system_hint = (
            "You are a friendly AI tutor helping a student understand a textbook. "
            "Use ONLY the given context. If the answer is not clearly in the context, "
            "say you are not sure and suggest where they might review in the chapter."
        )

    prompt = f"""{system_hint}

{chap_hint}Context:
{context_text}

Learner prompt: {question}

Answer:
"""

    out = gen_pipeline(prompt)
    if isinstance(out, list) and out:
        text = out[0].get("generated_text", "").strip()
    else:
        text = str(out)

    if prompt in text:
        text = text.split(prompt, 1)[-1].strip()

    return text


# -------------------------------------------------------------------
# 4. XP system (gamification)
# -------------------------------------------------------------------

XP_THRESHOLDS = [
    (1, 50),
    (2, 75),
    (3, 100),
    (4, 125),
    (5, 150),
    (6, 175),
    (7, 200),
    (8, 225),
    (9, 250),
]  # level : xp to next

def init_xp_state():
    if "xp" not in st.session_state:
        st.session_state.xp = 0
    if "level" not in st.session_state:
        st.session_state.level = 1

def add_xp(amount: int):
    st.session_state.xp += amount
    # update level
    remaining = st.session_state.xp
    level = 1
    for lv, need in XP_THRESHOLDS:
        if remaining >= need:
            remaining -= need
            level = lv + 1
        else:
            break
    st.session_state.level = level


def xp_to_next_level() -> Tuple[int, int]:
    """
    Return (xp_in_current_level, xp_needed_this_level)
    so we can show progress bar.
    """
    total = st.session_state.xp
    level = 1
    for lv, need in XP_THRESHOLDS:
        if total >= need:
            total -= need
            level = lv + 1
        else:
            return total, need
    # max level reached
    return 0, 1


# -------------------------------------------------------------------
# 5. Challenges (Flashcards, MCQ, Fill-in, Match, Timed, Scenario)
# -------------------------------------------------------------------

def build_quiz_items_for_chapter(
    chapter: str,
    chapter_questions: Dict[str, List[str]],
    chapter_answers: Dict[str, List[str]],
) -> List[Dict]:
    qs = chapter_questions.get(chapter, [])
    ans = chapter_answers.get(chapter, [])

    if not qs or not ans:
        return []

    all_answers = []
    for a_list in chapter_answers.values():
        all_answers.extend(a_list)
    all_answers = [a for a in all_answers if a]

    items = []
    for q, a in zip(qs, ans):
        correct = a
        distractors = [d for d in all_answers if d != correct]
        random.shuffle(distractors)
        distractors = distractors[:3]

        opts = distractors + [correct]
        random.shuffle(opts)

        items.append(
            {
                "question": q,
                "options": opts,
                "correct": correct,
            }
        )
    return items


def show_flashcards(chapter: str, qs: List[str], ans: List[str]):
    st.markdown("##### Flashcards – Flip for the answer")

    if "flash_idx" not in st.session_state:
        st.session_state.flash_idx = 0

    if not qs or not ans:
        st.info("No flashcards available for this chapter.")
        return

    idx = st.session_state.flash_idx % len(qs)

    st.markdown(f"**Card {idx+1}/{len(qs)}**")
    st.markdown(f"**Q:** {qs[idx]}")

    if st.button("Show answer"):
        st.markdown(f"**A:** {ans[idx]}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Got it"):
            add_xp(5)  # flashcards XP
            st.success("+5 XP")
            st.session_state.flash_idx = (st.session_state.flash_idx + 1) % len(qs)
            st.experimental_rerun()
    with col2:
        if st.button("🔁 Next card"):
            st.session_state.flash_idx = (st.session_state.flash_idx + 1) % len(qs)
            st.experimental_rerun()


def show_mcq_quiz(chapter: str, items: List[Dict]):
    st.markdown("##### Multiple-Choice Questions")

    if not items:
        st.info("No MCQ items available for this chapter.")
        return

    if "mcq_idx" not in st.session_state:
        st.session_state.mcq_idx = 0
        st.session_state.mcq_score = 0

    idx = st.session_state.mcq_idx
    if idx >= len(items):
        st.success(
            f"Quiz finished! ✅ You answered {st.session_state.mcq_score}/{len(items)} correctly."
        )
        if st.button("Restart MCQ"):
            st.session_state.mcq_idx = 0
            st.session_state.mcq_score = 0
            st.experimental_rerun()
        return

    item = items[idx]
    st.markdown(f"**Q{idx+1}. {item['question']}**")

    choice = st.radio(
        "Choose your answer:", item["options"], key=f"mcq_choice_{idx}"
    )

    if st.button("Submit", key=f"mcq_submit_{idx}"):
        if choice == item["correct"]:
            st.session_state.mcq_score += 1
            add_xp(10)  # MCQ XP
            st.success("✅ Correct! +10 XP")
        else:
            st.error("❌ Not quite.")
            st.info(f"Correct answer: **{item['correct']}**")

        st.session_state.mcq_idx += 1
        st.experimental_rerun()


def show_fill_in_blank(chapter: str, qs: List[str], ans: List[str]):
    st.markdown("##### Fill in the blanks")

    if "fib_idx" not in st.session_state:
        st.session_state.fib_idx = 0

    if not qs or not ans:
        st.info("No fill-in-the-blank items available for this chapter.")
        return

    idx = st.session_state.fib_idx % len(qs)
    q = qs[idx]
    a = ans[idx]

    # Simple: show the question and ask for short answer
    st.markdown(f"**Prompt:** {q}")
    user_answer = st.text_input("Type your answer:", key=f"fib_ans_{idx}")

    if st.button("Check answer", key=f"fib_check_{idx}"):
        if user_answer.strip():
            if user_answer.lower().strip() in a.lower():
                add_xp(10)
                st.success(f"✅ Close enough! Ideal answer: {a}  (+10 XP)")
            else:
                st.error("❌ Not quite.")
                st.info(f"Ideal answer: **{a}**")
            st.session_state.fib_idx = (st.session_state.fib_idx + 1) % len(qs)
            st.experimental_rerun()
        else:
            st.warning("Please type something first.")


def show_match(chapter: str, qs: List[str], ans: List[str]):
    st.markdown("##### Match the answers")

    if not qs or not ans:
        st.info("No matching items available for this chapter.")
        return

    # Use limited subset
    n = min(4, len(qs))
    qs_sub = qs[:n]
    ans_sub = ans[:n]
    shuffled_ans = ans_sub.copy()
    random.shuffle(shuffled_ans)

    st.write("Match each question/term to the correct answer:")

    correct = 0
    for i, q in enumerate(qs_sub):
        choice = st.selectbox(
            f"{i+1}. {q}",
            ["(select)"] + shuffled_ans,
            key=f"match_{chapter}_{i}",
        )
        if choice == ans_sub[i]:
            correct += 1

    if st.button("Check matching"):
        st.info(f"You matched **{correct}/{n}** correctly.")
        if correct == n:
            add_xp(12)  # match XP
            st.success("🎉 Perfect! +12 XP")


def show_timed_question(chapter: str, qs: List[str], ans: List[str]):
    st.markdown("##### Timed question (15 seconds)")

    if not qs or not ans:
        st.info("No timed items available for this chapter.")
        return

    if "timed_idx" not in st.session_state:
        st.session_state.timed_idx = 0
        st.session_state.timed_start = None

    idx = st.session_state.timed_idx % len(qs)
    q = qs[idx]
    a = ans[idx]

    if st.session_state.timed_start is None:
        st.session_state.timed_start = time.time()

    elapsed = int(time.time() - st.session_state.timed_start)
    remaining = max(0, 15 - elapsed)

    st.markdown(f"**Prompt:** {q}")
    st.caption(f"⏱️ Time left (soft): {remaining} seconds")

    user_answer = st.text_input("Your answer:", key=f"timed_ans_{idx}")

    if st.button("Submit timed", key=f"timed_submit_{idx}"):
        elapsed = int(time.time() - st.session_state.timed_start)
        if user_answer.lower().strip() in a.lower():
            # reward more XP if within time
            if elapsed <= 15:
                add_xp(15)
                st.success(f"✅ Correct and in time! (+15 XP). Ideal answer: {a}")
            else:
                add_xp(8)
                st.success(
                    f"✅ Correct but a bit slow (+8 XP). Time: {elapsed}s. Ideal answer: {a}"
                )
        else:
            st.error("❌ Not quite.")
            st.info(f"Ideal answer: **{a}** (time used: {elapsed}s)")
        st.session_state.timed_idx = (st.session_state.timed_idx + 1) % len(qs)
        st.session_state.timed_start = None
        st.experimental_rerun()


def show_scenario_hint(chapter: str, qs: List[str], ans: List[str]):
    st.markdown("##### Scenario-based (hint) question")

    if not qs or not ans:
        st.info("No scenario items available for this chapter.")
        return

    if "sc_idx" not in st.session_state:
        st.session_state.sc_idx = 0

    idx = st.session_state.sc_idx % len(qs)
    q = qs[idx]
    a = ans[idx]

    st.markdown(f"**Scenario:** Imagine a situation where: {q}")
    if st.button("Give me a hint"):
        st.info(f"💡 Hint: Think about: **{a[: min(40, len(a))]}...**")

    user_answer = st.text_area("Your answer in your own words:", key=f"sc_user_{idx}")

    if st.button("Check scenario", key=f"sc_check_{idx}"):
        # Just show model answer, reward XP if non-empty
        if user_answer.strip():
            add_xp(15)
            st.success("✅ Nice effort! +15 XP")
        st.info(f"Model-aligned answer:\n\n**{a}**")
        st.session_state.sc_idx = (st.session_state.sc_idx + 1) % len(qs)
        st.experimental_rerun()


# -------------------------------------------------------------------
# 6. Main Streamlit UI
# -------------------------------------------------------------------

def main():
    init_xp_state()

    (
        chapter_summaries,
        chapter_questions,
        chapter_answers,
        chapters_sorted,
    ) = build_chapter_structures()
    retriever = build_retriever(chapter_summaries, chapter_questions, chapter_answers)
    gen_pipeline = load_local_tutor_model()

    # Sidebar
    with st.sidebar:
        st.header("📘 Chapter & Progress")

        chapter_choice = st.selectbox(
            "Choose chapter:",
            options=chapters_sorted,
            index=0,
        )

        # XP / level view
        xp_cur, xp_need = xp_to_next_level()
        st.markdown(f"**Level:** {st.session_state.level}")
        st.markdown(f"**Total XP:** {st.session_state.xp}")
        st.progress(min(xp_cur / xp_need, 1.0))

        st.caption("Complete challenges to earn XP and level up.")

        st.markdown("---")
        st.markdown("### Modes")
        st.caption(
            """
- **Story** – narrative teaching
- **Business Case** – workplace scenario
- **Chat Tutor** – RAG-based Q&A
- **Challenges** – flashcards, MCQ, fill-in, match, timed, scenarios
"""
        )

    # Main tabs (roughly matching your modes)
    tab_story, tab_case, tab_chat, tab_challenges = st.tabs(
        ["📖 Story Mode", "🏢 Business Case", "💬 Chat Tutor", "🎮 Challenges"]
    )

    # ---------------- Story Mode ----------------
    with tab_story:
        st.subheader("📖 Storytelling from Chapter Content")

        summary = chapter_summaries.get(chapter_choice, "")
        st.markdown("**Base chapter summary (used as context):**")
        st.info(summary)

        if st.button("Generate Story"):
            with st.spinner("Generating story from chapter context…"):
                docs = retriever.get_relevant_documents(summary or chapter_choice)
                story_text = generate_tutor_answer(
                    gen_pipeline,
                    question="Create a short narrative that teaches the key ideas.",
                    context_docs=docs,
                    chapter=chapter_choice,
                    mode="story",
                )
            st.markdown("### Story")
            st.write(story_text)

    # ---------------- Business Case Mode ----------------
    with tab_case:
        st.subheader("🏢 Business Case Scenario")

        st.markdown(
            "Generate a workplace scenario based on this chapter. "
            "Useful for **applied reasoning** and **decision-making practice**."
        )

        if st.button("Generate Business Case"):
            with st.spinner("Generating business case…"):
                docs = retriever.get_relevant_documents(chapter_choice)
                case_text = generate_tutor_answer(
                    gen_pipeline,
                    question="Create a brief workplace scenario with a decision point and two options.",
                    context_docs=docs,
                    chapter=chapter_choice,
                    mode="case",
                )
            st.markdown("### Business Case")
            st.write(case_text)

    # ---------------- Chat Tutor Mode ----------------
    with tab_chat:
        st.subheader("💬 RAG-based Chat Tutor")

        st.markdown(
            """
Ask textbook questions.  
The tutor will retrieve relevant Q&A + summaries from your DeepSeek dataset and answer using Falcon.
"""
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for speaker, msg in st.session_state.chat_history:
            if speaker == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Tutor:** {msg}")

        default_q = "Explain the main idea of this chapter in simple terms."
        user_q = st.text_input("Your question:", value=default_q, key="chat_q")

        if st.button("Ask Tutor"):
            if user_q.strip():
                with st.spinner("Retrieving context and generating answer…"):
                    docs = retriever.get_relevant_documents(user_q)
                    # filter by chapter
                    filtered = [
                        d for d in docs if str(d.metadata.get("chapter")) == str(chapter_choice)
                    ]
                    if filtered:
                        docs = filtered
                    ans = generate_tutor_answer(
                        gen_pipeline,
                        question=user_q,
                        context_docs=docs,
                        chapter=chapter_choice,
                        mode="qa",
                    )
                st.session_state.chat_history.append(("user", user_q))
                st.session_state.chat_history.append(("tutor", ans))
                st.experimental_rerun()

    # ---------------- Challenges Mode ----------------
    with tab_challenges:
        st.subheader("🎮 Interactive Challenges & XP")

        qs = chapter_questions.get(chapter_choice, [])
        ans = chapter_answers.get(chapter_choice, [])

        challenge_type = st.selectbox(
            "Choose challenge type:",
            [
                "Flashcards",
                "MCQ Quiz",
                "Fill in the Blanks",
                "Match the Answers",
                "Timed Question",
                "Scenario (Hints)",
            ],
        )

        items = build_quiz_items_for_chapter(
            chapter_choice, chapter_questions, chapter_answers
        )

        if challenge_type == "Flashcards":
            show_flashcards(chapter_choice, qs, ans)
        elif challenge_type == "MCQ Quiz":
            show_mcq_quiz(chapter_choice, items)
        elif challenge_type == "Fill in the Blanks":
            show_fill_in_blank(chapter_choice, qs, ans)
        elif challenge_type == "Match the Answers":
            show_match(chapter_choice, qs, ans)
        elif challenge_type == "Timed Question":
            show_timed_question(chapter_choice, qs, ans)
        elif challenge_type == "Scenario (Hints)":
            show_scenario_hint(chapter_choice, qs, ans)

    st.markdown("---")
    st.caption(
        "Backend: Falcon-RW-1B · RAG with FAISS & MiniLM · Modes: Story, Case, Chat, Challenges with XP."
    )


if __name__ == "__main__":
    main()
