# challenges.py

import streamlit as st
import random
import re
import time
import matplotlib.pyplot as plt
from difflib import SequenceMatcher


# ----------------- XP CONFIG -----------------

XP_PER_CHALLENGE = {
    "Flashcards": 5,
    "MCQ Quiz": 10,
    "Fill in the Blank": 10,
    "Match the Answers": 12,
    "Timed Question": 15,
    "Scenario-Based (with Hint)": 15,
}


# ----------------- STATE INIT -----------------

def init_tutor_state():
    """
    Ensure st.session_state.tutor exists with all the fields we use.
    Can be called from app.py before using any challenge.
    """
    if "tutor" not in st.session_state:
        st.session_state.tutor = {
            "name": "",
            "chapter": None,
            "mode": None,
            "xp": 0,
            "level": 1,
            "history": [],

            # Flashcards
            "flashcard_index": 0,
            "flashcard_flipped": False,

            # MCQ
            "mcq_index": 0,
            "mcq_score": 0,
            "mcq_options": {},  # <-- store shuffled options per (chapter, question)

            # Fill in the blank (typing)
            "fib_index": 0,
            "fib_lives": 3,

            # Match answers
            "match_answers": None,

            # Timed questions
            "timed": None,

            # Scenario-based
            "scenario": None,
        }


def award_xp(label: str):
    """Give XP for a specific challenge type and handle basic leveling."""
    state = st.session_state.tutor
    gain = XP_PER_CHALLENGE.get(label, 5)
    state["xp"] += gain
    state["history"].append(f"{label} +{gain} XP")

    # Very simple leveling rule: every 50×level XP
    if state["xp"] >= state["level"] * 50:
        state["level"] += 1
        st.success(f"🎉 Level up! You reached level {state['level']}.")


# ----------------- SMALL UTIL -----------------

def best_qa_match(user_q: str, chapter: str, chapter_questions, chapter_answers):
    """Simple text similarity to find best QA match (for chat mode)."""
    q_list = chapter_questions.get(chapter, [])
    a_list = chapter_answers.get(chapter, [])
    if not q_list or not a_list:
        return None, None
    scores = []
    for q in q_list:
        s = SequenceMatcher(None, user_q.lower(), q.lower()).ratio()
        scores.append(s)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return q_list[best_idx], a_list[best_idx]


def _clean_chapter_name(chapter_name: str) -> str:
    """Local cleaner for scenario titles, independent from app.py."""
    return re.sub(r"^\d+(\.\d+)?\s*", "", str(chapter_name)).strip()


# ----------------- SIDEBAR (PROFILE/XP SNAPSHOT) -----------------

def tutor_sidebar():
    state = st.session_state.tutor
    st.sidebar.markdown("### 🎮 Tutor Progress")
    if state["name"]:
        st.sidebar.write(f"**Learner:** {state['name']}")
    st.sidebar.write(f"**Level:** {state['level']}")
    st.sidebar.write(f"**XP:** {state['xp']}")
    if state["history"]:
        st.sidebar.markdown("**Recent activity:**")
        for h in state["history"][-5:]:
            st.sidebar.caption("• " + h)


# ----------------- CHALLENGE UIs -----------------
# 1) FLASHCARDS
# 2) MCQ
# 3) FILL IN THE BLANK (typing)
# 4) MATCH THE ANSWERS (dropdown)
# 5) TIMED QUESTION
# 6) SCENARIO-BASED (WITH HINT)
# -------------------------------------------------

def flashcards_ui(chapter, chapter_questions, chapter_answers):
    state = st.session_state.tutor
    q_list = chapter_questions.get(chapter, [])
    a_list = chapter_answers.get(chapter, [])
    pairs = list(zip(q_list, a_list))
    if not pairs:
        st.info("No flashcards available for this chapter yet.")
        return

    idx = state["flashcard_index"]
    if idx >= len(pairs):
        st.success("You’ve finished all flashcards for this chapter! 🎉")
        if st.button("Restart flashcards", key="flash_restart"):
            state["flashcard_index"] = 0
            state["flashcard_flipped"] = False
            st.rerun()
        return

    q, a = pairs[idx]
    st.markdown(f"**Card {idx+1} / {len(pairs)}**")

    # ---- 1) read current state ----
    flipped = state.get("flashcard_flipped", False)

    # ---- 2) draw buttons + capture clicks (but don't render Q/A yet) ----
    flip_clicked = got_it_clicked = next_clicked = False

    if not flipped:
        # only show flip button
        flip_clicked = st.button("Flip card", key=f"flip_{idx}")
    else:
        # show answer + two actions
        col1, col2 = st.columns(2)
        with col1:
            got_it_clicked = st.button("👍 I got it", key=f"got_{idx}")
        with col2:
            next_clicked = st.button("➡️ Next card", key=f"next_{idx}")

    # ---- 3) update state based on clicks ----
    if flip_clicked:
        state["flashcard_flipped"] = True
        flipped = True

    elif got_it_clicked:
        award_xp("Flashcards")
        state["flashcard_index"] += 1
        state["flashcard_flipped"] = False
        st.rerun()

    elif next_clicked:
        state["flashcard_index"] += 1
        state["flashcard_flipped"] = False
        st.rerun()

    # ---- 4) now render based on (possibly updated) state ----
    if not state.get("flashcard_flipped", False):
        st.write("📘 **Question:**", q)
    else:
        st.write("💡 **Answer:**", a)



def mcq_ui(chapter, chapter_questions, chapter_answers):
    """
    MCQ where options are shuffled ONCE per question,
    feedback is shown immediately,
    and user clicks 'Next Question' to proceed.
    """
    state = st.session_state.tutor
    q_list = chapter_questions.get(chapter, [])
    a_list = chapter_answers.get(chapter, [])
    pairs = list(zip(q_list, a_list))
    if not pairs:
        st.info("No quiz questions available for this chapter yet.")
        return

    idx = state["mcq_index"]
    if idx >= min(len(pairs), 5):
        st.success(f"🎉 Quiz finished! Score: {state['mcq_score']} / {idx * 10}")
        if st.button("Restart quiz"):
            state["mcq_index"] = 0
            state["mcq_score"] = 0
            state["mcq_options"] = {}
            state["mcq_feedback"] = None
            st.rerun()
        return

    q, correct = pairs[idx]
    st.markdown(f"**Question {idx+1}**")
    st.write(q)

    # Init dicts
    if "mcq_options" not in state:
        state["mcq_options"] = {}
    if "mcq_feedback" not in state:
        state["mcq_feedback"] = None

    q_key = f"{chapter}_{idx}"

    # Create options only once per question
    if q_key not in state["mcq_options"]:
        other_answers = [a for a in a_list if a != correct]
        distractors = random.sample(other_answers, k=min(3, len(other_answers))) if other_answers else []
        options = [correct] + distractors
        random.shuffle(options)
        state["mcq_options"][q_key] = options

    options = state["mcq_options"][q_key]
    choice = st.radio("Choose an answer:", options, key=f"mcq_{q_key}")

    # Submit button
    if st.button("Submit", key=f"mcq_submit_{q_key}"):
        if choice == correct:
            state["mcq_feedback"] = ("correct", correct)
            award_xp("MCQ Quiz")
            state["mcq_score"] += 10
        else:
            state["mcq_feedback"] = ("wrong", correct)

    # Show feedback if available
    if state["mcq_feedback"]:
        status, correct_ans = state["mcq_feedback"]
        if status == "correct":
            st.success(f"✅ Correct! +10 XP\n\nThe right answer was: {correct_ans}")
        else:
            st.error(f"❌ Incorrect. The correct answer was: {correct_ans}")

        # Show Next Question button
        if st.button("➡️ Next Question"):
            state["mcq_index"] += 1
            state["mcq_feedback"] = None
            st.rerun()

def fill_in_blank_ui(chapter, chapter_answers):
    state = st.session_state.tutor
    answers = chapter_answers.get(chapter, [])
    if not answers:
        st.info("Not enough data for fill-in-the-blank in this chapter.")
        return

    # --- initialise / reset order per chapter (shuffled) ---
    if state.get("fib_chapter") != chapter or "fib_order" not in state:
        indices = list(range(len(answers)))
        random.shuffle(indices)
        state["fib_order"] = indices
        state["fib_chapter"] = chapter
        state["fib_index"] = 0
        state["fib_lives"] = 3
        state["fib_attempts"] = 0

    order = state["fib_order"]
    idx = state["fib_index"]
    lives = state["fib_lives"]

    # out of lives
    if lives <= 0:
        st.error("Game over – you ran out of lives. 💀")
        if st.button("Restart fill-in-the-blank"):
            indices = list(range(len(answers)))
            random.shuffle(indices)
            state["fib_order"] = indices
            state["fib_index"] = 0
            state["fib_lives"] = 3
            state["fib_attempts"] = 0
        return

    # finished round
    if idx >= min(len(order), 5):
        st.success("Nice work — you’ve completed the fill-in-the-blank round! 🎉")
        if st.button("Play again"):
            indices = list(range(len(answers)))
            random.shuffle(indices)
            state["fib_order"] = indices
            state["fib_index"] = 0
            state["fib_lives"] = 3
            state["fib_attempts"] = 0
        return

    real_idx = order[idx]
    full = answers[real_idx]

    words = re.findall(r"\b\w+\b", full)
    keyword = next((w for w in words if len(w) > 4), words[0] if words else None)
    if not keyword:
        st.info("This sentence is too short to blank out. Moving on.")
        state["fib_index"] += 1
        return

    sentence = full.replace(keyword, "____", 1)
    st.markdown(f"**Question {idx+1} / {min(len(order), 5)}**")
    st.markdown(f"**Lives:** {'❤️' * lives}")
    st.write(sentence)

    guess = st.text_input("Type the missing word:", key=f"fib_{idx}")

    col1, col2 = st.columns(2)
    with col1:
        check_clicked = st.button("Check", key=f"fib_check_{idx}")
    with col2:
        next_clicked = st.button("Next", key=f"fib_next_{idx}")

    # when user clicks "Check"
    if check_clicked:
        if guess.strip().lower() == keyword.lower():
            st.success(f"✅ Correct! The word was **{keyword}**. +10 XP")
            award_xp("Fill in the Blank")
            state["fib_attempts"] = 0
        else:
            state["fib_attempts"] += 1
            state["fib_lives"] -= 1

            if state["fib_attempts"] == 1:
                st.warning(
                    f"Hint: the word starts with **{keyword[0].upper()}** "
                    f"and has **{len(keyword)}** letters."
                )
            elif state["fib_attempts"] >= 2 or state["fib_lives"] <= 0:
                st.error(f"❌ The correct word was **{keyword}**.")
                state["fib_attempts"] = 0
                # move automatically to next question
                state["fib_index"] += 1
                return
            else:
                st.error(f"Not quite. You still have {state['fib_lives']} lives.")

    # when user clicks "Next"
    if next_clicked:
        state["fib_index"] += 1
        state["fib_attempts"] = 0

# ----------------- MATCH THE ANSWERS -----------------

def match_answers_ui(chapter, chapter_questions, chapter_answers):
    """Dropdown-based match-the-answer game in Streamlit."""
    state = st.session_state.tutor
    questions = chapter_questions.get(chapter, [])
    answers = chapter_answers.get(chapter, [])

    if not questions or not answers or len(questions) < 3:
        st.info("Not enough Q&A pairs for Match the Answers in this chapter.")
        return

    st.markdown("### 🧩 Match the Answers")
    st.write("Match each question with the correct answer from the dropdowns.")

    # Use at most 5 pairs
    pairs = list(zip(questions, answers))[:5]
    correct_answers = [a for _, a in pairs]

    # Initialise state for this challenge / chapter
    if (
        state.get("match_answers") is None
        or state["match_answers"].get("chapter") != chapter
    ):
        # Shuffle once and keep this order
        shuffled = correct_answers.copy()
        random.shuffle(shuffled)
        state["match_answers"] = {
            "chapter": chapter,
            "options": shuffled,            # stable shuffled list
            "selections": [""] * len(pairs),
            "submitted": False,
            "score": 0,
        }

    ms = state["match_answers"]
    options = ms["options"]   # stable across reruns

    # Render each question + dropdown
    for i, (q, _) in enumerate(pairs):
        st.markdown(f"**Q{i+1}. {q}**")
        key = f"match_q_{chapter}_{i}"

        choice = st.selectbox(
            "Choose answer",
            ["Select an answer"] + options,
            key=key,
        )
        ms["selections"][i] = choice if choice != "Select an answer" else ""

    if st.button("Check Matches"):
        score = 0
        st.markdown("---")
        st.markdown("### 📊 Match Results")

        for i, (_, correct) in enumerate(pairs):
            user_ans = ms["selections"][i] or "No answer selected"
            if user_ans == correct:
                st.success(f"Q{i+1}: ✅ Correct")
                score += 1
            else:
                st.error(f"Q{i+1}: ❌ Incorrect — correct answer: {correct}")

        ms["submitted"] = True
        ms["score"] = score

        st.markdown(f"**Final Score:** {score} / {len(pairs)}")

        if score == len(pairs):
            award_xp("Match the Answers")
            st.success("🏆 Perfect score! XP awarded.")
        else:
            st.info("💡 Review the correct answers and try again for full XP.")

    if ms["submitted"]:
        if st.button("Restart Match the Answers"):
            # Re-initialise with a new shuffle
            shuffled = correct_answers.copy()
            random.shuffle(shuffled)
            state["match_answers"] = {
                "chapter": chapter,
                "options": shuffled,
                "selections": [""] * len(pairs),
                "submitted": False,
                "score": 0,
            }
            st.rerun()

# ----------------- TIMED QUESTION -----------------
# ----------------- TIMED QUESTION -----------------

def timed_question_ui(chapter, chapter_questions, chapter_answers):
    """Timed MCQ challenge with visible countdown + next button after feedback."""
    state = st.session_state.tutor
    questions = chapter_questions.get(chapter, [])
    answers = chapter_answers.get(chapter, [])

    if not questions or not answers:
        st.info("No questions available for the timed challenge in this chapter.")
        return

    # Init state
    if state.get("timed") is None or state["timed"].get("chapter") != chapter:
        state["timed"] = {
            "chapter": chapter,
            "current_q": 0,
            "score": 0,
            "start_time": None,
            "max_q": 5,
            "options": {},
            "answered": False,
            "feedback": "",
        }

    timed = state["timed"]

    # Finished?
    if timed["current_q"] >= min(len(questions), timed["max_q"]):
        st.markdown("### 🎉 Timed Challenge Complete!")
        st.write(f"**Total Score:** {timed['score']} points")
        if st.button("Restart Timed Challenge"):
            state["timed"] = {
                "chapter": chapter,
                "current_q": 0,
                "score": 0,
                "start_time": None,
                "max_q": 5,
                "options": {},
                "answered": False,
                "feedback": "",
            }
        return

    idx = timed["current_q"]
    q = questions[idx]
    correct = answers[idx]

    # Start timer
    if timed["start_time"] is None:
        timed["start_time"] = time.time()

    elapsed_so_far = time.time() - timed["start_time"]
    remaining = max(0, 15 - elapsed_so_far)
    remaining_int = int(remaining)

    st.markdown(f"### 🕒 Timed Question {timed['current_q'] + 1} / {timed['max_q']}")

    col_main, col_timer = st.columns([4, 1])

    with col_main:
        st.write(q)
    with col_timer:
        st.markdown(
            f"""
            <div style="text-align:center; border:1px solid #ddd; padding:8px; border-radius:8px;">
                <div style="font-size:22px;">⏱</div>
                <div style="font-size:20px; font-weight:bold;">{remaining_int:02d}s</div>
                <div style="font-size:11px; color:#666;">to earn XP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    q_key = f"{chapter}_{idx}"

    # Build options once per question
    if q_key not in timed["options"]:
        other_answers = [a for a in answers if a != correct]
        distractors = random.sample(other_answers, k=min(3, len(other_answers))) if other_answers else []
        options = [correct] + distractors
        random.shuffle(options)
        timed["options"][q_key] = options

    options = timed["options"][q_key]
    choice = st.radio("Choose an answer:", options, key=f"timed_choice_{q_key}")

    # Submit button logic
    if not timed["answered"]:
        if st.button("Submit answer", key=f"timed_submit_{q_key}"):
            end_time = time.time()
            elapsed = end_time - timed["start_time"]
            fast_enough = elapsed <= 15

            if choice == correct:
                if fast_enough:
                    timed["feedback"] = f"✅ Correct in {elapsed:.1f}s! +15 XP"
                    award_xp("Timed Question")
                    timed["score"] += 15
                else:
                    timed["feedback"] = f"✅ Correct but too slow ({elapsed:.1f}s >15s). No XP this time."
            else:
                timed["feedback"] = f"❌ Incorrect. Correct answer: {correct} (took {elapsed:.1f}s)."

            timed["answered"] = True
            timed["start_time"] = None

    # Show feedback
    if timed["answered"]:
        if "✅" in timed["feedback"]:
            st.success(timed["feedback"])
        else:
            st.error(timed["feedback"])

        # Show Next button instead of resubmitting
        if st.button("➡️ Next Question"):
            timed["current_q"] += 1
            timed["answered"] = False
            timed["feedback"] = ""
            timed["start_time"] = None
            st.rerun()

    st.caption("Answer within 15 seconds to earn XP. Click Next to continue.")

# ----------------- SCENARIO-BASED (WITH HINT) -----------------
def generate_scenario(chapter, chapter_summaries, chapter_questions, chapter_answers):
    summary = chapter_summaries.get(chapter, "No summary available.")
    questions = chapter_questions.get(chapter, [])
    answers = chapter_answers.get(chapter, [])

    if not questions or not answers:
        return None

    idx = random.randint(0, min(len(questions), len(answers)) - 1)
    q = questions[idx]
    a = answers[idx]

    roles = ["data analyst", "IT coordinator", "junior accountant", "BI consultant"]
    names = ["Jordan", "Alex", "Taylor", "Sam", "Jamie", "Morgan"]
    actor = f"{random.choice(names)}, a {random.choice(roles)}"

    goal_text = re.sub(
        r"(?i)\bwhat is\b|\bhow can\b|\bdescribe\b|\bexplain\b",
        "",
        q,
    ).strip()
    goal = goal_text.capitalize() or f"Apply {_clean_chapter_name(chapter)} in a real task"

    base_steps = re.split(r"[.]", a)
    steps = [s.strip() for s in base_steps if s.strip()]
    if len(steps) < 3:
        steps += [
            "Reviewed the documentation",
            "Consulted with a senior colleague",
            "Tested the idea on a sample dataset",
        ]

    distractors = [
        "Skipped validation and sent the report immediately.",
        "Ignored the data and made a decision based only on intuition.",
        "Shared an outdated file without checking its accuracy.",
        "Used a completely unrelated tool instead of the one covered in this chapter.",
    ]

    correct_option = f"Applied the concepts correctly: {a}"
    options = random.sample(distractors, k=3) + [correct_option]
    random.shuffle(options)

    return {
        "title": f"📘 Use Case: Applying {_clean_chapter_name(chapter)}",
        "actor": actor,
        "goal": goal,
        "summary": summary,
        "success_path": steps[:4],
        "failure_paths": distractors,
        "question": f"What should {actor.split(',')[0]} do next to achieve their goal?",
        "options": options,
        "correct": correct_option,
        "hint": f"💡 Think about the main purpose of {_clean_chapter_name(chapter)}: what is it supposed to help with?",
    }


def scenario_ui(chapter, chapter_summaries, chapter_questions, chapter_answers):
    state = st.session_state.tutor

    # 👉 Only generate once per chapter and store in state
    if (
        state.get("scenario") is None
        or state["scenario"].get("chapter") != chapter
        or state["scenario"].get("data") is None
    ):
        scenario = generate_scenario(chapter, chapter_summaries, chapter_questions, chapter_answers)
        if not scenario:
            st.info("Not enough data to build a scenario for this chapter.")
            return
        state["scenario"] = {
            "chapter": chapter,
            "show_hint": False,
            "answered": False,
            "data": scenario,
        }
    else:
        scenario = state["scenario"]["data"]

    sc_state = state["scenario"]

    st.markdown(f"### {scenario['title']}")
    st.write(f"**👤 Actor:** {scenario['actor']}")
    st.write(f"**🎯 Goal:** {scenario['goal']}")

    with st.expander("See a possible success path"):
        for step in scenario["success_path"]:
            st.write("•", step)

    with st.expander("Possible failure paths"):
        for f in scenario["failure_paths"]:
            st.write("•", f)

    st.markdown("---")
    st.markdown(f"**Decision Point:** {scenario['question']}")

    choice = st.radio(
        "Choose the best action:",
        scenario["options"],
        key=f"scenario_choice_{chapter}",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Submit decision", key=f"scenario_submit_{chapter}"):
            if choice == scenario["correct"]:
                st.success("✅ Correct decision! +15 XP")
                award_xp("Scenario-Based (with Hint)")
                sc_state["answered"] = True
            else:
                st.error("❌ Not the best option. Think about how the concept is meant to be used.")

    with col2:
        if st.button("Show hint", key=f"scenario_hint_{chapter}"):
            sc_state["show_hint"] = True

    with col3:
        if st.button("🔁 New scenario", key=f"scenario_new_{chapter}"):
            # regenerate a fresh scenario for this chapter
            new_scenario = generate_scenario(chapter, chapter_summaries, chapter_questions, chapter_answers)
            if new_scenario:
                state["scenario"] = {
                    "chapter": chapter,
                    "show_hint": False,
                    "answered": False,
                    "data": new_scenario,
                }
                st.rerun()

    if sc_state["show_hint"]:
        st.info(scenario["hint"])

    with st.expander("📖 Chapter summary reminder"):
        st.write(scenario["summary"])

from collections import defaultdict

def compute_xp_breakdown():
    """
    Aggregate XP per challenge type based on history strings.
    Assumes award_xp stores entries like 'Flashcards +5 XP'.
    """
    state = st.session_state.tutor
    xp_by_challenge = defaultdict(int)

    for entry in state["history"]:
        for label, xp_val in XP_PER_CHALLENGE.items():
            if entry.startswith(label):
                xp_by_challenge[label] += xp_val

    return xp_by_challenge


def progress_dashboard_ui():
    """
    Simple visual dashboard of user's progress:
    - Level / XP summary
    - XP by challenge type (bar chart)
    - Recent activity
    """
    state = st.session_state.tutor

    st.subheader("📊 Your Learning Dashboard")

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Level", state["level"])
    with col2:
        st.metric("Total XP", state["xp"])
    with col3:
        st.metric("Challenges completed", len(state["history"]))

    # XP by challenge type
    xp_breakdown = compute_xp_breakdown()
    if xp_breakdown:
        st.markdown("#### 🧩 XP by challenge type")
        labels = list(xp_breakdown.keys())
        values = [xp_breakdown[l] for l in labels]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("XP")
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title("XP earned per challenge")
        st.pyplot(fig)
    else:
        st.info("Do a few challenges first and I’ll show your XP breakdown here ✨")

    # Recent history
    st.markdown("#### 📜 Recent activity")
    if state["history"]:
        for h in state["history"][-10:][::-1]:
            st.caption("• " + h)
    else:
        st.caption("No activity yet. Try some flashcards or a quiz!")

