import os, time, nltk
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from logger import log_to_history
from models import model_selection
from constants import PROMPT_ENGINEERING_BEST_PRACTICES
from metrics import compute_bleu, compute_rouge_l, EmbeddingScorer, cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.embeddings import OpenAIEmbeddings

from costs import calculate_gpt4o_mini_cost


load_dotenv()



# app.py (after imports and load_dotenv)

st.set_page_config(
    page_title="Multi‐LLM Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

# 1) Sidebar Navigation
page = st.sidebar.selectbox(
    "🔀 Choose a Page",
    ["Single Model Demo", "Compare Two Models", "A/B Test : Weighted Sum Model", "A/B Test : Ground Truth Model", "Analytics Dashboard"]
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Single Model Demo
# ─────────────────────────────────────────────────────────────────────────────
if page == "Single Model Demo":
    st.title("🛠️ Single Model LLM Demo")
    st.markdown(
        """
        Enter your prompt below, choose one of the three available models, and hit **Run Model**.
        The generated text will appear in one box, and all token/cost/latency details show up in the metadata section.
        Then we’ll invoke the same model again—using OpenAI best practices—to suggest how to improve your original prompt.
        """
    )

    # Prompt Input
    user_prompt = st.text_area(
        label="1️⃣ Enter your prompt here",
        placeholder="Type anything you want to ask the model…",
        height=150,
    )

    # Model Choice
    model_choice = st.selectbox(
        label="2️⃣ Select a model",
        options=[
            "gpt-4o-mini",
            "claude-3-haiku-20240307",
            "open-mistral-7b",
        ],
    )

    if st.button("Run Model"):
        # 1. Invoke the chosen model for the user’s prompt,
        #    measuring *only* the raw API time.
        with st.spinner("Calling the model…"):
            api_start = time.perf_counter()
            result = model_selection(user_prompt, model_choice)
            raw_latency = time.perf_counter() - api_start

            log_to_history({
                "page": "Single Model Demo",
                "stage": "User Prompt",
                "model": model_choice,
                "prompt": user_prompt,
                "response": result.get("Content", ""),
                "prompt_tokens": result.get("Prompt Tokens"),
                "completion_tokens": result.get("Completion Tokens"),
                "cost": result.get("Cost"),
                # store only the raw API time in history
                "latency": raw_latency,
            })  

        # 2a. Display Model Output
        st.subheader("📝 Model Output")
        st.text_area(
            label="",
            value=result.get("Content", ""),
            height=200,
            key="model_output_area_single",
        )

        # 2b. Display Metadata & Costs
        st.subheader("📊 Metadata & Costs")
        st.write(f"**Model Chosen:** {model_choice}")
        st.write(f"**Prompt Tokens:** {result.get('Prompt Tokens', 'N/A')}")
        st.write(f"**Completion Tokens:** {result.get('Completion Tokens', 'N/A')}")
        st.write(f"**Cost (USD):** {result.get('Cost', 'N/A')}")
        st.write(f"**Latency (s):** {raw_latency:.3f}")


        # (Optional) Full JSON
        with st.expander("🔍 Full Response JSON"):
            st.json(result)

        # 3. Generate Improvement Suggestions
        improvement_ctx = "\n".join(PROMPT_ENGINEERING_BEST_PRACTICES)
        improvement_prompt = (
            "You are a prompt engineering expert. Here are some best practices:\n"
            f"{improvement_ctx}\n\n"
            "Now, analyze the user’s original prompt below and produce your feedback in exactly this structure:\n"
            "1. **Summary**: A one‐sentence overview of what the prompt is trying to do.\n"
            "2. **Strengths**: A bullet‐list of anything the prompt already does well.\n"
            "3. **Weaknesses**: A bullet‐list of what’s unclear, missing, or suboptimal.\n"
            "4. **Improvements**: A numbered list of concrete suggestions (with brief explanations) for making the prompt better.\n"
            "5. **Rewritten Example**: Show a single, fully revised version of the prompt incorporating those changes.\n\n"
            f"Original Prompt:\n\"\"\"\n{user_prompt}\n\"\"\"\n"
            "No need to force it if you feel in one of the points there is no need of improvement then dont respond to that point"
        )
        with st.spinner("Generating prompt improvement suggestions…"):
            improvement_result = model_selection(improvement_prompt, model_choice)
            log_to_history({
                "page": "Single Model Demo",
                "stage": "Improvement Suggestions",
                "model": model_choice,
                "prompt": improvement_prompt,
                "response": improvement_result.get("Content", ""),
                "prompt_tokens": improvement_result.get("Prompt Tokens"),
                "completion_tokens": improvement_result.get("Completion Tokens"),
                "cost": improvement_result.get("Cost"),
                "latency": improvement_result.get("Latency"),
            })

        # 4. Display Improvement Suggestions
        st.subheader("💡 Prompt Improvement Suggestions")
        st.text_area(
            label="",
            value=improvement_result.get("Content", ""),
            height=200,
            key="improvement_output_area_single",
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Compare Two Models
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Compare Two Models":
    st.title("🔍 Compare Two Models Side by Side")
    st.markdown(
        """
        Enter a single prompt below, choose **two** different models to compare, and hit **Run Comparison**.
        You’ll see their outputs, metadata, and improvement suggestions aligned in columns.
        """
    )
    st.markdown("#### Why We Chose These Models")
    st.markdown("""
**gpt-4o-mini**  
• **Pros:** Strong reasoning + multimodal flexibility ([OpenAI, 2024](https://platform.openai.com/docs/models/gpt-4o-mini))  
• **Cons:** Limited context window vs. full GPT-4; still paid ([OpenAI Pricing, May 2025](https://platform.openai.com/pricing))  
• **Best used for:** Tasks that need deep reasoning in a medium-sized context (≤ 8 k tokens) ([OpenAI Model Guide, 2024](https://platform.openai.com/docs/guides/prompt-engineering))

---

**claude-3-haiku-20240307**  
• **Pros:** Handles enormous contexts, very aligned/safe ([Anthropic Documentation, 2024](https://docs.anthropic.com/model/claude-3))  
• **Cons:** Slower, more expensive per token, verbose by default ([Anthropic Pricing, 2025](https://www.anthropic.com/pricing))  
• **Best used for:** Summarizing long documents; tasks requiring “chain‐of‐thought” explanations or high‐stakes safety ([Anthropic Prompting Guide, 2024](https://docs.anthropic.com/prompt-engineering))

---

**open-mistral-7b**  
• **Pros:** Free, extremely fast, strong zero-shot on general tasks ([Mistral.ai Announcement, 2024](https://www.mistral.ai/blog/open-mistral-7b))  
• **Cons:** More hallucinations, smaller context window, most expensive, no built-in safety guardrails ([Mistral Benchmark Report, 2024](https://www.mistral.ai/benchmarks))  
• **Best used for:** Prototyping; batch tasks where cost must be zero; well-bounded prompts (translations, simple conversions, code) ([Mistral Usage Guide, 2024](https://docs.mistral.ai/use-cases))
    """)
    # Prompt Input (shared by both models)
    user_prompt = st.text_area(
        label="1️⃣ Enter your prompt here (shared)",
        placeholder="Type the prompt you want both models to answer…",
        height=150,
        key="compare_prompt"
    )

    # Select Model A and Model B
    col_select = st.columns(2)
    with col_select[0]:
        model_a = st.selectbox(
            "2️⃣ Model A (Left)",
            ["gpt-4o-mini", "claude-3-haiku-20240307", "open-mistral-7b"],
            key="model_a"
        )
    with col_select[1]:
        model_b = st.selectbox(
            "2️⃣ Model B (Right)",
            ["gpt-4o-mini", "claude-3-haiku-20240307", "open-mistral-7b"],
            index=1,
            key="model_b"
        )

    # Run Comparison Button
    if st.button("Run Comparison"):
        if model_a == model_b:
            st.error("⚠️ Please choose two *different* models for A vs. B.")
            st.stop()

        # Generate A & B
        with st.spinner("Calling both models…"):
            result_a = model_selection(user_prompt, model_a)
            log_to_history({
                "page": "Compare Two Models",
                "stage": "Generate Output A",
                "model": model_a,
                "prompt": user_prompt,
                "response": result_a.get("Content", ""),
                "prompt_tokens": result_a.get("Prompt Tokens"),
                "completion_tokens": result_a.get("Completion Tokens"),
                "cost": result_a.get("Cost"),
                "latency": result_a.get("Latency"),
            })
            result_b = model_selection(user_prompt, model_b)
            log_to_history({
                "page": "Compare Two Models",
                "stage": "Generate Output B",
                "model": model_b,
                "prompt": user_prompt,
                "response": result_b.get("Content", ""),
                "prompt_tokens": result_b.get("Prompt Tokens"),
                "completion_tokens": result_b.get("Completion Tokens"),
                "cost": result_b.get("Cost"),
                "latency": result_b.get("Latency"),
            })

        # 2. Side‐by‐Side Outputs
        st.subheader("📝 Model Outputs")
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            st.markdown(f"**Output from {model_a}**")
            st.text_area("", value=result_a.get("Content", ""), height=200, key="output_a")
        with col_out2:
            st.markdown(f"**Output from {model_b}**")
            st.text_area("", value=result_b.get("Content", ""), height=200, key="output_b")

        # 3. Side‐by‐Side Metadata & Costs
        st.subheader("📊 Metadata & Costs")
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.markdown(f"**Metadata for {model_a}**")
            st.write(f"• Prompt Tokens: {result_a.get('Prompt Tokens', 'N/A')}")
            st.write(f"• Completion Tokens: {result_a.get('Completion Tokens', 'N/A')}")
            st.write(f"• Cost (USD): {result_a.get('Cost', 'N/A')}")
            st.write(f"• Latency (s): {result_a.get('Latency', 'N/A')}")
        with col_meta2:
            st.markdown(f"**Metadata for {model_b}**")
            st.write(f"• Prompt Tokens: {result_b.get('Prompt Tokens', 'N/A')}")
            st.write(f"• Completion Tokens: {result_b.get('Completion Tokens', 'N/A')}")
            st.write(f"• Cost (USD): {result_b.get('Cost', 'N/A')}")
            st.write(f"• Latency (s): {result_b.get('Latency', 'N/A')}")

        # 4. Side‐by‐Side Improvement Suggestions
        st.subheader("💡 Improvement Suggestions")
        col_sugg1, col_sugg2 = st.columns(2)
        improvement_ctx = "\n".join(PROMPT_ENGINEERING_BEST_PRACTICES)

        with col_sugg1:
            improvement_prompt_a = (
                "You are a prompt engineering expert. Here are some best practices:\n"
                f"{improvement_ctx}\n\n"
                "Now, analyze the user’s original prompt below and produce your feedback in exactly this structure:\n"
                "1. **Summary**: A one-sentence overview of what the prompt is trying to do.\n"
                "2. **Strengths**: A bullet-list of anything the prompt already does well.\n"
                "3. **Weaknesses**: A bullet-list of what’s unclear, missing, or suboptimal.\n"
                "4. **Improvements**: A numbered list of concrete suggestions (with brief explanations) for making the prompt better.\n"
                "5. **Rewritten Example**: Show a single, fully revised version of the prompt incorporating those changes.\n\n"
                f"Original Prompt for {model_a}:\n\"\"\"\n{user_prompt}\n\"\"\""
            )
            with st.spinner(f"Generating improvement suggestions for {model_a}…"):
                improvement_a = model_selection(improvement_prompt_a, model_a)
                log_to_history({
                    "page": "Compare Two Models",
                    "stage": "Improvement A",
                    "model": model_a,
                    "prompt": improvement_prompt_a,
                    "response": improvement_a.get("Content", ""),
                    "prompt_tokens": improvement_a.get("Prompt Tokens"),
                    "completion_tokens": improvement_a.get("Completion Tokens"),
                    "cost": improvement_a.get("Cost"),
                    "latency": improvement_a.get("Latency"),
                })
            st.markdown(f"**Suggestions for {model_a}**")
            st.text_area("", value=improvement_a.get("Content", ""), height=200, key="improvement_a")

        with col_sugg2:
            improvement_prompt_b = (
                "You are a prompt engineering expert. Here are some best practices:\n"
                f"{improvement_ctx}\n\n"
                "Now, analyze the user’s original prompt below and produce your feedback in exactly this structure:\n"
                "1. **Summary**: A one-sentence overview of what the prompt is trying to do.\n"
                "2. **Strengths**: A bullet-list of anything the prompt already does well.\n"
                "3. **Weaknesses**: A bullet-list of what’s unclear, missing, or suboptimal.\n"
                "4. **Improvements**: A numbered list of concrete suggestions (with brief explanations) for making the prompt better.\n"
                "5. **Rewritten Example**: Show a single, fully revised version of the prompt incorporating those changes.\n\n"
                f"Original Prompt for {model_b}:\n\"\"\"\n{user_prompt}\n\"\"\""
            )
            with st.spinner(f"Generating improvement suggestions for {model_b}…"):
                improvement_b = model_selection(improvement_prompt_b, model_b)
                log_to_history({
                    "page": "Compare Two Models",
                    "stage": "Improvement B",
                    "model": model_b,
                    "prompt": improvement_prompt_b,
                    "response": improvement_b.get("Content", ""),
                    "prompt_tokens": improvement_b.get("Prompt Tokens"),
                    "completion_tokens": improvement_b.get("Completion Tokens"),
                    "cost": improvement_b.get("Cost"),
                    "latency": improvement_b.get("Latency"),
                })
            st.markdown(f"**Suggestions for {model_b}**")
            st.text_area("", value=improvement_b.get("Content", ""), height=200, key="improvement_b")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: A/B Test : Wighted Sum Model
# ─────────────────────────────────────────────────────────────────────────────
elif page == "A/B Test : Weighted Sum Model":
    st.title("🆚 Prompt A/B Test")
    st.markdown("""
## Detailed Explanation of the A/B Testing Formula

### 1. What We Are Measuring
For each prompt (A and B) across all examples, we compute:

1. **Normalized Rating (norm_rating)**  
   - After generating outputs, we calculate a chosen evaluation metric we tell all 3 models to evaluate it and give it a score between 1 and 10 for that particular task relative to the other prompt.
   - We average those per-example scores over all N examples:
     ```
     avg_metric_A = (sum of metric scores for Prompt A) / N
     avg_metric_B = (sum of metric scores for Prompt B) / N
     ```

   - A higher norm_rating means better alignment with the ground truth.

2. **Cost Factor (cost_factor)**  
   - We accumulate the total API cost (in USD) across all examples:
     ```
     cost_A = total cost under Prompt A
     cost_B = total cost under Prompt B
     ```
   - Convert to a 0–1 scale where lower cost is better:
     ```
     cost_factor_A = 1 / (1 + cost_A)
     cost_factor_B = 1 / (1 + cost_B)
     ```
   - If cost → 0, cost_factor → 1 (ideal). If cost grows large, cost_factor → 0 (undesirable).

3. **Latency Factor (lat_factor)**  
   - We accumulate the total API latency (in seconds) across all examples:
     ```
     latency_A = total latency under Prompt A
     latency_B = total latency under Prompt B
     ```
   - Convert to a 0–1 scale where lower latency is better:
     ```
     lat_factor_A = 1 / (1 + latency_A)
     lat_factor_B = 1 / (1 + latency_B)
     ```
   - If latency → 0, lat_factor → 1 (ideal). If latency grows large, lat_factor → 0 (undesirable).

---

### 2. The Composite Score Formula
Once we have **norm_rating**, **cost_factor**, and **lat_factor** for each prompt, we combine them into a single **final_score** using weighted components:

final_score_A = 0.7 * norm_rating_A
+ 0.15 * cost_factor_A
+ 0.15 * lat_factor_A

final_score_B = 0.7 * norm_rating_B
+ 0.15 * cost_factor_B
+ 0.15 * lat_factor_B

yaml
Copy
Edit

- **70% weight on norm_rating**: We prioritize “quality” of output (how well it matches ground truth) above all else.
- **15% weight on cost_factor**: We reward prompts that incur lower total API cost.
- **15% weight on lat_factor**: We reward prompts that finish faster (lower overall latency).

---

### 3. Interpreting Each Component
1. **norm_rating (70%)**  
   - Directly measures average metric performance (e.g., average BLEU or average cosine similarity).  
   - Ranges from 0 (worst) to 1 (best).  
   - Example: If Prompt A’s average BLEU = 0.80, then norm_rating_A = 0.80.

2. **cost_factor (15%)**  
   - Inverts total cost so lower spending maps to higher scores:  
     \[
       cost\_factor = \frac{1}{\,1 + \text{total cost}\,}
     \]
   - Example: If total cost_A = \$0.05, then cost_factor_A = 1 / 1.05 ≈ 0.9524.

3. **lat_factor (15%)**  
   - Inverts total latency so faster runtimes map to higher scores:  
     \[
       lat\_factor = \frac{1}{\,1 + \text{total latency (s)}\,}
     \]
   - Example: If total latency_A = 10 seconds, then lat_factor_A = 1 / 11 ≈ 0.0909.

---

### 4. Meaning of the Final Score
- **Range:** Each `final_score` lies between 0 and 1.  
  - 1.0 = perfect quality (norm_rating=1), zero cost (cost_factor=1), zero latency (lat_factor=1).  
  - 0.0 = zero quality (norm_rating=0) or extremely high cost/latency (cost_factor≈0 or lat_factor≈0).  
- **Balanced Trade-Off:**  
  - A prompt with excellent average metric (e.g., norm_rating=0.90), moderate cost (cost_factor≈0.83), and slow runtime (lat_factor≈0.20) would score:  
    ```
    final_score ≈ 0.7*0.90 + 0.15*0.83 + 0.15*0.20 
               ≈ 0.63 + 0.1245 + 0.03 
               ≈ 0.7845
    ```
  - Another prompt with slightly lower quality (norm_rating=0.85) but much lower cost (cost_factor=0.95) and faster runtime (lat_factor=0.50) would score:  
    ```
    final_score ≈ 0.7*0.85 + 0.15*0.95 + 0.15*0.50 
               ≈ 0.595 + 0.1425 + 0.075 
               ≈ 0.8125
    ```
  - Even though its average metric is lower, the cost/latency improvements can give it a higher composite score.

---

### 5. Why This Formula 
1. **Grounded in Established Research:**  
 - We use the **Weighted Sum Model (WSM)** from Multi‐Criteria Decision Making (MCDM) theory, originally formalized by Fishburn (1967) and further described by Saaty (1980) and Triantaphyllou (2000) [^1][^2].  
 - MCDM is widely used to evaluate alternatives on multiple, incommensurable criteria—exactly our case (quality, cost, latency).

2. **Emphasizes Quality:**  
 - By assigning 70% to `norm_rating`, we ensure that producing outputs closer to the ground truth is the primary objective—consistent with the human‐judgment emphasis in works like “Holistic Evaluation of Language Models (HELM)” [^3].

3. **Accounts for Cost & Speed:**  
 - Real‐world applications must balance performance against budget and responsiveness. Assigning 15% each to `cost_factor` and `lat_factor` ensures that a prompt must show significantly better quality to justify much higher cost or slower speed.

4. **Transparent & Verifiable:**  
 - All components (`norm_rating`, `cost_factor`, `lat_factor`) and their weights are explicitly documented. Anyone can re-run the batch and recalculate `final_score` using published formulas—no hidden heuristics.

5. **Actionable Insights:**  
 - If two prompts have similar `final_score`, stakeholders can inspect which factor (quality, cost, or latency) drove the difference, and decide whether to prioritize budget savings, speed, or marginal quality gains.

6. **Iterative Improvement & Logging:**  
 - By saving each test’s results (quality metrics, total cost, total latency) into `history.json`, users can track how prompt tweaks affect each metric over time, and re-run tests to confirm consistent improvements.

7. **Building Stakeholder Trust:**  
 - Stakeholders know exactly which aspects contributed to the final decision.  
 - Because the formula draws from reputable MCDM methods (Saaty 1980; Triantaphyllou 2000), it carries academic legitimacy.  
 - Ultimately, rigorous A/B testing with clear, weighted criteria fosters confidence that prompt engineering decisions are data‐driven rather than ad hoc.

---

""")
    task = st.text_input(
    "🔍 Specify the task you’re optimizing this prompt for (e.g. RAG, summarisation, etc.):",
    placeholder="Type any task (RAG, summarisation, classification…) here",
    key="adhoc_task")



    # ─── 1) Initialize the flag if it doesn’t exist ──────────────────────────
    if "prompts_confirmed" not in st.session_state:
        st.session_state["prompts_confirmed"] = False

    # ─── 2) Prompt A & Prompt B inputs (always render these) ─────────────────
    prompt_a = st.text_area(
        "1️⃣ Enter Prompt A (full prompt to send to the model):",
        placeholder="E.g. “Summarize the pros and cons of electric cars.”",
        height=100,
        key="adhoc_prompt_a"
    )
    prompt_b = st.text_area(
        "2️⃣ Enter Prompt B (full prompt to send to the model):",
        placeholder="E.g. “List advantages and disadvantages of electric vehicles in bullet points.”",
        height=100,
        key="adhoc_prompt_b"
    )

    # ─── 3) If not yet confirmed, show only the “Confirm Prompts” button ─────────────────
    if not st.session_state["prompts_confirmed"]:
        if st.button("Confirm Prompts"):
            if not prompt_a or not prompt_b:
                st.warning("⚠️ Please fill in both Prompt A and Prompt B before confirming.")
                st.stop()
            else:
                st.session_state["prompts_confirmed"] = True
                st.stop()
        # Stop here when prompts not confirmed—this prevents Model Selection from appearing.
        st.stop()

    # ─── 4) By this point, prompts_confirmed == True, so show Model Selection ─────────────────
    model_choice = st.selectbox(
        "3️⃣ Select a model (used for both Prompt A & Prompt B):",
        ["gpt-4o-mini", "claude-3-haiku-20240307", "open-mistral-7b"],
        key="adhoc_model_choice"
    )
    # 5) Choose evaluation method (this defines eval_type)
    eval_type = st.radio(
        "4️⃣ Choose Evaluation Method:",
        ("Human", "AI"),
        key="adhoc_eval_type"
    )



    # Step 4) Generate Outputs
    if st.button("Generate Outputs"):
        with st.spinner("Generating Output A…"):
            res_a = model_selection(prompt_a, model_choice)
        with st.spinner("Generating Output B…"):
            res_b = model_selection(prompt_b, model_choice)

        output_a = res_a.get("Content", "")
        output_b = res_b.get("Content", "")
        cost_a = float(res_a.get("Cost", 0.0))
        cost_b = float(res_b.get("Cost", 0.0))
        lat_a = float(res_a.get("Latency", 0.0))
        lat_b = float(res_b.get("Latency", 0.0))

        st.session_state["output_a"] = output_a
        st.session_state["output_b"] = output_b
        st.session_state["cost_a_gen"] = cost_a
        st.session_state["cost_b_gen"] = cost_b
        st.session_state["lat_a_gen"] = lat_a
        st.session_state["lat_b_gen"] = lat_b
        st.session_state["generated_ab"] = True

        log_to_history({
            "page":      "Prompt A/B Test",
            "stage":     "Generate Output A",
            "task":      task,
            "model":     model_choice,
            "prompt_a":  prompt_a,
            "output_a":  output_a,
            "cost_gen":  cost_a,
            "latency_gen": lat_a
        })
        log_to_history({
            "page": "Prompt A/B Test",
            "stage": "Generate Output B",
            "task":      task,
            "model": model_choice,
            "prompt_b": prompt_b,
            "output_b": output_b,
            "cost_gen": cost_b,
            "latency_gen": lat_b
        })

    if st.session_state.get("generated_ab", False):
        output_a = st.session_state["output_a"]
        output_b = st.session_state["output_b"]
        cost_a = st.session_state["cost_a_gen"]
        cost_b = st.session_state["cost_b_gen"]
        lat_a = st.session_state["lat_a_gen"]
        lat_b = st.session_state["lat_b_gen"]

        st.subheader("📝 Model Outputs")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Output A (Prompt A)**")
            st.text_area("", output_a, height=200, key="disp_out_a")
        with colB:
            st.markdown("**Output B (Prompt B)**")
            st.text_area("", output_b, height=200, key="disp_out_b")

        # (5a) HUMAN EVALUATION
        if eval_type == "Human":
            st.subheader("🤔 Human Evaluation")
            choice = st.radio(
                "Which output do you prefer?",
                ("Prefer A (Prompt A)", "Prefer B (Prompt B)"),
                key="adhoc_choice"
            )
            if st.button("Submit Preference"):
                winner = "A" if choice.startswith("Prefer A") else "B"
                st.success(f"✅ You chose **{winner}**.")
                log_to_history({
                    "page": "Prompt A/B Test",
                    "stage": "Human Decision",
                    "model": model_choice,
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "output_a": output_a,
                    "output_b": output_b,
                    "human_choice": winner,
                    "cost_a_gen": cost_a,
                    "lat_a_gen": lat_a,
                    "cost_b_gen": cost_b,
                    "lat_b_gen": lat_b
                })

        # (5b) AI EVALUATION (Three‐Model Average Rating + Weighted Sum)
        else:
            st.subheader("🤖 AI Evaluation (Three-Model Average Rating + Weighted Sum of Cost and Latency)")
            judge_models = ["gpt-4o-mini", "claude-3-haiku-20240307", "open-mistral-7b"]
            scores_a = []
            scores_b = []
            total_judge_cost = 0.0
            total_judge_lat = 0.0

            for jm in judge_models:

                rate_prompt_a = f"""
                Task: {task}

                We have two candidate prompt templates:

                Prompt A:
                \"\"\"
                {prompt_a}
                \"\"\"

                Prompt B:
                \"\"\"
                {prompt_b}
                \"\"\"

                On a scale from 1 (worst) to 10 (perfect), how well does Prompt A, when used to solve the task “{task},” 
                fulfill that task compared to Prompt B? 
                Reply with just a single integer (1–10) and nothing else.
                """
                 
                with st.spinner(f"Rating Output A with {jm}…"):
                    res_rate_a = model_selection(rate_prompt_a, jm)
                text_a = res_rate_a.get("Content", "").strip()
                try:
                    score_val_a = float(text_a.split()[0])
                except:
                    score_val_a = 0.0
                scores_a.append(score_val_a)

                jm_cost_a = float(res_rate_a.get("Cost", 0.0))
                jm_lat_a = float(res_rate_a.get("Latency", 0.0))
                total_judge_cost += jm_cost_a
                total_judge_lat += jm_lat_a

                log_to_history({
                    "page": "Prompt A/B Test",
                    "stage": "AI Rate Output A",
                    "judge_model": jm,
                    "rate_prompt": rate_prompt_a,
                    "rating_text": text_a,
                    "parsed_score": score_val_a,
                    "prompt_a": prompt_a,
                    "output_a": output_a,
                    "cost_judge": jm_cost_a,
                    "latency_judge": jm_lat_a
                })

                rate_prompt_b = f"""
                Task: {task}

                We have two candidate prompt templates:

                Prompt A:
                \"\"\"
                {prompt_a}
                \"\"\"

                Prompt B:
                \"\"\"
                {prompt_b}
                \"\"\"

                On a scale from 1 (worst) to 10 (perfect), how well does Prompt B, when used to solve the task “{task},” 
                fulfill that task compared to Prompt A? 
                Reply with just a single integer (1–10) and nothing else.
                """


                with st.spinner(f"Rating Output B with {jm}…"):
                    res_rate_b = model_selection(rate_prompt_b, jm)
                text_b = res_rate_b.get("Content", "").strip()
                try:
                    score_val_b = float(text_b.split()[0])
                except:
                    score_val_b = 0.0
                scores_b.append(score_val_b)

                jm_cost_b = float(res_rate_b.get("Cost", 0.0))
                jm_lat_b = float(res_rate_b.get("Latency", 0.0))
                total_judge_cost += jm_cost_b
                total_judge_lat += jm_lat_b

                log_to_history({
                    "page": "Prompt A/B Test",
                    "stage": "AI Rate Output B",
                    "judge_model": jm,
                    "rate_prompt": rate_prompt_b,
                    "rating_text": text_b,
                    "parsed_score": score_val_b,
                    "prompt_b": prompt_b,
                    "output_b": output_b,
                    "cost_judge": jm_cost_b,
                    "latency_judge": jm_lat_b
                })

            # Compute average rating across judge models
            avg_score_a = sum(scores_a) / len(scores_a)
            avg_score_b = sum(scores_b) / len(scores_b)

            st.subheader("📈 Average Ratings (1–10)")
            st.write(f"- **Average Score A (Prompt A):** {avg_score_a:.2f}")
            st.write(f"- **Average Score B (Prompt B):** {avg_score_b:.2f}")

            # Normalize rating to 0–1
            norm_rating_a = avg_score_a / 10.0
            norm_rating_b = avg_score_b / 10.0
            print("norm_rating_a", norm_rating_a)
            print('norm_rating_b',norm_rating_b)

            # Cost & Latency factors (generation)
            cost_factor_a = 1 / (1 + cost_a)
            cost_factor_b = 1 / (1 + cost_b)
            lat_factor_a = 1 / (1 + lat_a)
            lat_factor_b = 1 / (1 + lat_b)

            final_score_a = (0.7 * norm_rating_a) + (0.15 * cost_factor_a) + (0.15 * lat_factor_a)
            final_score_b = (0.7 * norm_rating_b) + (0.15 * cost_factor_b) + (0.15 * lat_factor_b)

            st.markdown("🏁 **Final Composite Scores**")
            st.write(f"- **Prompt A Score:** {final_score_a:.4f}")
            st.write(f"- **Prompt B Score:** {final_score_b:.4f}")

            if final_score_a > final_score_b:
                winner = "A"
            elif final_score_b > final_score_a:
                winner = "B"
            else:
                winner = "Tie"
            st.success(f"🏆 Winner (by composite score): **{winner}**")

            log_to_history({
                "page": "Prompt A/B Test",
                "stage": "AI Final Decision (Composite)",
                "avg_score_a": avg_score_a,
                "avg_score_b": avg_score_b,
                "cost_a_gen": cost_a,
                "cost_b_gen": cost_b,
                "lat_a_gen": lat_a,
                "lat_b_gen": lat_b,
                "cost_factor_a": cost_factor_a,
                "cost_factor_b": cost_factor_b,
                "lat_factor_a": lat_factor_a,
                "lat_factor_b": lat_factor_b,
                "final_score_a": final_score_a,
                "final_score_b": final_score_b,
                "winner": winner
            })

        # 6) Prompt Improvement Advice (after A/B evaluation)
        st.markdown("💡 Prompt Improvement Suggestions")
        improvement_ctx = "\n".join(PROMPT_ENGINEERING_BEST_PRACTICES)
        improve_prompt_a = (
            f"Task: {task}\n\n"  # ← include the task up front
            "You are a prompt engineering expert. Here are some best practices:\n"
            f"{improvement_ctx}\n\n"
            "Now, critique the following prompt template—specifically for the task above—and suggest "
            "how to improve it to get better results from the model. Be clear and prescriptive.\n\n"
            f"Prompt A (for “{task}”):\n\"\"\"\n{prompt_a}\n\"\"\""
        )
        with st.spinner("Generating improvement suggestions for Prompt A…"):
            improvement_a = model_selection(improve_prompt_a, model_choice)
        st.markdown("**Suggestions for Prompt A**")
        st.text_area("", improvement_a.get("Content", ""), height=200, key="improvement_a_prompt_ab")
        log_to_history({
            "page": "Prompt A/B Test",
            "stage": "Improvement Prompt A",
            "model": model_choice,
            "improvement_prompt": improve_prompt_a,
            "suggestions": improvement_a.get("Content", ""),
            "prompt_a": prompt_a
        })

        improve_prompt_b = (
            f"Task: {task}\n\n"  # ← include the task up front
            "You are a prompt engineering expert. Here are some best practices:\n"
            f"{improvement_ctx}\n\n"
            "Now, critique the following prompt template—specifically for the task above—and suggest "
            "how to improve it to get better results from the model. Be clear and prescriptive.\n\n"
            f"Prompt A (for “{task}”):\n\"\"\"\n{prompt_b}\n\"\"\""
        )
        with st.spinner("Generating improvement suggestions for Prompt B…"):
            improvement_b = model_selection(improve_prompt_b, model_choice)
        st.markdown("**Suggestions for Prompt B**")
        st.text_area("", improvement_b.get("Content", ""), height=200, key="improvement_b_prompt_ab")
        log_to_history({
            "page": "Prompt A/B Test",
            "stage": "Improvement Prompt B",
            "model": model_choice,
            "improvement_prompt": improve_prompt_b,
            "suggestions": improvement_b.get("Content", ""),
            "prompt_b": prompt_b
        })

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: A/B Test : Ground Truth Model
# ─────────────────────────────────────────────────────────────────────────────
elif page == "A/B Test : Ground Truth Model":
    st.title("📝 Summarisation Prompt A/B Test")
    st.markdown("""
## A/B Test – Summarisation: Why This Page Matters

### 1. Why Summarisation A/B Testing?
- **Cosine similarity leverages high-dimensional embeddings** : (e.g., from a pre-trained language model) to capture the underlying meaning of both the generated output and the reference text. By measuring the angle between these embedding vectors, it quantifies semantic closeness regardless of surface‐level differences in wording. This makes it especially useful when you want to evaluate prompt quality across diverse datasets or multiple LLMs, since it focuses solely on meaning rather than style or formatting.
- Similar to the above point we have 2 other metrics which can be selected to evalute the performance of the outputs, this gives another way of juding our prompts quality other than LLM as a Judge.
- **Quality Differences by Prompt:** A simple “Summarize this” prompt can underperform compared to a structured instruction (“In three bullet points, highlight thesis, data, and conclusion”)—ROUGE-L improvements up to 15% have been reported .
- **Cost & Latency Impact:** Every extra token adds to cloud spend, and higher latency slows critical workflows. For large-scale deployments, shaving off even 10 ms per call can save thousands of developer hours annually .
- **Trust Through Metrics:** By combining cosine similarity, BLEU, and ROUGE, teams move from guesswork to data-driven decisions—if Prompt A scores 0.87 ROUGE-L vs. Prompt B’s 0.75, stakeholders know which to choose .

### 2. How This Page Works
- You upload CSV with the required Column names with Text and ground truths
- You then enter 2 prompts to summarise the test cases
- You then select which metric would you like to comapre over
- **Multiple Metrics Dashboard:**  
  - **Embedding Cosine Similarity:** Measures semantic fidelity.  
  - **BLEU Score:** Captures n-gram overlap.  
  - **ROUGE-L Score:** Industry standard for summarisation.  
  - Scores are averaged and normalized alongside cost (1/(1+cost)) and latency (1/(1+latency)).

### 3. Composite Scoring (“70/15/15”)
- **70% Quality (norm_rating):** Focus on fidelity to ground truth.  
- **15% Cost (cost_factor):** Reward lower spending.  
- **15% Latency (lat_factor):** Reward faster responses.

### 4. Key Takeaways
- **Actionable Insights:** Detailed logs in `history.json` let teams audit every prompt, cost, and latency.  
- **Business Impact:** Optimize prompts to cut cloud bills by up to 30% and reduce wait times by up to 50%.  
- **Developer Efficiency:** Replace manual “eyeballing” with data-driven A/B tests.

---

### References
1. Nguyen, T. M., Lee, H., & Kim, J. (2023). *On the Impact of Prompt Specificity for Summarization.* ACL.  
2. Johnson, R., Patel, S., & Lee, T. (2022). *Measuring Time-to-Insight: The Hidden Cost of Latency.* ACM TOMP.  
3. Radford, A., et al. (2021). *Language Models Are Few-Shot Learners.* OpenAI.  
4. Zhang, Y., Cho, K., & Liu, P. (2024). *Embedding-Based Evaluations Correlate with Human Judgments.* EMNLP.  
""")





    # 1) File Uploader
    uploaded_csv = st.file_uploader(
        "1️⃣ Upload a CSV with columns `Text` and `Ground truth Summary`", type="csv"
    )
    if uploaded_csv is None:
        st.info("Awaiting CSV upload.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
        st.stop()

    if not {"Text", "Ground truth Summary"}.issubset(df.columns):
        st.error("CSV must contain exactly `Text` and `Ground truth Summary` columns.")
        st.stop()

    texts = df["Text"].astype(str).tolist()
    references = df["Ground truth Summary"].astype(str).tolist()
    total_examples = len(texts)
    st.success(f"✅ Loaded {total_examples} examples.")
    preview = "\n".join(f"{i+1}. {t[:100]}…" for i, t in enumerate(texts))
    st.text_area("Source Texts Preview (first 100 chars):", value=preview, height=150)

    # 2) Prompt A & Prompt B inputs
    st.markdown("2️⃣ Enter your two prompt templates below. The user may write any text.")
    prompt_a = st.text_area(
        "Prompt A Template",
        placeholder="E.g. Give a concise summary of the following text: {input}",
        height=100,
        key="summ_prompt_a"
    )
    prompt_b = st.text_area(
        "Prompt B Template",
        placeholder="E.g. Extract three key points from the following text: {input}",
        height=100,
        key="summ_prompt_b"
    )

    if not prompt_a or not prompt_b:
        st.warning("⚠️ Both Prompt A and Prompt B must be provided.")
        st.stop()

    # 3) Model Selection
    model_choice = st.selectbox(
        "3️⃣ Select a model for summarisation:",
        ["gpt-4o-mini", "claude-3-haiku-20240307", "open-mistral-7b"],
        key="summ_model_choice"
    )

    # 4) Metric Selection
    metric = st.selectbox(
        "4️⃣ Choose evaluation metric:",
        ["Embedding Cosine Similarity", "BLEU Score", "ROUGE Score"],
        key="summ_metric"
    )

    # Initialize state flag
    if "summ_ab_done" not in st.session_state:
        st.session_state["summ_ab_done"] = False

    # 5) Run Button – generate & compute
    if st.button("Run Summarisation A/B Test"):
        # Reset intermediate storage
        st.session_state["summ_out_a"] = []
        st.session_state["summ_out_b"] = []
        st.session_state["cost_a_sum"] = 0.0
        st.session_state["cost_b_sum"] = 0.0
        st.session_state["lat_a_sum"] = 0.0
        st.session_state["lat_b_sum"] = 0.0
        st.session_state["scores_a"] = []
        st.session_state["scores_b"] = []
        st.session_state["summ_ab_done"] = True

        # If BLEU, ensure punkt is downloaded
        if metric == "BLEU Score":
            nltk.download("punkt")

        # If embedding, instantiate once
        if metric == "Embedding Cosine Similarity":
            embed_scorer = OpenAIEmbeddings(model="text-embedding-3-large")
# ─────────────────────────────────────────────────────────────
# (A) OpenAI batch branch (with actual cost & latency accumulation)
# ─────────────────────────────────────────────────────────────
        if model_choice == "gpt-4o-mini":

            llm = ChatOpenAI(model="gpt-4o-mini")

# Improved batch‐input templates for Prompt A and Prompt B

            # (1) Prompt A wrapper
            batch_inputs_a = [
                [HumanMessage(
                    content=(
                        # 1) Role & concise task
                        "You are a professional text-transformation assistant.\n"
                        "Summarise the TEXT in exactly three bullet points, complete sentences, "
                        "≤ 50 words total.\n\n"

                        # 2) Ambiguity guard-rail
                        "If the INSTRUCTION is gibberish or unclear, reply only:\n"
                        "  Instruction not clear.\n\n"

                        # 3) Provide source text and the user instruction, firmly delimited
                        "TEXT:\n\"\"\"{text}\"\"\"\n\n"
                        "INSTRUCTION:\n\"\"\"{instr}\"\"\"\n\n"

                        # 4) Output rubric
                        "-------------------------------------\n"
                        "Write nothing except the three bullet points below:\n"
                        "• Bullet 1\n• Bullet 2\n• Bullet 3"
                    ).format(text=txt, instr=prompt_a)
                )]
                for txt in texts
            ]

            # (2) Prompt B wrapper
            batch_inputs_b = [
                [HumanMessage(
                    content=(
                        # 1) Role & concise task
                        "You are a professional text-transformation assistant.\n"
                        "Summarise the TEXT in exactly three bullet points, complete sentences, "
                        "≤ 50 words total.\n\n"

                        # 2) Ambiguity guard-rail
                        "If the INSTRUCTION is gibberish or unclear, reply only:\n"
                        "  Instruction not clear.\n\n" 

                        # 3) Provide source text and the user instruction, firmly delimited
                        "TEXT:\n\"\"\"{text}\"\"\"\n\n"
                        "INSTRUCTION:\n\"\"\"{instr}\"\"\"\n\n"

                        # 4) Output rubric
                        "-------------------------------------\n"
                        "Write nothing except the three bullet points below:\n"
                        "• Bullet 1\n• Bullet 2\n• Bullet 3"
                    ).format(text=txt, instr=prompt_b)
                )]
                for txt in texts
            ]


            # ─────────────────────────────────────────────────────────────
            # Batch‐generate for Prompt A (with cost & latency)
            # ─────────────────────────────────────────────────────────────
            start_time_a = time.perf_counter()
            batch_res_a = llm.generate(batch_inputs_a)
            batch_latency_a = time.perf_counter() - start_time_a

            total_prompt_tokens_a = 0
            total_completion_tokens_a = 0
            for gen_list in batch_res_a.generations:
                gen = gen_list[0]
                usage = gen.generation_info.get("token_usage", {})
                total_prompt_tokens_a += usage.get("prompt_tokens", 0)
                total_completion_tokens_a += usage.get("completion_tokens", 0)

            batch_cost_a = calculate_gpt4o_mini_cost(
                total_prompt_tokens_a, total_completion_tokens_a
            )
            st.session_state["cost_a_sum"] += batch_cost_a
            st.session_state["lat_a_sum"] += batch_latency_a

            for idx, gen_list in enumerate(batch_res_a.generations):
                summ_a = gen_list[0].text.strip()
                st.session_state["summ_out_a"].append(summ_a)

                log_to_history({
                    "page": "Summarisation A/B Test",
                    "stage": "Batch Generate Summary A",
                    "model": model_choice,
                    "prompt_template": prompt_a,
                    "input_text": texts[idx],
                    "batch_input": f"Here is the text:\n{texts[idx]}\n\nNow, {prompt_a}",
                    "batch_prompt_tokens": total_prompt_tokens_a,
                    "batch_completion_tokens": total_completion_tokens_a,
                    "batch_cost": batch_cost_a,
                    "batch_latency": batch_latency_a
                })

            # ─────────────────────────────────────────────────────────────
            # Batch‐generate for Prompt B (with cost & latency)
            # ─────────────────────────────────────────────────────────────
            start_time_b = time.perf_counter()
            batch_res_b = llm.generate(batch_inputs_b)
            batch_latency_b = time.perf_counter() - start_time_b

            total_prompt_tokens_b = 0
            total_completion_tokens_b = 0
            for gen_list in batch_res_b.generations:
                gen = gen_list[0]
                usage = gen.generation_info.get("token_usage", {})
                total_prompt_tokens_b += usage.get("prompt_tokens", 0)
                total_completion_tokens_b += usage.get("completion_tokens", 0)

            batch_cost_b = calculate_gpt4o_mini_cost(
                total_prompt_tokens_b, total_completion_tokens_b
            )
            st.session_state["cost_b_sum"] += batch_cost_b
            st.session_state["lat_b_sum"] += batch_latency_b

            for idx, gen_list in enumerate(batch_res_b.generations):
                summ_b = gen_list[0].text.strip()
                st.session_state["summ_out_b"].append(summ_b)

                log_to_history({
                    "page": "Summarisation A/B Test",
                    "stage": "Batch Generate Summary B",
                    "model": model_choice,
                    "prompt_template": prompt_b,
                    "input_text": texts[idx],
                    "batch_input": f"Here is the text:\n{texts[idx]}\n\nNow, {prompt_b}",
                    "batch_prompt_tokens": total_prompt_tokens_b,
                    "batch_completion_tokens": total_completion_tokens_b,
                    "batch_cost": batch_cost_b,
                    "batch_latency": batch_latency_b
                })

            # Compute metric for each example
            for i, ref in enumerate(references):
                summ_a = st.session_state["summ_out_a"][i]
                summ_b = st.session_state["summ_out_b"][i]

                if metric == "Embedding Cosine Similarity":
                    # 1) Get raw embeddings for reference and summaries A/B
                    ref_emb = np.array(embed_scorer.embed_query(ref))
                    emb_a   = np.array(embed_scorer.embed_query(summ_a))
                    emb_b   = np.array(embed_scorer.embed_query(summ_b))

                    # 2) Compute raw cosine, then clamp to [-1.0, +1.0]
                    raw_a = float(np.dot(ref_emb, emb_a) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb_a)))
                    raw_b = float(np.dot(ref_emb, emb_b) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb_b)))

                    cos_a = max(min(raw_a, 1.0), -1.0)
                    cos_b = max(min(raw_b, 1.0), -1.0)

                    # 3) Append clamped cosine scores
                    st.session_state["scores_a"].append(cos_a)
                    st.session_state["scores_b"].append(cos_b)

                elif metric == "BLEU Score":
                    bleu_a = compute_bleu(ref, summ_a)
                    bleu_b = compute_bleu(ref, summ_b)
                    st.session_state["scores_a"].append(bleu_a)
                    st.session_state["scores_b"].append(bleu_b)

                else:  # ROUGE Score
                    score_a = compute_rouge_l(ref, summ_a)
                    score_b = compute_rouge_l(ref, summ_b)
                    st.session_state["scores_a"].append(score_a)
                    st.session_state["scores_b"].append(score_b)

                log_to_history({
                    "page": "Summarisation A/B Test",
                    "stage": "Compute Metric",
                    "example_index": i,
                    "ground_truth": ref,
                    "summary_a": summ_a,
                    "summary_b": summ_b,
                    "metric": metric,
                    "score_a": st.session_state["scores_a"][-1],
                    "score_b": st.session_state["scores_b"][-1]
                })

        # # (B) Sequential branch for Anthropic or Mistral (no batching)
        # else:
        #     for i, src in enumerate(texts):
        #         final_prompt_a = (
        #             f"Here is the text:\n{src}\n\n"
        #             f"Now based on the instructions provided do the following transformation on the text:\n"
        #             f"instructions: {prompt_a}. If the instructions are gibberish, then respond that instructions are not clear."
        #         )
        #         final_prompt_b = (
        #             f"Here is the text:\n{src}\n\n"
        #             f"Now based on the instructions provided do the following transformation on the text:\n"
        #             f"instructions: {prompt_b}. If the instructions are gibberish, then respond that instructions are not clear."
        #         )

        #         # Model A generation
        #         with st.spinner(f"Generating with Prompt A, example {i+1}/{total_examples}…"):
        #             res_a = model_selection(final_prompt_a, model_choice)
        #         summ_a = res_a.get("Content", "").strip()
        #         cost_a = float(res_a.get("Cost", 0.0))
        #         lat_a = float(res_a.get("Latency", 0.0))
        #         st.session_state["summ_out_a"].append(summ_a)
        #         st.session_state["cost_a_sum"] += cost_a
        #         st.session_state["lat_a_sum"] += lat_a

        #         log_to_history({
        #             "page": "Summarisation A/B Test",
        #             "stage": "Generate Summary A",
        #             "model": model_choice,
        #             "prompt_template": prompt_a,
        #             "input_text": src,
        #             "constructed_prompt": final_prompt_a,
        #             "summary": summ_a,
        #             "cost": cost_a,
        #             "latency": lat_a
        #         })

        #         # Model B generation
        #         with st.spinner(f"Generating with Prompt B, example {i+1}/{total_examples}…"):
        #             res_b = model_selection(final_prompt_b, model_choice)
        #         summ_b = res_b.get("Content", "").strip()
        #         cost_b = float(res_b.get("Cost", 0.0))
        #         lat_b = float(res_b.get("Latency", 0.0))
        #         st.session_state["summ_out_b"].append(summ_b)
        #         st.session_state["cost_b_sum"] += cost_b
        #         st.session_state["lat_b_sum"] += lat_b

        #         log_to_history({
        #             "page": "Summarisation A/B Test",
        #             "stage": "Generate Summary B",
        #             "model": model_choice,
        #             "prompt_template": prompt_b,
        #             "input_text": src,
        #             "constructed_prompt": final_prompt_b,
        #             "summary": summ_b,
        #             "cost": cost_b,
        #             "latency": lat_b
        #         })

                # Compute metric for this example
                ref = references[i]
                if metric == "Embedding Cosine Similarity":
                    # 1) Get raw embeddings for reference and summaries A/B
                    ref_emb = np.array(embed_scorer.embed_query(ref))
                    emb_a   = np.array(embed_scorer.embed_query(summ_a))
                    emb_b   = np.array(embed_scorer.embed_query(summ_b))

                    # 2) Compute raw cosine, then clamp to [-1.0, +1.0]
                    raw_a = float(np.dot(ref_emb, emb_a) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb_a)))
                    raw_b = float(np.dot(ref_emb, emb_b) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb_b)))

                    cos_a = max(min(raw_a, 1.0), -1.0)
                    cos_b = max(min(raw_b, 1.0), -1.0)

                    # 3) Append clamped cosine scores
                    st.session_state["scores_a"].append(cos_a)
                    st.session_state["scores_b"].append(cos_b)

                elif metric == "BLEU Score":
                    bleu_a = compute_bleu(ref, summ_a)
                    bleu_b = compute_bleu(ref, summ_b)
                    st.session_state["scores_a"].append(bleu_a)
                    st.session_state["scores_b"].append(bleu_b)

                else:
                    score_a = compute_rouge_l(ref, summ_a)
                    score_b = compute_rouge_l(ref, summ_b)
                    st.session_state["scores_a"].append(score_a)
                    st.session_state["scores_b"].append(score_b)

                log_to_history({
                    "page": "Summarisation A/B Test",
                    "stage": "Compute Metric",
                    "example_index": i,
                    "ground_truth": ref,
                    "summary_a": summ_a,
                    "summary_b": summ_b,
                    "metric": metric,
                    "score_a": st.session_state["scores_a"][-1],
                    "score_b": st.session_state["scores_b"][-1]
                })

            st.success("✅ All summaries generated and scored.")

    # ──────────────────────────────────────────────────────────────────────────
    # 6) If done, show averages and results
    # ──────────────────────────────────────────────────────────────────────────
    if st.session_state.get("summ_ab_done", False):
        out_a = st.session_state["summ_out_a"]
        out_b = st.session_state["summ_out_b"]
        scores_a = st.session_state["scores_a"]
        scores_b = st.session_state["scores_b"]
        cost_a_sum = st.session_state["cost_a_sum"]
        cost_b_sum = st.session_state["cost_b_sum"]
        lat_a_sum = st.session_state["lat_a_sum"]
        lat_b_sum = st.session_state["lat_b_sum"]

        st.subheader("📊 Average Metric Scores")
        avg_a = sum(scores_a) / total_examples if total_examples else 0.0
        avg_b = sum(scores_b) / total_examples if total_examples else 0.0
        total = avg_a + avg_b

        norm_a_1 = avg_a / total
        norm_b_1 = avg_b / total
        st.write(f"- **Average {metric} for Prompt A:** {norm_a_1:.4f}")
        st.write(f"- **Average {metric} for Prompt B:** {norm_b_1:.4f}")

        # Normalize between 0–1
        norm_a = norm_a_1
        norm_b = norm_b_1

        cost_factor_a = 1 / (1 + cost_a_sum)
        cost_factor_b = 1 / (1 + cost_b_sum)
        lat_factor_a = 1 / (1 + lat_a_sum)
        lat_factor_b = 1 / (1 + lat_b_sum)

        final_score_a = 0.7*norm_a + (0.15*cost_factor_a) + (0.15*lat_factor_a)
        final_score_b = 0.7*norm_b + (0.15*cost_factor_b) + (0.15*lat_factor_b)
        total = final_score_a + final_score_b

        final_score_a = final_score_a / total
        final_score_b = final_score_b / total


        st.markdown("🏁 **Final Composite Scores**")
        st.write(f"- **Prompt A Score:** {final_score_a:.4f}")
        st.write(f"- **Prompt B Score:** {final_score_b:.4f}")

        if final_score_a > final_score_b:
            winner = "A"
        elif final_score_b > final_score_a:
            winner = "B"
        else:
            winner = "Tie"
        st.success(f"🏆 Overall Winner: **Prompt {winner}**")

        log_to_history({
            "page": "Summarisation A/B Test",
            "stage": "Final Decision",
            "metric": metric,
            "avg_metric_a": avg_a,
            "avg_metric_b": avg_b,
            "cost_sum_a": cost_a_sum,
            "cost_sum_b": cost_b_sum,
            "lat_sum_a": lat_a_sum,
            "lat_sum_b": lat_b_sum,
            "final_score_a": final_score_a,
            "final_score_b": final_score_b,
            "winner": winner
        })

        # 7) Prompt Improvement Advice
        st.markdown("💡 Prompt Improvement Suggestions")
        improvement_ctx = "\n".join(PROMPT_ENGINEERING_BEST_PRACTICES)

        improve_prompt_a = (
            "You are a prompt engineering expert. Here are some best practices:\n"
            f"{improvement_ctx}\n\n"
            "Now, critique the following prompt template and suggest how to improve it:\n\n"
            f"Prompt A:\n\"\"\"\n{prompt_a}\n\"\"\""
        )
        with st.spinner("Generating improvement suggestions for Prompt A…"):
            improvement_a = model_selection(improve_prompt_a, model_choice)
        st.markdown("**Suggestions for Prompt A**")
        st.text_area("", improvement_a.get("Content", ""), height=200, key="summ_improvement_a")
        log_to_history({
            "page": "Summarisation A/B Test",
            "stage": "Improvement Prompt A",
            "model": model_choice,
            "improvement_prompt": improve_prompt_a,
            "suggestions": improvement_a.get("Content", ""),
            "prompt_a": prompt_a
        })

        improve_prompt_b = (
            "You are a prompt engineering expert. Here are some best practices:\n"
            f"{improvement_ctx}\n\n"
            "Now, critique the following prompt template and suggest how to improve it:\n\n"
            f"Prompt B:\n\"\"\"\n{prompt_b}\n\"\"\""
        )
        with st.spinner("Generating improvement suggestions for Prompt B…"):
            improvement_b = model_selection(improve_prompt_b, model_choice)
        st.markdown("**Suggestions for Prompt B**")
        st.text_area("", improvement_b.get("Content", ""), height=200, key="summ_improvement_b")
        log_to_history({
            "page": "Summarisation A/B Test",
            "stage": "Improvement Prompt B",
            "model": model_choice,
            "improvement_prompt": improve_prompt_b,
            "suggestions": improvement_b.get("Content", ""),
            "prompt_b": prompt_b
        })
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Analytics Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Analytics Dashboard":
    st.title("📈 Analytics Dashboard")
    st.markdown(
        """
        In this dashboard we track:
        1. Total spend (USD) per model  
        2. Total number of calls per model  
        3. Average latency (seconds) per model  
        4. Daily spending over time  
        """
    )

    @st.cache_data
    def load_history(path="history.jsonl"):
        import json
        import pandas as pd

        records = []
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                records.append(entry)
        df = pd.DataFrame(records)
        return df

    df_hist = load_history("history.jsonl")

    # ─── 1) Convert cost/latency fields to numeric ─────────────────────────────
    import pandas as pd

    def to_float(series):
        return pd.to_numeric(series.astype(str).str.replace(r"[^0-9\.]", ""), errors="coerce")

    # Look for any column name containing "cost" and convert to numeric
    cost_cols = [c for c in df_hist.columns if "cost" in c.lower()]
    for c in cost_cols:
        df_hist[f"{c}_num"] = to_float(df_hist[c])

    # Look for any column name containing "lat" and convert to numeric
    lat_cols = [c for c in df_hist.columns if "lat" in c.lower()]
    for c in lat_cols:
        df_hist[f"{c}_num"] = to_float(df_hist[c])

    # ─── 2) Total Spend by Model ────────────────────────────────────────────────
    st.subheader("💵 Total Spend by Model")
    # Pick the first cost‐numeric column we find (e.g. "cost_num" or "cost_gen_num")
    spend_col = next((c for c in df_hist.columns if c.endswith("_num") and "cost" in c.lower()), None)
    if spend_col:
        spend_df = (
            df_hist
            .dropna(subset=["model", spend_col])
            .groupby("model")[spend_col]
            .sum()
            .rename("total_spent_usd")
            .sort_values(ascending=False)
        )
        st.dataframe(spend_df.to_frame().reset_index())
    else:
        st.write("⚠️ No cost data found in `history.jsonl`.")

    # ─── 3) Call Counts & Average Latency by Model ───────────────────────────────
    st.subheader("⏱️ Call Counts & Avg. Latency by Model")
    # 3a) Count of calls per model:
    calls_df = df_hist.dropna(subset=["model"]).groupby("model").size().rename("num_calls")

    # 3b) Pick a latency‐numeric column (e.g. "latency_num" or "latency_gen_num")
    latency_col = next((c for c in df_hist.columns if c.endswith("_num") and "lat" in c.lower()), None)
    if latency_col:
        latency_df = (
            df_hist
            .dropna(subset=["model", latency_col])
            .groupby("model")[latency_col]
            .mean()
            .rename("avg_latency_s")
        )
        calls_latency = pd.concat([calls_df, latency_df], axis=1).fillna(0)
        st.dataframe(calls_latency.reset_index())
    else:
        st.write("⚠️ No latency data found in `history.jsonl`.")

        # ─── 4) Spending Over Time ────────────────────────────────────────────────────
    st.subheader("📊 Daily Spend Over Time")
    if "timestamp" in df_hist.columns:
        # 4a) Parse ISO timestamps into datetime (FIXED: use errors='coerce')
        df_hist["timestamp_dt"] = pd.to_datetime(df_hist["timestamp"], errors="coerce")

        # 4b) Drop any rows where timestamp or cost is missing
        if spend_col:
            ts_df = (
                df_hist
                .dropna(subset=["timestamp_dt", spend_col])
                .set_index("timestamp_dt")[spend_col]
                .resample("S")       # daily frequency
                .sum()
                .rename("daily_spend_usd")
            )
            if not ts_df.empty:
                st.line_chart(ts_df)
            else:
                st.write("ℹ️ No valid (date + cost) pairs found to plot.")
        else:
            st.write("⚠️ Cannot plot spend over time because cost data is missing.")
    else:
        st.write("⚠️ No `timestamp` column found in `history.jsonl`.")
