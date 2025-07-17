import sys
import os
import json
import datetime
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator, EvaluatorType
from lightrag import QueryParam
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from query import initialize_rag
import asyncio
from openai import OpenAI



GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")
client = OpenAI()


async def run_evaluation():
    with open("eval/eval_set.json", "r") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} evaluations from eval_set.json")

    os.makedirs("eval/results", exist_ok=True)
    os.makedirs("eval/summary", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"eval/results/run_{timestamp}.json"
    summary_path = f"eval/summary/summary_{timestamp}.json"

    grader_correct = 0
    gpt_correct = 0
    total = 0
    results = []

    rag = await initialize_rag()
    prompt = """Follow these additional instructions. In case of conflict with previous instructions, the following instructions take precedence:
            (1) EXCLUDE references
            (2) Don't use MD formatting, provide plain string output.
            (3) Keep the answers to the point and relevant to the question and avoid unnecessary verbosity.
            """

    for i, entry in enumerate(eval_set, 1):
        question = entry.get("question")
        reference = entry.get("reference_answer")

        if not question or not reference:
            print(f"Skipping invalid entry {i}: missing question or reference_answer")
            continue


        print(f"Evaluation {i}: {question}")

        result = await rag.aquery(question, param=QueryParam(mode="hybrid", user_prompt=prompt ))
        generated_answer = result["output_text"] if isinstance(result, dict) else result


        grader = load_evaluator(
            EvaluatorType.LABELED_CRITERIA,
            llm=ChatOpenAI(model=GPT_MODEL, temperature=0.3),
            criteria = {
                "correctness": "Does the answer reasonably address the main point of the question and identifies if the information is not available?",
                "relevance":   "Are the details drawn mainly from the provided context and kept on topic?",
                "abstraction": "Does the answer synthesize across sources without leaning on long quotes?"
            }
        )
        graded = grader.evaluate_strings(
            input=question,
            prediction=generated_answer,
            reference=reference
        )

        score = graded.get("score", 0)
        reasoning = graded.get("reasoning", "N/A")

        GRADER_PROMPT= f"""
        You are an expert answer evaluator.

        QUESTION:
        {question}

        REFERENCE ANSWER:
        {reference}

        CANDIDATE ANSWER:
        {result}

        ------------------------------------------------
        TASK
        Evaluate the Candidate Answer against the Reference Answer on the three criteria below.
        For each criterion, assign an integer score from **1 (poor)** to **7 (excellent)**.
        If the information is not available and the answer hallucinates, the correctness must be 1.
        if the information is not available and answer correctly identifies, the correctness must be 7
        Criteria & Weights
        • Correctness  – Does the answer reasonably address the main point?              (50 %)
        • Relevance    – Are details drawn mainly from the provided context and on topic? (30 %)
        • Abstraction  – Does the answer synthesize across sources without long quotes?   (20 %)

        Weighted-average formula (THIS FORMULA MUST BE FOLLOWED)
        overall = 0.50 × Correctness
                + 0.30 × Relevance
                + 0.20 × Abstraction

        
        If overall > 5.25 → output flag **1** (pass)  
        Else               → output flag **0** (fail)

        ------------------------------------------------
        OUTPUT FORMAT (exactly):

        <flag>
        reasoning
        "Correctness": <1-7>, "Relevance":   <1-7>, "Abstraction": <1-7>

        Example:

        1
        correctness has issue x, but y and z were good, relevance was good, abstraction was good because....
        "Correctness": 6, "Relevance":   6, "Abstraction": 7
   

        0
        correctness has issue x, but y and z were good, relevance was good, abstraction was good because....
        "Correctness": 3, "Relevance":   2, "Abstraction": 5

        ------------------------------------------------
        Begin your evaluation now.
        """
        response = client.chat.completions.create(
            model=GPT_MODEL,
            temperature=.3,
            messages=[{"role": "user", "content": GRADER_PROMPT}],
        )
        gpt_result = response.choices[0].message.content.strip()
        raw_out = gpt_result.strip().split('\n')
        gpt_score = int(raw_out[0])
        gpt_metrics = raw_out[-1]
        gpt_reasoning = "\n".join(raw_out[1:-1])
        print(gpt_metrics)
        print(f"✅ GraderScore: {score}, GPTScore: {gpt_score}")
        print("=" * 100)

        results.append({
            "question": question,
            "grader_score": score,
            "gpt_score": gpt_score,
            "gpt_metrics": gpt_metrics,
            "grader_reasoning": reasoning,
            "gpt_reasoning": gpt_reasoning,
            "generated_answer": generated_answer,
            "reference_answer": reference,
        })

        total += 1
        if score == 1:
            grader_correct += 1

        if gpt_score == 1:
            gpt_correct += 1

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "total": total,
        "grader_correct": grader_correct,
        "grader_accuracy": round(grader_correct / total, 2) if total else 0,
        "gpt_correct": gpt_correct,
        "gpt_accuracy": round(gpt_correct / total, 2) if total else 0,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Eval complete\nGrader Results: {grader_correct}/{total} correct ({summary['grader_accuracy']*100:.1f}%) from Grader")
    print(f"\nGPT Results: {gpt_correct}/{total} correct ({summary['gpt_accuracy']*100:.1f}%) from GPT")
    print(f"📂 Results saved to: {result_path}")
    print(f"📊 Summary saved to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())