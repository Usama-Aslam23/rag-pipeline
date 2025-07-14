import sys
import os
import json
import datetime
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator, EvaluatorType
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from query import query_pipeline


load_dotenv()

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")


def run_evaluation():
    with open("eval/eval_set.json", "r") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} evaluations from eval_set.json")

    os.makedirs("eval/results", exist_ok=True)
    os.makedirs("eval/summary", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"eval/results/run_{timestamp}.json"
    summary_path = f"eval/summary/summary_{timestamp}.json"

    correct = 0
    total = 0
    results = []

    for i, entry in enumerate(eval_set, 1):
        question = entry.get("question")
        reference = entry.get("reference_answer")
        company = entry.get("company")

        if not question or not reference:
            print(f"Skipping invalid entry {i}: missing question or reference_answer")
            continue


        print(f"Evaluation {i}: {question}")

        result = query_pipeline(question, company=company)
        generated_answer = result["output_text"] if isinstance(result, dict) else result


        grader = load_evaluator(
            EvaluatorType.LABELED_CRITERIA,
            llm=ChatOpenAI(model=GPT_MODEL, temperature=0.1),
            criteria={
                "correctness": "Does the answer correctly address the input question?",
                "relevance": "Is the answer focused on relevant facts from the context?",
                "coverage_of_sources": "Does the answer reflect insights from all required documents?",
                "abstraction": "Does the answer generalize across sources instead of quoting them?"
            }
        )
        graded = grader.evaluate_strings(
            input=question,
            prediction=generated_answer,
            reference=reference
        )

        score = graded.get("score", 0)
        reasoning = graded.get("reasoning", "N/A")
        

        print(f"✅ Score: {score}")
        print(f"📋 Reasoning: {reasoning}")
        print("=" * 100)

        results.append({
            "question": question,
            "company": company,
            "score": score,
            "reasoning": reasoning,
            "generated_answer": generated_answer,
            "reference_answer": reference,
        })

        total += 1
        if score == 1:
            correct += 1

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 2) if total else 0
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Eval complete: {correct}/{total} correct ({summary['accuracy']*100:.1f}%)")
    print(f"📂 Results saved to: {result_path}")
    print(f"📊 Summary saved to: {summary_path}")

if __name__ == "__main__":
    run_evaluation()