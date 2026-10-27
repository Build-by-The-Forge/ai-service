"""Generate the distractor documents of the retrieval evaluation set (P3-5).

    python scripts/generate_retrieval_distractors.py \
        --out data/retrieval_eval/v1/distractors.json

Real course material is dominated by past-question papers: the same subjects
and vocabulary as the lecture notes, but rarely the explanation a student
needs. These are the hard distractors a retriever must rank below the
right passage, so the evaluation set includes original past-question papers
generated here from templates over the corpus's eight subjects. The output
is deterministic for a seed and committed; the evaluation hashes it with the
rest of the set.

No template reproduces an evidence span from questions.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20261027
PAPERS_PER_COURSE = 8
PAGES_PER_PAPER = 3
QUESTIONS_PER_PAGE = 7

COURSES: dict[str, list[str]] = {
    "MTS 102": [
        "Evaluate lim(x→{a}) ({b}x² + {c}x)/(x + {d}).",
        "Find the derivative of y = {b}x³ − {c}x² + {d}x.",
        "If f(x) = (x² + {a})^{b}, find f′(x) using the chain rule.",
        "Find the gradient of the curve y = {b}x² − {c} at the point where x = {a}.",
        "Determine whether f(x) = (x² − {a}²)/(x − {a}) can be made continuous at x = {a}.",
        "Find dy/dx if {b}xy + y² = {c}.",
        "Find the stationary points of y = x³ − {c}x and state their nature.",
        "Evaluate lim(x→∞) ({b}x³ + {a})/({c}x³ − {d}).",
        "A particle moves so that s = {b}t² + {c}t. Find its velocity when t = {a}.",
        "Differentiate y = sin({b}x) cos({c}x) with respect to x.",
    ],
    "PHY 102": [
        "Two charges of {a} μC and {b} μC are {c} cm apart. Calculate the force between them.",
        "A charge of {a} nC is placed in a field of {b} N/C. Find the force on it.",
        "Calculate the potential {c} cm from a point charge of {a} μC.",
        "A rod rubbed with a cloth acquires a charge of {a} nC. How many electrons were transferred?",
        "Three resistors of {a} Ω, {b} Ω and {c} Ω are connected in series. Find the total resistance.",
        "A {a} Ω and a {b} Ω resistor are connected in parallel across {c} V. Find the current drawn.",
        "A heater draws {a} A from a {b} V supply. Calculate its power.",
        "Find the resistance of a wire {a} m long with cross-sectional area {b} mm² and resistivity {c} × 10⁻⁸ Ω m.",
        "Sketch the field lines between two charges of {a} μC and −{a} μC.",
        "A {a} V battery drives {b} C of charge round a circuit. How much energy is transferred?",
    ],
    "CHM 101": [
        "Calculate the heat absorbed when {a} g of water is warmed by {b} K.",
        "Given ΔHf° values of −{a} kJ/mol and −{b} kJ/mol, calculate ΔH for the reaction.",
        "When {a} g of a salt dissolves, the temperature falls by {b} °C. Is the process exothermic or endothermic?",
        "Use bond enthalpies of {a} kJ/mol (C–H) and {b} kJ/mol (O=O) to estimate ΔH of combustion.",
        "Construct an energy cycle to find ΔH for the formation of a compound from ΔHc values of −{a} and −{b} kJ/mol.",
        "{a} cm³ of acid neutralises {b} cm³ of alkali with a temperature rise of {c} K. Find the enthalpy of neutralisation.",
        "State the standard conditions used for enthalpy changes and explain why they are needed.",
    ],
    "BIO 101": [
        "Draw and label a plant cell as seen under the light microscope (Question {a}).",
        "Give {a} differences between prokaryotic and eukaryotic cells.",
        "A potato strip in {a}% sucrose solution lost {b}% of its mass. Explain the result.",
        "List {a} organelles found in animal cells and give one function of each.",
        "Explain why root hair cells have many mitochondria (Question {b}).",
        "Describe an experiment to demonstrate diffusion using a {a}% solution of potassium permanganate.",
        "Calculate the actual size of a cell that measures {a} mm in a drawing magnified {b} times.",
    ],
    "CSC 101": [
        "What is printed by: x = [{a}, {b}, {c}]; x.append({d}); print(len(x))?",
        "Write a loop that prints the numbers from {a} to {b}.",
        'What is the value of s[{a}:{b}] if s = "programming"?',
        "Write a function that returns the largest of {a} numbers in a list.",
        'What does the expression {{"a": {a}, "b": {b}}}["b"] evaluate to?',
        "Explain the output of: t = ({a}, {b}); t[0] = {c}.",
        "Write a list comprehension that keeps the even numbers from range({a}).",
    ],
    "STA 111": [
        "Find the mean of {a}, {b}, {c}, {d} and {a}.",
        "Find the median of the scores {b}, {a}, {d}, {c}, {b}, {a}.",
        "State the mode of the data {a}, {b}, {b}, {c}, {d}, {b}.",
        "Calculate the range of {a}, {c}, {b}, {d}.",
        "The marks of {a} students have mean {b} and standard deviation {c}. Interpret these values.",
        "Classify each variable as qualitative or quantitative: age, faculty, height (Question {a}).",
        "Draw a box plot for data with Q1 = {a}, median = {b} and Q3 = {c}.",
    ],
}
SESSIONS = ["2018/2019", "2019/2020", "2020/2021", "2021/2022", "2022/2023"]


def question(rng: random.Random, template: str) -> str:
    values = {k: rng.randint(2, 9) for k in "abcd"}
    return template.format(**values)


def paper(rng: random.Random, course: str, session: str, number: int) -> dict:
    templates = COURSES[course]
    pages = []
    asked = 0
    for _ in range(PAGES_PER_PAPER):
        lines = []
        for _ in range(QUESTIONS_PER_PAGE):
            asked += 1
            lines.append(f"{asked}. {question(rng, rng.choice(templates))}")
            a, b, c = (rng.randint(1, 99) for _ in range(3))
            lines.append(f"A. {a}  B. {b}  C. {c}  D. none of the above")
        pages.append("\n".join(lines))
    pages[0] = f"{course} Past Questions, {session} session, Paper {number}\n\n" + pages[0]
    slug = course.lower().replace(" ", "")
    return {
        "id": f"pq-{slug}-{session.replace('/', '-')}-{number}",
        "title": f"{course} Past Questions {session} (Paper {number})",
        "pages": pages,
    }


def generate(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    documents = [
        paper(rng, course, SESSIONS[n % len(SESSIONS)], n + 1)
        for course in COURSES
        for n in range(PAPERS_PER_COURSE)
    ]
    return {
        "version": "v1",
        "description": "Generated past-question papers (distractors) for the retrieval evaluation; see scripts/generate_retrieval_distractors.py.",
        "seed": seed,
        "documents": documents,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    text = json.dumps(generate(args.seed), indent=1, ensure_ascii=False) + "\n"
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
