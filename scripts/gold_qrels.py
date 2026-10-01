"""Queries and relevance judgements (Modules 5-6).

Procedure: each query has a written information need. Every one of the 30 documents
was judged against that need from its content (reading the passages where the
concept appears, and the document's topic), BEFORE any system's result list was
looked at. Relevance is topical: a document can be relevant without containing the
query words (e.g. a TalkMoves transcript is classroom dialogue even though it never
says "dialogue"), and can contain them without being relevant (a percent sign in a
paper's results is not "about percents"). This is what lets Boolean retrieval score
below 1.0 and makes the comparison between pipelines meaningful.

Judged by Claude (the build assistant) from document content; to be reviewed by the
student before submission. 15 queries x 30 documents = 450 binary judgements.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [f"D{i:02d}" for i in range(1, 31)]

Q = [
    ("Q01", "denominator", "Documents that explain, practise or show learners reasoning explicitly about denominators.",
     {"D01": "defines numerator and denominator", "D02": "denominators in multiplying and dividing fractions",
      "D03": "adding with common denominators", "D04": "adding with different denominators, LCD",
      "D05": "decimals written as fractions with denominators 10, 100", "D06": "converting fractions to decimals divides by the denominator",
      "D07": "ratios written as fractions", "D08": "percent as a fraction with denominator 100",
      "D16": "misconceptions about denominators", "D25": "student explanations about denominators",
      "D26": "student questions about denominators", "D27": "teacher notes on denominators", "D28": "tutor works on denominators",
      "D29": "unit page: adding denominators misconception", "D30": "worksheet item on why 3/4 + 1/2 is not 4/6"}),
    ("Q02", "misconception", "Documents that identify, describe or study students' misconceptions (systematic misunderstandings).",
     {"D09": "papers use misconceptions as a signal / for distractors", "D11": "papers on distractors reflecting misconceptions",
      "D12": "math misconception benchmark paper", "D14": "MathDial: student confusions and misconception profiles",
      "D15": "MaE misconception dataset card", "D16": "MaE misconception catalogue", "D17": "dialogues addressing student confusions",
      "D18": "incorrect solutions with teacher-described confusion", "D25": "explanations each expressing a malrule",
      "D26": "questions voicing misconception-driven confusion", "D27": "teacher notes on observed misconceptions",
      "D28": "learner's add-across misconception and its correction", "D29": "page lists misconceptions to address",
      "D30": "diagnostic items targeting misconceptions"}),
    ("Q03", "common denominator", "Documents that explain or apply finding and using a common denominator.",
     {"D03": "section on common denominators", "D04": "LCD and converting to common denominators",
      "D05": "common denominator used to compare decimals", "D07": "least common denominator of 8/10 and 5/100",
      "D16": "misconception MaE09 about common denominators", "D25": "explanations using a common denominator",
      "D26": "question about needing the same denominator", "D27": "lesson on common denominators",
      "D28": "learner converts halves and thirds to sixths", "D29": "objective: why fractions need a common denominator",
      "D30": "item requiring a common denominator"}),
    ("Q04", "knowledge tracing model", "Documents describing knowledge tracing models (estimating or predicting student knowledge over time).",
     {"D09": "SQKT knowledge tracing model", "D10": "CIKT knowledge tracing framework", "D11": "ORBITS and error-tracing KT models"}),
    ("Q05", "added the numerators", "Documents where adding numerators (with or without adding denominators) is described, as the correct procedure or as the add-across error.",
     {"D03": "add the numerators over the common denominator", "D04": "add the numerators after converting to the LCD",
      "D16": "fraction-addition error examples", "D25": "students adding tops and bottoms", "D26": "question: is 1/2 + 1/2 = 2/4?",
      "D27": "students writing 1/2 + 1/4 = 2/6", "D28": "learner added the top and bottom numbers",
      "D29": "misconception: adding numerators and adding denominators", "D30": "item: explain why not 4/6"}),
    ("Q06", "fraction AND misconception", "Documents about misconceptions specifically concerning fractions.",
     {"D15": "MaE card lists fraction operation misconceptions", "D16": "MaE fraction misconceptions MaE06-MaE22",
      "D25": "fraction malrules", "D26": "fraction confusions in questions", "D27": "fraction misconceptions observed",
      "D28": "add-across fraction misconception", "D29": "fraction misconceptions to address", "D30": "fraction diagnostic items"}),
    ("Q07", 'LCD OR "least common denominator"', "Documents that explain or use the least common denominator.",
     {"D04": "section teaches the LCD", "D07": "least common denominator of 8/10 and 5/100", "D16": "learner determines the LCD",
      "D25": "LCD of 4 and 6", "D26": "question: what does LCD mean", "D27": "LCD for 4 and 6 lesson",
      "D28": "learner rewrites halves and thirds in sixths (the LCD)"}),
    ("Q08", "denominator AND NOT common", "Documents about denominators in contexts OTHER than finding a common denominator (meaning of the denominator, multiplying and dividing, decimals and percents as fractions).",
     {"D01": "meaning of the denominator", "D02": "multiplying and dividing fractions", "D05": "decimals as fractions (one common-denominator aside)",
      "D06": "fractions and decimals", "D07": "ratios as fractions (one LCD aside)", "D08": "percent as a fraction over 100"}),
    ("Q09", "dialogue AND (tutor OR teacher)", "Documents that contain or study tutoring or classroom dialogue between teachers or tutors and students.",
     {"D09": "tutoring dialogue systems papers", "D10": "dialog-based tutoring benchmark papers", "D11": "classroom transcripts and tutoring dialogue shared task",
      "D12": "MathDial dialogue tutoring paper", "D13": "coding manual for classroom talk", "D14": "MathDial dialogue dataset card",
      "D17": "tutoring dialogues", "D19": "classroom transcript", "D20": "classroom transcript", "D21": "classroom transcript",
      "D22": "classroom transcript", "D23": "classroom transcript", "D24": "classroom transcript", "D28": "tutor chat log"}),
    ("Q10", '"knowledge tracing" AND NOT programming', "Knowledge tracing research outside programming education.",
     {"D10": "CIKT and general KT", "D11": "ORBITS KT for video-based learning (the volume also has a programming KT paper)"}),
    ("Q11", "decimal OR percent", "Documents that teach, practise or discuss decimals or percents.",
     {"D05": "decimals section", "D06": "decimals and fractions", "D07": "unit rates with decimals", "D08": "percent section",
      "D16": "decimal and percent misconceptions", "D17": "percent word problems", "D18": "percent word problems",
      "D25": "decimal and percent explanations", "D26": "decimal and percent questions", "D27": "decimals and percents lessons",
      "D28": "decimal multiplication in session 3", "D29": "unit covers decimals and percents", "D30": "decimal and percent items"}),
    ("Q12", '"student explanation"', "Documents that contain students' explanations of their reasoning, or study such explanations.",
     {"D10": "think-aloud and student answer assessment papers", "D11": "papers assessing student explanations",
      "D12": "benchmark of 52,000 written explanations", "D13": "codes students providing evidence and reasoning",
      "D17": "students talk through their solutions", "D18": "written student solutions with reasoning",
      "D19": "students explain in class", "D20": "students explain in class", "D21": "students explain in class",
      "D22": "students explain in class", "D23": "students explain in class", "D24": "students explain in class",
      "D25": "student explanations", "D28": "learner explains why 2/5 is wrong"}),
    ("Q13", "number line", "Documents that use the number line to represent or compare fractions or decimals.",
     {"D01": "locating fractions on the number line", "D05": "locating decimals on the number line", "D06": "ordering on the number line",
      "D24": "lesson on the number line and fractions", "D25": "misconception about placing 3/4 on a number line",
      "D26": "question about closeness to 1 on the number line", "D27": "reteach addition with the number line",
      "D29": "unit models fractions with number lines", "D30": "item: justify with a number line"}),
    ("Q14", "misconception AND student AND fraction", "Documents about students' misconceptions about fractions.",
     {"D15": "MaE card: student fraction misconceptions", "D16": "student fraction misconceptions", "D25": "student fraction malrules",
      "D26": "student fraction confusions", "D27": "students' fraction misconceptions", "D28": "a student's fraction misconception",
      "D29": "students' fraction misconceptions", "D30": "fraction diagnostic items for students"}),
    ("Q15", 'LLM AND "knowledge tracing"', "Documents on using large language models for knowledge tracing.",
     {"D10": "CIKT: LLM-based knowledge tracing"}),
]


def main():
    with open(ROOT / "gold" / "queries.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "query", "information_need", "n_relevant"])
        for qid, q, need, rel in Q:
            w.writerow([qid, q, need, len(rel)])
    n = 0
    with open(ROOT / "gold" / "qrels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "doc_id", "relevant", "reason"])
        for qid, _, _, rel in Q:
            for d in DOCS:
                w.writerow([qid, d, int(d in rel), rel.get(d, "")])
                n += 1
    print(f"{len(Q)} queries, {n} judgements, {sum(len(r) for *_, r in Q)} relevant pairs")


if __name__ == "__main__":
    main()
