"""Hand-annotated NER gold set (Exercise 13), inline markup [text](TYPE).

61 sentences in the style and vocabulary of the corpus (many are verbatim corpus
sentences: ACL author lines, MathDial problems, TalkMoves turns, OpenStax text, the
teacher notes and worksheet). 7 standard types + 5 domain types.

Guidelines: PERSON excludes titles (Ms., Mr.). DATE follows OntoNotes (includes
durations such as "five months"). Domain types follow config/domain_lexicon.csv;
every lexicon concept mention is annotated, including plain "fraction".
"""
import json
import re
from pathlib import Path

G = [
    "Ms. [Priya Raman](PERSON) is Head of Mathematics at [Greenfield Public School](ORG) in [Bengaluru](GPE), [Karnataka](GPE).",
    "The observation period ran from [3 August 2026](DATE) to [28 August 2026](DATE).",
    "[Rohan](PERSON) and [Meera](PERSON) wrote 2/3 = 4/5 by adding the same number to the [numerator](CONCEPT) and the [denominator](CONCEPT).",
    "Order 40 sets of [fraction](CONCEPT) tiles for the class, with a budget of [Rs 2,500](MONEY) approved by Mr. [Suresh Kumar](PERSON).",
    "Selected students will represent the school at the [Inter-School Math Olympiad](EVENT) on [18 October 2026](DATE) in [Mysuru](GPE).",
    "Share progress with parents on [Google Classroom](PRODUCT) by [4 September 2026](DATE).",
    "The [Parent Maths Workshop](EVENT) is on [25 September 2026](DATE) and the fee is [Rs 200](MONEY) per family.",
    "The [Fraction Fair 2026](EVENT) exhibition opens on [6 November 2026](DATE).",
    "Learner: [Aarav](PERSON). Tutor: Ms. [Neha Kapoor](PERSON). Platform: school tutoring pilot, [Bengaluru](GPE).",
    "On [2026-09-01](DATE) [Aarav](PERSON) said 1/2 + 1/3 = 2/5, a clear case of [adding numerators and denominators](MISCONCEPTION).",
    "A cricket bat costs [Rs 1,200](MONEY) during the sale.",
    "A [Samsung](ORG) tablet costs [Rs 18,000](MONEY) in [Mumbai](GPE) and [Rs 17,100](MONEY) in [Pune](GPE).",
    "[Kabir](PERSON) scored 42 out of 50 in the practice test on [5 September 2026](DATE).",
    "[Knowledge Tracing](CONCEPT) in Programming Education Integrating Students' Questions. Authors: [Doyoun Kim](PERSON) ([Seoul National University](ORG)); [Suin Kim](PERSON) ([Elice](ORG)); [Yohan Jo](PERSON) ([Seoul National University](ORG))",
    "This paper introduces [SQKT](KT_MODEL), a [knowledge tracing](CONCEPT) model that leverages students' questions and automatically extracted skill information.",
    "In in-domain experiments, [SQKT](KT_MODEL) achieved a 33.1% absolute improvement in [AUC](METRIC) compared to baseline models.",
    "[CIKT](KT_MODEL): A Collaborative and Iterative [Knowledge Tracing](CONCEPT) Framework with Large Language Models",
    "Authors: [Runze Li](PERSON); [Siyu Wu](PERSON); [Jun Wang](PERSON); [Wei Zhang](PERSON) ([East China Normal University](ORG))",
    "Authors: [Jialin Ouyang](PERSON) ([Columbia University](ORG))",
    "Authors: [Nicy Scaria](PERSON) ([Indian Institute of Science](ORG)); [Deepak Subramani](PERSON) ([Indian Institute of Science](ORG))",
    "Authors: [Yann Hicke](PERSON) ([Cornell University](ORG)); [Abhishek Masand](PERSON) ([Cornell University](ORG))",
    "We benchmark 18 assistants, including the latest LLMs such as [GPT-5](PRODUCT), [Claude 4.1 Opus](PRODUCT), and [Gemini 2.5 Pro](PRODUCT).",
    "Through a disentangled evaluation on [GSM8K](DATASET) and [SVAMP](DATASET), we find that the final-answer [accuracy](METRIC) of [Llama-3](PRODUCT) and [Qwen2.5](PRODUCT) is bottlenecked by arithmetic.",
    "[MathDial](DATASET): A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems",
    "Authors: [Jakub Macina](PERSON); [Nico Daheim](PERSON); [Sankalan Pal Chowdhury](PERSON)",
    "The [TalkMoves](DATASET) Dataset: K-12 Mathematics Lesson Transcripts Annotated for Teacher and Student Discursive Moves",
    "The [NCTE Transcripts](DATASET): A Dataset of Elementary Math Classroom Transcripts. Authors: [Dorottya Demszky](PERSON) ([Stanford University](ORG)); [Heather Hill](PERSON) ([Harvard](ORG))",
    "The [MaE](DATASET) dataset is a collection of 220 diagnostic examples that represent 55 common algebra misconceptions among middle school students.",
    "Dataset tested with [GPT-4](PRODUCT), achieving 83.9% [accuracy](METRIC) when constrained by topic.",
    "This dataset supports the paper by [Nancy Otero](PERSON), [Stefania Druga](PERSON), and [Andrew Lan](PERSON).",
    "Misconception MaE01 (Number sense): when students don't understand how to represent proportional relationships. Source: [Ashlock](PERSON), [2006](DATE).",
    "Source: [Bush](PERSON), [2011](DATE)",
    "[Karan](PERSON) borrowed [$3,650](MONEY) for [five months](DATE) at an interest [rate](CONCEPT) of 10%.",
    "He keeps 1/3 of them and gets to go to the amusement park with [$50](MONEY) in spending cash.",
    "In total, [Mandy](PERSON) paid [$205](MONEY) for data in [the first 6 months](DATE).",
    "So the total contribution before [Harry](PERSON) added his [$30](MONEY) was 3x.",
    "Use your answer to estimate the sales tax [Felipa](PERSON) would pay on a [$95](MONEY) dress.",
    "For our [$5.03](MONEY) lunch, we can write the [decimal](CONCEPT) 5.03 as a [mixed number](CONCEPT).",
    "We know that [$1](MONEY) is the same as [$1.00](MONEY).",
    "Teacher: [Danielle](PERSON) why don't you come up and show them what you're thinking?",
    "[Brian](PERSON): I just found out another way that a half can be...",
    "[Graham](PERSON) and [Kelly](PERSON) had something very interesting to say about why one half is another name for two fourths.",
    "Find the [least common denominator](CONCEPT) ([LCD](CONCEPT)) of 4 and 6, then write each [equivalent fraction](CONCEPT).",
    "Convert the [mixed number](CONCEPT) to an [improper fraction](CONCEPT): 11 1/3.",
    "The [reciprocal](CONCEPT) of 5 is 1/5, so dividing by 5 is the same as multiplying by 1/5.",
    "Locate 3/4 on the [number line](CONCEPT) and compare it with 2/3.",
    "The [fraction](CONCEPT) with the [larger denominator is always the larger fraction](MISCONCEPTION), so 3/10 > 3/5.",
    "[BKT](KT_MODEL) models mastery with slip and guess parameters, while [DKT](KT_MODEL) uses a recurrent neural network.",
    "[Bayesian Knowledge Tracing](KT_MODEL) was introduced by [Corbett](PERSON) and [Anderson](PERSON) in [1994](DATE).",
    "[Deep Knowledge Tracing](KT_MODEL) was evaluated on [ASSISTments](DATASET) and reported a higher [AUC](METRIC) than [BKT](KT_MODEL).",
    "Models like [Q-MCKT](KT_MODEL) are evaluated on [ASSIST2009](DATASET) and [EdNet](DATASET).",
    "[FoundationalASSIST](DATASET) contains 1.7 million interactions released by [ASSISTments](ORG).",
    "We report [F1](METRIC), [ECE](METRIC) and [RMSE](METRIC) on the held-out split.",
    "Inter-annotator agreement was [Cohen's kappa](METRIC) = 0.85 on [ASSIST2009](DATASET).",
    "The [TalkMoves](DATASET) application gives teachers feedback on [talk moves](CONCEPT) such as revoicing.",
    "[ETH Zurich](ORG) released [MathDial](DATASET) under a CC BY-SA licence.",
    "[OpenStax](ORG) publishes Prealgebra 2e from [Rice University](ORG) in [Houston](GPE).",
    "[Eedi](ORG) ran the [Mining Misconceptions in Mathematics](EVENT) competition on [Kaggle](ORG) in [2024](DATE).",
    "Students often think a [longer decimal is bigger](MISCONCEPTION), so they say 0.125 is greater than 0.5.",
    "When students [add across](MISCONCEPTION), they write 1/4 + 1/2 = 2/6 instead of using a [common denominator](CONCEPT).",
    "The paper was presented at [ACL 2025](EVENT) in [Vienna](GPE), [Austria](GPE), in [July 2025](DATE).",
]

MARK = re.compile(r"\[([^\]]+)\]\(([A-Z_]+)\)")


def parse(s: str):
    text, ents, pos, out = "", [], 0, []
    for m in MARK.finditer(s):
        out.append(s[pos:m.start()])
        start = sum(len(x) for x in out)
        out.append(m.group(1))
        ents.append([start, start + len(m.group(1)), m.group(2)])
        pos = m.end()
    out.append(s[pos:])
    text = "".join(out)
    for a, b, lab in ents:
        assert text[a:b] and "[" not in text[a:b], (s, a, b)
    return text, ents


def main():
    out = Path(__file__).resolve().parents[1] / "gold" / "ner_gold.jsonl"
    n = 0
    types = {}
    with open(out, "w", encoding="utf-8") as fh:
        for i, s in enumerate(G, 1):
            text, ents = parse(s)
            n += len(ents)
            for e in ents:
                types[e[2]] = types.get(e[2], 0) + 1
            fh.write(json.dumps({"id": f"N{i:02d}", "text": text, "entities": ents}, ensure_ascii=False) + "\n")
    print(f"wrote {len(G)} sentences, {n} entities", dict(sorted(types.items())))


if __name__ == "__main__":
    main()
