"""Hand-tagged POS + lemma gold set (Exercises 10 and 12).

32 domain sentences, tokenized with the project's conventions (fractions and mixed
numbers are one token), tagged with Penn Treebank tags and lemmas by hand.
Written in the style of the corpus (textbook instructions, student explanations,
classroom talk, research abstracts). 20 sentences train the ML taggers, 12 are
held out for testing. The held-out set contains domain terms both seen in
training (LCD, halves) and unseen (tenth, rod, 2 1/2).
"""
import json
from pathlib import Path

# each sentence: "token/TAG/lemma" items separated by " | "
S = {
    "S01": "Simplify/VB/simplify | the/DT/the | fraction/NN/fraction | 6/8/CD/6/8 | before/IN/before | you/PRP/you | add/VBP/add | ./././",
    "S02": "Find/VB/find | the/DT/the | LCD/NN/lcd | of/IN/of | 4/CD/4 | and/CC/and | 6/CD/6 | ./././",
    "S03": "I/PRP/i | added/VBD/add | the/DT/the | top/JJ/top | numbers/NNS/number | and/CC/and | the/DT/the | bottom/JJ/bottom | numbers/NNS/number | ./././",
    "S04": "Flip/VB/flip | the/DT/the | second/JJ/second | fraction/NN/fraction | and/CC/and | multiply/VB/multiply | ./././",
    "S05": "One/CD/one | over/IN/over | two/CD/two | is/VBZ/be | the/DT/the | same/JJ/same | as/IN/as | 1/2/CD/1/2 | ./././",
    "S06": "BKT/NNP/bkt | models/VBZ/model | mastery/NN/mastery | with/IN/with | slip/NN/slip | and/CC/and | guess/NN/guess | parameters/NNS/parameter | ./././",
    "S07": "You/PRP/you | carry/VBP/carry | the/DT/the | one/NN/one | to/TO/to | the/DT/the | tens/NNS/ten | place/NN/place | ./././",
    "S08": "Cross/VB/cross | multiply/VB/multiply | to/TO/to | compare/VB/compare | 3/4/CD/3/4 | and/CC/and | 2/3/CD/2/3 | ./././",
    "S09": "Halves/NNS/half | are/VBP/be | bigger/JJR/big | than/IN/than | thirds/NNS/third | because/IN/because | the/DT/the | pieces/NNS/piece | are/VBP/be | larger/JJR/large | ./././",
    "S10": "The/DT/the | reciprocal/NN/reciprocal | of/IN/of | 5/CD/5 | is/VBZ/be | 1/5/CD/1/5 | ./././",
    "S11": "Write/VB/write | 0.75/CD/0.75 | as/IN/as | a/DT/a | fraction/NN/fraction | in/IN/in | simplest/JJS/simple | form/NN/form | ./././",
    "S12": "Meredith/NNP/meredith | thinks/VBZ/think | 1/8/CD/1/8 | is/VBZ/be | bigger/JJR/big | than/IN/than | 1/6/CD/1/6 | ./././",
    "S13": "SQKT/NNP/sqkt | uses/VBZ/use | student/NN/student | questions/NNS/question | to/TO/to | predict/VB/predict | performance/NN/performance | ./././",
    "S14": "Subtract/VB/subtract | the/DT/the | numerators/NNS/numerator | and/CC/and | keep/VB/keep | the/DT/the | common/JJ/common | denominator/NN/denominator | ./././",
    "S15": "Dividing/VBG/divide | by/IN/by | 1/2/CD/1/2 | doubles/VBZ/double | the/DT/the | number/NN/number | ./././",
    "S16": "The/DT/the | teacher/NN/teacher | asked/VBD/ask | Brian/NNP/brian | to/TO/to | explain/VB/explain | his/PRP$/his | answer/NN/answer | ./././",
    "S17": "Round/VB/round | 3.456/CD/3.456 | to/TO/to | the/DT/the | nearest/JJS/near | tenth/NN/tenth | ./././",
    "S18": "Is/VBZ/be | 3/4/CD/3/4 | greater/JJR/great | than/IN/than | 2/3/CD/2/3 | ?/./?",
    "S19": "The/DT/the | student/NN/student | confused/VBD/confuse | the/DT/the | numerator/NN/numerator | with/IN/with | the/DT/the | denominator/NN/denominator | ./././",
    "S20": "Our/PRP$/our | model/NN/model | outperforms/VBZ/outperform | DKT/NNP/dkt | on/IN/on | ASSIST2009/NNP/assist2009 | ./././",
    "S21": "Shade/VB/shade | three/CD/three | of/IN/of | the/DT/the | four/CD/four | equal/JJ/equal | parts/NNS/part | ./././",
    "S22": "Multiply/VB/multiply | 2/3/CD/2/3 | x/SYM/x | 3/4/CD/3/4 | and/CC/and | simplify/VB/simplify | ./././",
    "S23": "A/DT/a | price/NN/price | that/WDT/that | drops/VBZ/drop | 50%/CD/50% | and/CC/and | rises/VBZ/rise | 50%/CD/50% | ends/VBZ/end | lower/JJR/low | ./././",
    "S24": "Tutors/NNS/tutor | use/VBP/use | scaffolding/NN/scaffolding | questions/NNS/question | instead/RB/instead | of/IN/of | telling/VBG/tell | the/DT/the | answer/NN/answer | ./././",
    "S25": "Erik/NNP/erik | cut/VBD/cut | the/DT/the | rod/NN/rod | into/IN/into | halves/NNS/half | ./././",
    "S26": "We/PRP/we | need/VBP/need | a/DT/a | common/JJ/common | denominator/NN/denominator | because/IN/because | the/DT/the | pieces/NNS/piece | must/MD/must | be/VB/be | the/DT/the | same/JJ/same | size/NN/size | ./././",
    "S27": "The/DT/the | LCD/NN/lcd | of/IN/of | the/DT/the | fractions/NNS/fraction | is/VBZ/be | 12/CD/12 | ./././",
    "S28": "Kavya/NNP/kavya | calculated/VBD/calculate | the/DT/the | second/JJ/second | discount/NN/discount | on/IN/on | the/DT/the | new/JJ/new | price/NN/price | ./././",
    "S29": "Plot/VB/plot | 2 1/2/CD/2 1/2 | on/IN/on | the/DT/the | number/NN/number | line/NN/line | ./././",
    "S30": "Students/NNS/student | often/RB/often | think/VBP/think | that/IN/that | longer/JJR/long | decimals/NNS/decimal | are/VBP/be | larger/JJR/large | ./././",
    "S31": "idk/UH/idk | why/WRB/why | u/PRP/u | add/VBP/add | the/DT/the | bottoms/NNS/bottom | ./././",
    "S32": "Danielle/NNP/danielle | said/VBD/say | two/CD/two | fourths/NNS/fourth | is/VBZ/be | another/DT/another | name/NN/name | for/IN/for | one/CD/one | half/NN/half | ./././",
}
TEST = {"S05", "S07", "S09", "S12", "S17", "S18", "S22", "S25", "S27", "S29", "S31", "S32"}


def parse(item: str):
    # token may contain "/" (fractions), tag never does: split from the right on the tag
    item = item.strip()
    for tag in sorted({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS",
                       "RB", "RBR", "RBS", "DT", "IN", "CC", "CD", "PRP", "PRP$", "TO", "MD", "WDT", "WRB", "UH",
                       "SYM", "."}, key=len, reverse=True):
        key = f"/{tag}/"
        if key in item:
            tok, _, lemma = item.partition(key)
            if tok:
                return tok, tag, (tok if tag == "." else lemma)
    raise ValueError(item)


def main():
    from nltk.tag.mapping import map_tag

    out = Path(__file__).resolve().parents[1] / "gold" / "pos_gold.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for sid, s in S.items():
            items = [parse(x) for x in s.split(" | ")]
            toks, tags, lemmas = zip(*items)
            fh.write(json.dumps({"id": sid, "split": "test" if sid in TEST else "train", "tokens": list(toks),
                                 "penn": list(tags), "upos": [map_tag("en-ptb", "universal", t) for t in tags],
                                 "lemmas": list(lemmas)}, ensure_ascii=False) + "\n")
            n += len(toks)
    print(f"wrote {len(S)} sentences, {n} tokens ({len(TEST)} test sentences) -> {out}")


if __name__ == "__main__":
    main()
