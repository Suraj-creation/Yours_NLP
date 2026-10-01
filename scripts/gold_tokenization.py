"""Hand-segmented tokenization gold set (Exercise 3). Written from the guidelines in
gold/annotation_guidelines.md, not from any tokenizer's output.

Conventions (short form): a fraction, a no-space expression, a mixed number, a
decimal, a percent, a currency amount with its symbol, a ratio/time, an ISO date,
a hyphenated or apostrophe word, an abbreviation with periods and an
alphanumeric name are ONE token. Other punctuation is one token per mark
("..." is one). Operators surrounded by spaces are their own tokens.
"""
import json
from pathlib import Path

G = [
    # (problem_type, source doc, text, tokens)
    ("fraction", "D18", "To maintain 2 students per computer, they need 98/2 = 49 computers in total.",
     "To|maintain|2|students|per|computer|,|they|need|98/2|=|49|computers|in|total|."),
    ("fraction", "D16", "Explanation: The learner estimated 5/6 = 1",
     "Explanation|:|The|learner|estimated|5/6|=|1"),
    ("fraction", "D03", "There are five 1/8 pieces, or five-eighths.",
     "There|are|five|1/8|pieces|,|or|five-eighths|."),
    ("fraction", "D25", "1/8 is bigger than 1/6 because 8 is bigger than 6.",
     "1/8|is|bigger|than|1/6|because|8|is|bigger|than|6|."),
    ("fraction", "D25", "When I multiply 2/3 x 3/4 I keep the denominator the same so it is 6/4.",
     "When|I|multiply|2/3|x|3/4|I|keep|the|denominator|the|same|so|it|is|6/4|."),
    ("expression", "D01", "If you are, you're right, because 10/20=1/2.",
     "If|you|are|,|you're|right|,|because|10/20=1/2|."),
    ("expression", "D17", "hence 40-14=26 guests did not get a second hotdog",
     "hence|40-14=26|guests|did|not|get|a|second|hotdog"),
    ("expression", "D17", "Therefore, it will take Billy a total of 5400+300 = 5700 seconds to finish prepping the potatoes.",
     "Therefore|,|it|will|take|Billy|a|total|of|5400+300|=|5700|seconds|to|finish|prepping|the|potatoes|."),
    ("expression", "D16", "6.40/3=2.1333 dollars per kg, for the 3-kg package",
     "6.40/3=2.1333|dollars|per|kg|,|for|the|3-kg|package"),
    ("expression", "D25", "I did 1/2+1/3=2/5 because you add the tops and then add the bottoms.",
     "I|did|1/2+1/3=2/5|because|you|add|the|tops|and|then|add|the|bottoms|."),
    ("expression", "D18", "Then, the price of butter is 0.8x (80% of x) and the price of bread is 2*(0.8x) = 1.6x.",
     "Then|,|the|price|of|butter|is|0.8x|(|80%|of|x|)|and|the|price|of|bread|is|2|*|(|0.8x|)|=|1.6x|."),
    ("mixed_number", "D01", "Convert the mixed number to an improper fraction: 11 1/3.",
     "Convert|the|mixed|number|to|an|improper|fraction|:|11 1/3|."),
    ("mixed_number", "D01", "The fraction 8/5 is one whole, 1, plus three fifths, 3/5, or 1 3/5, which is read as one and three-fifths.",
     "The|fraction|8/5|is|one|whole|,|1|,|plus|three|fifths|,|3/5|,|or|1 3/5|,|which|is|read|as|one|and|three-fifths|."),
    ("mixed_number", "D01", "To locate 5/2 and -5/2, it may be helpful to rewrite them as the mixed numbers 2 1/2 and -2 1/2.",
     "To|locate|5/2|and|-|5/2|,|it|may|be|helpful|to|rewrite|them|as|the|mixed|numbers|2 1/2|and|-|2 1/2|."),
    ("mixed_number", "D25", "3 1/2 is the same as 3/2 because the 3 goes on top of the 2.",
     "3 1/2|is|the|same|as|3/2|because|the|3|goes|on|top|of|the|2|."),
    ("mixed_number", "D30", "A recipe needs 2 1/4 cups of flour. Meera has 1 1/2 cups.",
     "A|recipe|needs|2 1/4|cups|of|flour|.|Meera|has|1 1/2|cups|."),
    ("hyphenated", "D09", "To enable such tutoring, we construct DDxReasoning, a dataset of 933 clinical cases with fine-grained diagnostic steps verified by doctors.",
     "To|enable|such|tutoring|,|we|construct|DDxReasoning|,|a|dataset|of|933|clinical|cases|with|fine-grained|diagnostic|steps|verified|by|doctors|."),
    ("hyphenated", "D18", "Teacher-described confusion: Student could not grasp that the weekly nights had to be multiplied by 2.",
     "Teacher-described|confusion|:|Student|could|not|grasp|that|the|weekly|nights|had|to|be|multiplied|by|2|."),
    ("hyphenated", "D08", "71 out of 100 full-time community college faculty have a master's degree.",
     "71|out|of|100|full-time|community|college|faculty|have|a|master's|degree|."),
    ("hyphenated", "D09", "Finally, we release the source code of our VTA system, fostering future advancements in AI-driven education.",
     "Finally|,|we|release|the|source|code|of|our|VTA|system|,|fostering|future|advancements|in|AI-driven|education|."),
    ("hyphenated", "D11", "Our system is built on MPNet, a Transformer-based language model that combines BERT and XLNet's pre-training advantages.",
     "Our|system|is|built|on|MPNet|,|a|Transformer-based|language|model|that|combines|BERT|and|XLNet's|pre-training|advantages|."),
    ("acronym", "D12", "Factor analyses (EFA and CFA) and clustering showed GPT-4o reproduced the AMS structure.",
     "Factor|analyses|(|EFA|and|CFA|)|and|clustering|showed|GPT-4o|reproduced|the|AMS|structure|."),
    ("acronym", "D09", "LECTURE4ALL: A Lightweight Approach to Precise Timestamp Detection in Online Lecture Videos",
     "LECTURE4ALL|:|A|Lightweight|Approach|to|Precise|Timestamp|Detection|in|Online|Lecture|Videos"),
    ("acronym", "D11", "Our system prompts GPT-3.5-turbo to generate initial suggestions, which are then subjected to reranking.",
     "Our|system|prompts|GPT-3.5-turbo|to|generate|initial|suggestions|,|which|are|then|subjected|to|reranking|."),
    ("acronym", "D04", "The least common denominator (LCD) of two fractions is the least common multiple (LCM) of their denominators.",
     "The|least|common|denominator|(|LCD|)|of|two|fractions|is|the|least|common|multiple|(|LCM|)|of|their|denominators|."),
    ("acronym", "D09", "Models like Q-MCKT are evaluated on ASSIST2009, e.g. by a B.Tech student.",
     "Models|like|Q-MCKT|are|evaluated|on|ASSIST2009|,|e.g.|by|a|B.Tech|student|."),
    ("currency_percent", "D15", "Overall accuracy: 65.45% (including expert-validated corrections)",
     "Overall|accuracy|:|65.45%|(|including|expert-validated|corrections|)"),
    ("currency_percent", "D17", "So, her annual pension after 30 years would be $50,000 + ($50,000 x 50%) = $75,000.",
     "So|,|her|annual|pension|after|30|years|would|be|$50,000|+|(|$50,000|x|50%|)|=|$75,000|."),
    ("currency_percent", "D05", "We read $5.03 as five dollars and three cents.",
     "We|read|$5.03|as|five|dollars|and|three|cents|."),
    ("currency_percent", "D17", "Student: The interest amount Karan has to pay is 3,650 x 10/100 x 5 = $1825.",
     "Student|:|The|interest|amount|Karan|has|to|pay|is|3,650|x|10/100|x|5|=|$1825|."),
    ("currency_percent", "D30", "A cricket bat costs Rs 1,200 and the fee is ₹200 per family, 25% off.",
     "A|cricket|bat|costs|Rs|1,200|and|the|fee|is|₹200|per|family|,|25%|off|."),
    ("decimal_ratio_date", "D28", "[2026-09-15T16:04:10] Ms. Neha Kapoor: What is 0.3 x 0.2?",
     "[|2026-09-15T16:04:10|]|Ms.|Neha|Kapoor|:|What|is|0.3|x|0.2|?"),
    ("decimal_ratio_date", "D18", "She needs to cut up 1/3 x 4 = 1.33 (rounded to two decimal places) green beans.",
     "She|needs|to|cut|up|1/3|x|4|=|1.33|(|rounded|to|two|decimal|places|)|green|beans|."),
    ("decimal_ratio_date", "D17", "Carla downloaded 40% of the file before the restart, which is 0.4 x 200 = 80 GB.",
     "Carla|downloaded|40%|of|the|file|before|the|restart|,|which|is|0.4|x|200|=|80|GB|."),
    ("decimal_ratio_date", "D30", "The ratio of boys to girls in Grade 6B is 3:5. If there are 32 students, how many are girls?",
     "The|ratio|of|boys|to|girls|in|Grade|6B|is|3:5|.|If|there|are|32|students|,|how|many|are|girls|?"),
    ("decimal_ratio_date", "D25", "0.125 is bigger than 0.5 because it has more digits.",
     "0.125|is|bigger|than|0.5|because|it|has|more|digits|."),
    ("informal", "D28", "[2026-09-01T16:09:50] Aarav: the half piece and the third piece are not the same size so I cant just count them.",
     "[|2026-09-01T16:09:50|]|Aarav|:|the|half|piece|and|the|third|piece|are|not|the|same|size|so|I|cant|just|count|them|."),
    ("informal", "D20", "Danielle: Um, I thought, um, it would be, um, two fourths.",
     "Danielle|:|Um|,|I|thought|,|um|,|it|would|be|,|um|,|two|fourths|."),
    ("informal", "D25", "idk why its wrong, 2/3 + 1/6 is 3/9 cause u add across.",
     "idk|why|its|wrong|,|2/3|+|1/6|is|3/9|cause|u|add|across|."),
    ("informal", "D24", "Graham and Kelly had something very interesting to say about why um...one half is another name for two fourths.",
     "Graham|and|Kelly|had|something|very|interesting|to|say|about|why|um|...|one|half|is|another|name|for|two|fourths|."),
]


def main():
    out = Path(__file__).resolve().parents[1] / "gold" / "tokenization_gold.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i, (ptype, doc, text, toks) in enumerate(G, 1):
            tokens = toks.split("|")
            assert "".join(tokens).replace(" ", "") == text.replace(" ", ""), (i, text)
            fh.write(json.dumps({"id": f"T{i:02d}", "problem_type": ptype, "doc_id": doc, "text": text,
                                 "tokens": tokens}, ensure_ascii=False) + "\n")
    print(f"wrote {len(G)} gold sentences, {sum(len(t.split('|')) for *_, t in G)} tokens -> {out}")


if __name__ == "__main__":
    main()
