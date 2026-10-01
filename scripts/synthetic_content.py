"""Content of the six clearly labelled synthetic documents (D25-D30).

Why synthetic documents exist at all: the real learner-language sources are
MathDial (students are LLM-simulated, teachers are real) and TalkMoves (real
classroom talk, but short turns). Neither contains many free-text student
*explanations* of fraction procedures, which is the signal the research problem
is about. These documents fill that gap and are always marked synthetic=true in
document_catalog.csv. Every explanation line is tied to a named malrule, listed
in gold/synthetic_labels.csv, and no model is ever evaluated on labels it made.
"""

# D25 - student explanations. (text, malrule_id)  M00 = correct reasoning, SLIP = slip
EXPLANATIONS = [
    ("I did 1/2+1/3=2/5 because you add the tops and then add the bottoms.", "M01"),
    ("I added the top numbers and the bottom numbers so 2/7 + 3/7 = 5/14.", "M01"),
    ("For 3/4 + 1/4 I got 4/8 because 3 plus 1 is 4 and 4 plus 4 is 8.", "M01"),
    ("idk why its wrong, 2/3 + 1/6 is 3/9 cause u add across.", "M01"),
    ("1/8 is bigger than 1/6 because 8 is bigger than 6.", "M02"),
    ("The fraction with the larger denominator is always the larger fraction, so 3/10 > 3/5.", "M02"),
    ("1/100 is more than 1/10 since a hundred pieces is a lot more pieces.", "M02"),
    ("When I multiply 2/3 x 3/4 I keep the denominator the same so it is 6/4.", "M03"),
    ("To multiply fractions you need a common denominator first, so I changed 1/2 x 1/3 into 3/6 x 2/6.", "M03"),
    ("For 3/4 divided by 1/2 I flipped the 3/4 and got 4/3 x 1/2 = 4/6.", "M04"),
    ("Dividing by 1/2 makes it smaller, so 6 divided by 1/2 is 3.", "M04"),
    ("0.125 is bigger than 0.5 because it has more digits.", "M05"),
    ("0.45 is greater than 0.6 since 45 is greater than 6.", "M05"),
    ("0.3 x 0.2 = 0.6 because 3 times 2 is 6.", "M06"),
    ("Multiplying always makes a number bigger so 8 x 0.5 must be more than 8.", "M06"),
    ("If a price goes down 50% and then up 50% it is back to the same price.", "M07"),
    ("A shirt costs Rs 400, 25% off and then another 25% off means 50% off, so it is Rs 200.", "M07"),
    ("To add 2/3 + 1/4 I cross multiplied and got 8/3.", "M08"),
    ("3½ is the same as 3/2 because the 3 goes on top of the 2.", "M09"),
    ("2 1/4 + 1 1/2 = 3 2/6 because I added the wholes and then added the fractions across.", "M09"),
    ("2/3 is equal to 4/5 because I added 2 to the top and 2 to the bottom.", "M10"),
    ("You can make an equivalent fraction by adding the same number to both parts, like 1/2 = 3/4.", "M10"),
    ("3/4 is not one number, it is a 3 and a 4 so you cant put it on a number line.", "M11"),
    ("A fraction cannot be bigger than 1, so 5/4 is not a real fraction.", "M11"),
    ("I knew I needed a common denominator, so 1/2 + 1/3 = 3/6 + 2/6, but I wrote 6/6 by mistake.", "SLIP"),
    ("I found the LCD of 4 and 6 is 12, then 3/12 + 10/12 = 13/12 oops I meant 3/12 + 2/12.", "SLIP"),
    ("I changed 1/2 + 1/3 into 3/6 + 2/6 so the answer is 5/6.", "M00"),
    ("You need the same denominator because you can only add pieces that are the same size.", "M00"),
    ("1/6 is bigger than 1/8 because when the whole is cut into fewer pieces each piece is bigger.", "M00"),
    ("0.5 is bigger than 0.125 because 0.5 is 0.500 and 500 thousandths is more than 125 thousandths.", "M00"),
    ("To divide 3/4 by 1/2 I keep 3/4, change to multiply and flip 1/2, so 3/4 x 2/1 = 6/4 = 1 1/2.", "M00"),
    ("2/3 = 4/6 because I multiplied the top and the bottom by 2.", "M00"),
    ("Multiplying by a number less than 1 like 0.5 makes the answer smaller, 8 x 0.5 = 4.", "M00"),
    ("I think 1/2 + 1/2 = 2/4, which is still a half so it works.", "M01"),
    ("dont you just add the numerators and add the denominators? my answer was 4/10 for 1/5 + 3/5.", "M01"),
    ("i said 7/12 is less than 1/3 because 12 is a big number so the pieces are tiny.", "M02"),
    ("The answer to 12.5% of 80 is 12.5 because percent means the same number.", "M07"),
    ("50% of 40 is 20, then I added 50% of 20 to get back to 40.", "M07"),
    ("When the denominators are different I add them anyway, 1/4 + 1/2 = 2/6.", "M01"),
    ("I multiplied top and bottom by different numbers, 3/5 = 6/15.", "M10"),
]

MALRULES = {
    "M00": "Correct reasoning (control)",
    "M01": "Adds numerators and adds denominators",
    "M02": "Larger denominator means larger fraction",
    "M03": "Keeps or equalises denominators when multiplying",
    "M04": "Inverts the wrong fraction / division always makes smaller",
    "M05": "Longer decimal is bigger",
    "M06": "Multiplication always makes bigger (decimals treated as whole numbers)",
    "M07": "Percentages treated as additive or as absolute amounts",
    "M08": "Cross-multiplies when adding",
    "M09": "Misreads mixed numbers",
    "M10": "Equivalent fractions by adding the same number",
    "M11": "Fraction seen as two separate whole numbers",
    "SLIP": "Correct procedure, arithmetic slip",
}

# D26 - questions students ask during a fractions lesson
QUESTIONS = [
    "Why do we need the same denominator to add but not to multiply?",
    "Is 3/4 bigger than 2/3 or are they the same?",
    "What does LCD mean and how is it different from LCM?",
    "Can a fraction be bigger than 1?",
    "How do u turn 0.75 into a fraction?",
    "Why does dividing by 1/2 make the number bigger?",
    "If I add 1/2 and 1/2 do I get 2/4 or 1?",
    "Is 0.5 the same as 50%?",
    "Why is 1/8 smaller than 1/6 when 8 is bigger?",
    "Do I flip the first fraction or the second one when I divide?",
    "What is a mixed number and how do I change 3½ into an improper fraction?",
    "Can the denominator ever be 0?",
    "Why does 0.3 x 0.2 give 0.06 and not 0.6?",
    "How do I know which fraction is closer to 1 on the number line?",
    "Is 12.5% the same as 1/8?",
    "Why do we simplify fractions if the answer is already right?",
    "What is the difference between a ratio and a fraction?",
    "When I cross multiply am I adding or comparing?",
    "Is 2/4 really equal to 1/2 if the pieces look different?",
    "How do I add 2 1/4 and 1 1/2 without making them improper?",
    "What does it mean when the numerator is bigger than the denominator?",
    "Why can't I just add 2 to the top and bottom to get an equivalent fraction?",
    "Is a percent always out of 100?",
    "How do I compare 0.45 and 0.6 without a calculator?",
    "What is the reciprocal of 5?",
    "Does multiplying always make things bigger?",
    "If the price goes up 10% and then down 10% is it the same?",
    "idk how to find a common denominator for 4 and 6, do I just multiply them?",
    "Why is 5/6 bigger than 4/5 if both are one piece away from a whole?",
    "Can a decimal be written as a fraction every time?",
]

# D27 - teacher observation notes (DOCX)
TEACHER_NOTES = {
    "title": "Grade 6B Fractions Unit - Teacher Observation Notes",
    "meta": [
        "Teacher: Ms. Priya Raman",
        "School: Greenfield Public School, Bengaluru, Karnataka",
        "Unit: Fractions, decimals and percents (NCERT Class 6, Chapter 7)",
        "Observation period: 3 August 2026 to 28 August 2026",
    ],
    "entries": [
        ("3 August 2026", "Readiness quiz on equivalent fractions. 14 of 32 students wrote 2/3 = 4/5, adding the same number to the numerator and the denominator. Rohan and Meera explained it as 'keeping the difference the same'."),
        ("5 August 2026", "Fraction strips activity. Most students could show 1/2 + 1/4 with strips, but on paper 9 students still wrote 1/2 + 1/4 = 2/6. The model and the written rule are not yet connected."),
        ("7 August 2026", "Comparing unit fractions. Aarav insisted 1/8 is bigger than 1/6 because 8 is bigger. After folding paper he said 'the pieces get smaller when there are more of them', which is the idea we want."),
        ("12 August 2026", "Common denominators. Students used the LCD for 4 and 6 correctly (12) but several multiplied the denominators every time (24), which works but makes simplification harder. Not a misconception, just an inefficient strategy."),
        ("14 August 2026", "Decimals. Ishaan said 0.125 is bigger than 0.5 because it is longer. Five others agreed. Used a place-value chart and money: Rs 0.50 versus Rs 0.125 is not a real price, which confused them more. Switch to metre sticks next time."),
        ("19 August 2026", "Multiplying decimals. The class believes multiplication always makes numbers bigger. 8 x 0.5 = 4 surprised almost everyone. Plan: area model on grid paper."),
        ("21 August 2026", "Percents. Diwali sale word problem: 20% off and then 20% off again. 18 students said 40% off. Only Kavya calculated the second discount on the new price."),
        ("26 August 2026", "Dividing fractions. Many students can recite 'keep, change, flip' but flip the wrong fraction. Nobody could explain why the rule works. Use measurement division: how many 1/2 cups are in 3 cups?"),
        ("28 August 2026", "Unit test. Common errors, in order: adding denominators (11 students), larger denominator means larger fraction (7), longer decimal is bigger (6), percent discounts added (9)."),
    ],
    "next_steps": [
        "Reteach addition with the number line before the symbolic rule.",
        "Order 40 sets of fraction tiles for the class (budget Rs 2,500, approved by the Principal, Mr. Suresh Kumar).",
        "Prepare the team for the Inter-School Math Olympiad on 18 October 2026 in Mysuru.",
        "Share progress with parents on Google Classroom by 4 September 2026.",
        "Try the GeoGebra fraction applet for comparing fractions.",
    ],
}

# D28 - tutor chat log, one learner over three sessions (JSON)
TUTOR_CHAT = {
    "learner_id": "pseudonymous_A17",
    "learner_name": "Aarav",
    "tutor": "Ms. Neha Kapoor",
    "platform": "school tutoring pilot, Bengaluru",
    "sessions": [
        {"session": 1, "date": "2026-09-01", "turns": [
            {"t": "2026-09-01T16:02:10", "speaker": "tutor", "text": "What is 1/2 + 1/3?"},
            {"t": "2026-09-01T16:02:41", "speaker": "learner", "text": "2/5"},
            {"t": "2026-09-01T16:02:55", "speaker": "tutor", "text": "How did you get 2/5?"},
            {"t": "2026-09-01T16:03:30", "speaker": "learner", "text": "I added the top and the bottom numbers. 1 plus 1 is 2 and 2 plus 3 is 5."},
            {"t": "2026-09-01T16:04:05", "speaker": "tutor", "text": "Is 2/5 bigger or smaller than 1/2?"},
            {"t": "2026-09-01T16:04:40", "speaker": "learner", "text": "smaller i think. that is weird because I added more."},
            {"t": "2026-09-01T16:05:20", "speaker": "tutor", "text": "Good noticing. Let's cut a roti into halves and thirds and see what size the pieces are."},
            {"t": "2026-09-01T16:09:50", "speaker": "learner", "text": "the half piece and the third piece are not the same size so I cant just count them."},
        ]},
        {"session": 2, "date": "2026-09-08", "turns": [
            {"t": "2026-09-08T16:01:05", "speaker": "tutor", "text": "What is 2/7 + 3/7?"},
            {"t": "2026-09-08T16:01:30", "speaker": "learner", "text": "5/7 because the pieces are already the same size, sevenths."},
            {"t": "2026-09-08T16:02:15", "speaker": "tutor", "text": "And 1/4 + 1/2?"},
            {"t": "2026-09-08T16:03:10", "speaker": "learner", "text": "I need a common denominator. 1/2 is 2/4 so it is 3/4."},
            {"t": "2026-09-08T16:04:00", "speaker": "tutor", "text": "What about 2/3 + 1/4?"},
            {"t": "2026-09-08T16:05:30", "speaker": "learner", "text": "3/7? no wait. the denominators are different. 8/12 + 3/12 = 11/12."},
        ]},
        {"session": 3, "date": "2026-09-15", "turns": [
            {"t": "2026-09-15T16:00:40", "speaker": "tutor", "text": "Last week we added fractions. Can you explain to a friend why 1/2 + 1/3 is not 2/5?"},
            {"t": "2026-09-15T16:02:10", "speaker": "learner", "text": "Because halves and thirds are different sized pieces. You change both to sixths first, 3/6 + 2/6 = 5/6. 2/5 is less than 1/2 so it cant be the answer."},
            {"t": "2026-09-15T16:03:00", "speaker": "tutor", "text": "Which is bigger, 1/8 or 1/6?"},
            {"t": "2026-09-15T16:03:25", "speaker": "learner", "text": "1/6, because sixths are bigger pieces than eighths."},
            {"t": "2026-09-15T16:04:10", "speaker": "tutor", "text": "What is 0.3 x 0.2?"},
            {"t": "2026-09-15T16:04:50", "speaker": "learner", "text": "0.6"},
            {"t": "2026-09-15T16:05:20", "speaker": "tutor", "text": "Is that bigger or smaller than 0.3?"},
            {"t": "2026-09-15T16:06:15", "speaker": "learner", "text": "bigger. but multiplying by 0.2 is like taking a part of it so it should be smaller. maybe 0.06?"},
        ]},
    ],
}

# D29 - class web page (HTML) with deliberate boilerplate: nav, cookie banner, footer, script
CLASS_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Grade 6 Fractions Unit | Greenfield Public School Maths Department</title>
<meta name="description" content="Lesson plan, weekly schedule and parent information for the Grade 6 fractions unit.">
<script>window.analytics = { track: function () {} }; /* tracking stub */</script>
<style>body{font-family:sans-serif} nav a{margin-right:1em}</style>
</head>
<body>
<div class="cookie-banner">We use cookies to improve your experience. <button>Accept all cookies</button></div>
<nav>
  <a href="/">Home</a><a href="/about">About Us</a><a href="/admissions">Admissions</a>
  <a href="/departments/maths">Maths Department</a><a href="/contact">Contact</a><a href="/login">Parent Login</a>
</nav>
<main>
<article>
<h1>Grade 6 Fractions Unit: Lesson Plan and Schedule</h1>
<p class="byline">Posted by Ms. Priya Raman, Head of Mathematics, on 1 August 2026</p>
<p>This four-week unit builds understanding of fractions, decimals and percents before procedures.
Students first model fractions with strips, number lines and area diagrams, and only then learn the
symbolic rules. The unit follows NCERT Class 6, Chapter 7, and the Karnataka state syllabus.</p>
<h2>Learning objectives</h2>
<ul>
<li>Explain why fractions need a common denominator before they can be added or subtracted.</li>
<li>Compare fractions with unlike denominators using benchmarks such as 1/2 and 1.</li>
<li>Convert between fractions, decimals and percents, for example 3/4 = 0.75 = 75%.</li>
<li>Multiply and divide fractions and explain why dividing by 1/2 doubles a number.</li>
</ul>
<h2>Weekly schedule</h2>
<table>
<tr><th>Week</th><th>Dates</th><th>Focus</th></tr>
<tr><td>1</td><td>3 August 2026 to 7 August 2026</td><td>Equivalent fractions and comparing fractions</td></tr>
<tr><td>2</td><td>10 August 2026 to 14 August 2026</td><td>Adding and subtracting fractions with common and unlike denominators</td></tr>
<tr><td>3</td><td>17 August 2026 to 21 August 2026</td><td>Decimals, place value and decimal multiplication</td></tr>
<tr><td>4</td><td>24 August 2026 to 28 August 2026</td><td>Percents, discounts and the unit test</td></tr>
</table>
<h2>Common misconceptions we will address</h2>
<p>Adding numerators and adding denominators (1/2 + 1/3 = 2/5); believing a larger denominator means a
larger fraction; thinking a longer decimal such as 0.125 is bigger than 0.5; and adding successive
percentage discounts. Each misconception gets a diagnostic question at the start of the lesson.</p>
<h2>Events</h2>
<p>The Parent Maths Workshop is on 25 September 2026 in the school auditorium (fee Rs 200 per family).
Selected students will represent the school at the Inter-School Math Olympiad on 18 October 2026 in
Mysuru. The Fraction Fair 2026 exhibition opens on 6 November 2026.</p>
</article>
</main>
<aside class="sidebar"><h3>Related links</h3><ul><li><a href="/events">School calendar</a></li><li><a href="/news">Latest news</a></li></ul></aside>
<footer>
<p>&copy; 2026 Greenfield Public School, Bengaluru. All rights reserved.</p>
<p><a href="/privacy">Privacy policy</a> | <a href="/terms">Terms of use</a> | Follow us on social media</p>
</footer>
<script>document.querySelector('.cookie-banner button').onclick = function () { this.parentNode.remove(); };</script>
</body>
</html>
"""

# D30 - word-problem worksheet (PDF)
WORKSHEET = {
    "title": "Fractions, Decimals and Percents - Practice Worksheet (Grade 6)",
    "header": "Greenfield Public School, Bengaluru | Set by Ms. Priya Raman | Due 11 September 2026",
    "problems": [
        "Riya bought 3/4 kg of apples on Monday and 1/2 kg on Tuesday at the Mysuru market. How many kilograms did she buy in all? Explain why you cannot write 4/6 kg.",
        "A cricket bat costs Rs 1,200. During the Diwali sale it is 25% off, and members get another 10% off the sale price. What does a member pay?",
        "Arjun says 0.125 is greater than 0.5 because it has more digits. Is he right? Use a place-value chart to explain.",
        "A recipe needs 2 1/4 cups of flour. Meera has 1 1/2 cups. How much more does she need?",
        "The school bus travels 18.6 km from Whitefield to Koramangala every morning. How far does it travel in 5 school days?",
        "Kabir scored 42 out of 50 in the Inter-School Math Olympiad practice test on 5 September 2026. What percent is that?",
        "A water tank in Chennai is 3/5 full. After 1/4 of the water is used, what fraction of the tank is full?",
        "An Amul milk packet of 500 ml costs Rs 28. What is the price per litre?",
        "Which is larger, 5/6 or 4/5? Draw a number line to justify your answer.",
        "Tara says 3/4 divided by 1/2 is 3/8. Find her mistake and correct it.",
        "A Samsung tablet costs Rs 18,000 in Mumbai and Rs 17,100 in Pune. By what percent is the Pune price lower?",
        "The ratio of boys to girls in Grade 6B is 3:5. If there are 32 students, how many are girls?",
    ],
}
