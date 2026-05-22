"""
ai_analyzer.py
Pure Python NLP - No API key needed.
Features: Summary, Keywords, Q&A, Sentiment, Section Insights, Error Detection
"""

import re
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

try:
    import language_tool_python
    grammar_tool = language_tool_python.LanguageTool('en-US')
except Exception:
    grammar_tool = None

# ── Stopwords ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","dare","ought","used","this","that","these","those",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","what","which","who","whom",
    "when","where","why","how","all","each","every","both","few","more",
    "most","other","some","such","no","not","only","same","so","than",
    "too","very","just","as","if","then","because","while","although",
    "however","therefore","thus","hence","also","about","after","before",
    "during","through","between","into","up","down","out","off","over",
    "under","again","further","once","here","there","any","s","t"
}

# ── Sentiment Words ────────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","outstanding",
    "positive","success","successful","achieve","achievement","benefit","beneficial",
    "improve","improvement","effective","efficient","innovative","creative","best",
    "better","superior","advantage","opportunity","growth","progress","strong",
    "powerful","significant","important","valuable","useful","helpful","support",
    "recommend","approve","love","like","enjoy","happy","pleased","satisfied",
    "perfect","ideal","optimal","remarkable","impressive","exceptional","brilliant"
}

NEGATIVE_WORDS = {
    "bad","poor","terrible","awful","horrible","negative","fail","failure","problem",
    "issue","concern","risk","danger","harmful","damage","loss","decline","decrease",
    "weak","ineffective","inefficient","difficult","challenge","obstacle","barrier",
    "limitation","disadvantage","threat","crisis","error","mistake","wrong","worse",
    "worst","inferior","inadequate","insufficient","lack","missing","absent","broken",
    "corrupt","fraud","illegal","violation","abuse","attack","conflict","dispute"
}

# ── Common Misspellings Dictionary ─────────────────────────────────────────────
COMMON_MISSPELLINGS = {
    "teh":"the","recieve":"receive","occured":"occurred","seperate":"separate",
    "definately":"definitely","accomodate":"accommodate","occurance":"occurrence",
    "independant":"independent","existance":"existence","persistance":"persistence",
    "consistant":"consistent","relevent":"relevant","prominant":"prominent",
    "dominance":"dominance","grammer":"grammar","writting":"writing","speling":"spelling",
    "adress":"address","begining":"beginning","beleive":"believe","calender":"calendar",
    "cemetary":"cemetery","collegue":"colleague","comming":"coming","commitee":"committee",
    "concious":"conscious","curiousity":"curiosity","dissapear":"disappear",
    "embarass":"embarrass","enviroment":"environment","excercise":"exercise",
    "existance":"existence","familier":"familiar","finaly":"finally","foriegn":"foreign",
    "freind":"friend","goverment":"government","gaurd":"guard","happend":"happened",
    "harrass":"harass","hieght":"height","humourous":"humorous","ignorence":"ignorance",
    "immediatly":"immediately","independance":"independence","intresting":"interesting",
    "knowlege":"knowledge","liason":"liaison","libary":"library","lisence":"license",
    "maintainance":"maintenance","millenium":"millennium","mischievious":"mischievous",
    "misspell":"misspell","neccessary":"necessary","nieghbor":"neighbor","noticable":"noticeable",
    "occassion":"occasion","paralell":"parallel","parliment":"parliament","peice":"piece",
    "percieve":"perceive","perseverence":"perseverance","phenomemon":"phenomenon",
    "posession":"possession","potatos":"potatoes","prefered":"preferred","privelege":"privilege",
    "probaly":"probably","proffesional":"professional","publically":"publicly",
    "reccomend":"recommend","refered":"referred","religous":"religious","remeber":"remember",
    "repitition":"repetition","resistence":"resistance","responsability":"responsibility",
    "rythm":"rhythm","sargent":"sergeant","sieze":"seize","similer":"similar",
    "speach":"speech","succesful":"successful","suprise":"surprise","tatoo":"tattoo",
    "tendancy":"tendency","tommorow":"tomorrow","tounge":"tongue","truely":"truly",
    "twelth":"twelfth","tyrany":"tyranny","untill":"until","vaccum":"vacuum",
    "visious":"vicious","wierd":"weird","whereever":"wherever","wich":"which",
    "wont":"won't","youre":"you're","dont":"don't","cant":"can't","isnt":"isn't",
    "wasnt":"wasn't","werent":"weren't","hasnt":"hasn't","havent":"haven't",
    "doesnt":"doesn't","didnt":"didn't","wouldnt":"wouldn't","couldnt":"couldn't",
    "shouldnt":"shouldn't","thier":"their","alot":"a lot","alright":"all right",
    "untill":"until","occured":"occurred","begining":"beginning","writen":"written"
}

# ── Passive Voice Patterns ─────────────────────────────────────────────────────
PASSIVE_PATTERN = re.compile(
    r'\b(is|are|was|were|be|been|being)\s+([\w]+ed|[\w]+en)\b',
    re.IGNORECASE
)

# ── Filler / Weak Words ────────────────────────────────────────────────────────
WEAK_WORDS = {
    "very","really","quite","rather","somewhat","fairly","pretty","basically",
    "actually","literally","honestly","clearly","obviously","simply","just",
    "thing","things","stuff","nice","good","bad","big","small","get","got","make"
}

# ── Double Word Pattern ────────────────────────────────────────────────────────
DOUBLE_WORD_PATTERN = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_document(text: str, chunks: list) -> dict:
    """Run all NLP analysis and error detection."""
    sentences  = split_sentences(text)
    words      = tokenize(text)
    word_freq  = get_word_freq(words)
    errors     = detect_errors(text, sentences, words)
    readability = analyze_readability(text, sentences, words)
    doc_type   = detect_document_type(text, words)
    scores     = compute_scores(errors, readability)

    return {
        "summary":          generate_summary(sentences, word_freq),
        "short_summary":    generate_short_summary(sentences, word_freq),
        "keywords":         extract_keywords(word_freq),
        "qa_pairs":         generate_qa(sentences, word_freq),
        "sentiment":        analyze_sentiment(words),
        "section_insights": generate_section_insights(chunks) if len(chunks) > 1 else [],
        "errors":           errors,
        "readability":      readability,
        "doc_type":         doc_type,
        "ai_feedback":      generate_ai_feedback(text, sentences, words, errors, readability, doc_type),
        "scores":           scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def tokenize(text: str) -> list:
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())


def get_word_freq(words: list) -> Counter:
    return Counter(w for w in words if w not in STOPWORDS)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def score_sentences(sentences: list, word_freq: Counter) -> list:
    max_freq = max(word_freq.values(), default=1)
    tf = {w: c / max_freq for w, c in word_freq.items()}
    scored = []
    for s in sentences:
        w = tokenize(s)
        if not w:
            continue
        score = sum(tf.get(x, 0) for x in w) / len(w)
        scored.append((score, s))
    return sorted(scored, reverse=True)


def generate_short_summary(sentences: list, word_freq: Counter) -> str:
    """Generate a 1-2 sentence short summary."""
    if not sentences:
        return "No content to summarize."
    scored = score_sentences(sentences, word_freq)
    top = scored[0][1] if scored else sentences[0]
    # Try to get a second sentence for context
    if len(scored) > 1:
        return top + " " + scored[1][1]
    return top


def generate_summary(sentences: list, word_freq: Counter, n: int = 5) -> str:
    if not sentences:
        return "No content to summarize."
        
    # Advanced AI Summarization using LSA
    try:
        text = " ".join(sentences)
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, n)
        result = [str(s) for s in summary_sentences]
        if result:
            return " ".join(result)
    except Exception as e:
        print(f"Advanced summarization failed: {e}")

    # Fallback to local frequency-based summarization
    scored = score_sentences(sentences, word_freq)
    top = {s for _, s in scored[:n]}
    result = [s for s in sentences if s in top]
    return " ".join(result) if result else sentences[0]


# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════

def extract_keywords(word_freq: Counter, n: int = 12) -> list:
    if not word_freq:
        return []
    total = sum(word_freq.values())
    tf = {w: c / total for w, c in word_freq.items()}
    return [w for w, _ in sorted(tf.items(), key=lambda x: x[1], reverse=True)[:n]]


# ══════════════════════════════════════════════════════════════════════════════
#  Q&A GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_qa(sentences: list, word_freq: Counter, n: int = 5) -> list:
    scored = score_sentences(sentences, word_freq)
    qa_pairs = []
    for _, sent in scored:
        if len(qa_pairs) >= n:
            break
        qa = sentence_to_qa(sent)
        if qa:
            qa_pairs.append(qa)
    return qa_pairs


def sentence_to_qa(sentence: str) -> dict | None:
    s = sentence.strip()
    if len(s) < 30:
        return None
    m = re.match(r'^(.+?)\s+(is|are|was|were)\s+(.+)', s, re.IGNORECASE)
    if m:
        subject = m.group(1).strip()
        verb    = m.group(2).strip()
        if len(subject.split()) <= 6:
            return {"question": f"What {verb} {subject}?", "answer": s}
    if re.match(r'^[A-Z][a-z]', s):
        topic = " ".join(s.split()[:4])
        return {"question": f"What does the document say about '{topic}...'?", "answer": s}
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(words: list) -> dict:
    pos   = sum(1 for w in words if w in POSITIVE_WORDS)
    neg   = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg

    if total == 0:
        return {"sentiment": "Neutral", "confidence": "Medium",
                "explanation": "No strong sentiment indicators found.", "pos": 0, "neg": 0}

    ratio = pos / total
    if ratio >= 0.65:
        s, c = "Positive", "High" if ratio >= 0.80 else "Medium"
        exp  = f"Document has {pos} positive and {neg} negative indicators."
    elif ratio <= 0.35:
        s, c = "Negative", "High" if ratio <= 0.20 else "Medium"
        exp  = f"Document has {neg} negative and {pos} positive indicators."
    else:
        s, c = "Neutral", "Medium"
        exp  = f"Balanced mix: {pos} positive and {neg} negative indicators."

    return {"sentiment": s, "confidence": c, "explanation": exp, "pos": pos, "neg": neg}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def generate_section_insights(chunks: list) -> list:
    insights = []
    for i, chunk in enumerate(chunks[:6]):
        sentences = split_sentences(chunk)
        words     = tokenize(chunk)
        freq      = get_word_freq(words)
        keywords  = extract_keywords(freq, n=4)
        top_sent  = score_sentences(sentences, freq)
        preview   = top_sent[0][1] if top_sent else (sentences[0] if sentences else "")
        insights.append({
            "section":  i + 1,
            "insight":  preview[:200],
            "keywords": keywords
        })
    return insights


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_errors(text: str, sentences: list, words: list) -> dict:
    """
    Detects:
    - Spelling mistakes (common misspellings)
    - Double/repeated words
    - Very long sentences (readability)
    - Passive voice usage
    - Weak/filler words
    - Repeated sentences
    - Missing punctuation at sentence end
    - Capitalization errors after period
    """
    errors = {
        "spelling":          [],
        "grammar":           [],
        "double_words":      [],
        "long_sentences":    [],
        "passive_voice":     [],
        "weak_words":        [],
        "repeated_sentences":[],
        "punctuation":       [],
        "capitalization":    [],
        "summary": {}
    }

    # 1. Spelling mistakes
    raw_words = re.findall(r'\b[a-zA-Z]+\b', text)
    seen_misspellings = set()
    for w in raw_words:
        lower = w.lower()
        if lower in COMMON_MISSPELLINGS and lower not in seen_misspellings:
            seen_misspellings.add(lower)
            errors["spelling"].append({
                "word":       w,
                "suggestion": COMMON_MISSPELLINGS[lower]
            })

    # 1b. Grammar & Style checking via AI (language-tool)
    if grammar_tool:
        try:
            matches = grammar_tool.check(text)
            for match in matches[:15]: # Limit to top 15 to avoid huge response
                if match.ruleId != 'MORFOLOGIK_RULE_EN_US': # Skip spell check rules (handled above)
                    errors["grammar"].append({
                        "issue": match.message,
                        "context": match.context,
                        "replacements": [r for r in match.replacements[:3]]
                    })
        except Exception as e:
            print(f"Grammar check failed: {e}")

    # 2. Double/repeated words
    for match in DOUBLE_WORD_PATTERN.finditer(text):
        errors["double_words"].append({
            "found":   match.group(0),
            "word":    match.group(1),
            "pos":     match.start()
        })

    # 3. Long sentences (> 40 words = hard to read)
    for sent in sentences:
        wc = len(sent.split())
        if wc > 40:
            errors["long_sentences"].append({
                "sentence": sent[:120] + ("..." if len(sent) > 120 else ""),
                "word_count": wc
            })

    # 4. Passive voice
    seen_passive = set()
    for sent in sentences:
        matches = PASSIVE_PATTERN.findall(sent)
        if matches:
            key = sent[:60]
            if key not in seen_passive:
                seen_passive.add(key)
                errors["passive_voice"].append({
                    "sentence": sent[:120] + ("..." if len(sent) > 120 else ""),
                    "pattern":  f"{matches[0][0]} {matches[0][1]}"
                })

    # 5. Weak/filler words
    weak_found = Counter(w for w in words if w in WEAK_WORDS)
    errors["weak_words"] = [
        {"word": w, "count": c}
        for w, c in weak_found.most_common(10)
        if c >= 2
    ]

    # 6. Repeated sentences (near-duplicates)
    seen_sents = {}
    for sent in sentences:
        key = re.sub(r'\s+', ' ', sent.lower().strip())[:80]
        if key in seen_sents:
            errors["repeated_sentences"].append({
                "sentence": sent[:100] + ("..." if len(sent) > 100 else "")
            })
        else:
            seen_sents[key] = True

    # 7. Missing punctuation (sentence not ending in . ! ?)
    for sent in sentences:
        stripped = sent.strip()
        if stripped and stripped[-1] not in ".!?":
            errors["punctuation"].append({
                "sentence": stripped[:100] + ("..." if len(stripped) > 100 else ""),
                "issue":    "Missing punctuation at end"
            })

    # 8. Capitalization after period
    cap_errors = re.findall(r'[.!?]\s+([a-z][a-zA-Z]*)', text)
    for word in cap_errors[:10]:
        errors["capitalization"].append({
            "word":  word,
            "issue": f"'{word}' should be capitalized after sentence end"
        })

    # Summary counts
    errors["summary"] = {
        "spelling_count":     len(errors["spelling"]),
        "grammar_count":      len(errors.get("grammar", [])),
        "double_words_count": len(errors["double_words"]),
        "long_sentences_count": len(errors["long_sentences"]),
        "passive_voice_count":  len(errors["passive_voice"]),
        "weak_words_count":     len(errors["weak_words"]),
        "repeated_sentences_count": len(errors["repeated_sentences"]),
        "punctuation_count":    len(errors["punctuation"]),
        "capitalization_count": len(errors["capitalization"]),
        "total": (
            len(errors["spelling"]) +
            len(errors.get("grammar", [])) +
            len(errors["double_words"]) +
            len(errors["long_sentences"]) +
            len(errors["passive_voice"]) +
            len(errors["punctuation"]) +
            len(errors["capitalization"])
        )
    }

    return errors


# ══════════════════════════════════════════════════════════════════════════════
#  READABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

DIFFICULT_WORD_THRESHOLD = 3  # syllables

def count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    vowels = re.findall(r'[aeiouy]+', word)
    count = len(vowels)
    if word.endswith('e'):
        count -= 1
    return max(1, count)


def analyze_readability(text: str, sentences: list, words: list) -> dict:
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    avg_sent_len = round(word_count / sent_count, 1)

    syllable_counts = [count_syllables(w) for w in words]
    total_syllables = sum(syllable_counts)
    difficult_words = [w for w, s in zip(words, syllable_counts) if s >= DIFFICULT_WORD_THRESHOLD]
    difficult_unique = list(set(difficult_words))[:15]

    # Flesch Reading Ease
    avg_syllables = total_syllables / max(word_count, 1)
    flesch = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syllables)
    flesch = round(max(0, min(100, flesch)), 1)

    # Passive voice ratio
    passive_count = len(PASSIVE_PATTERN.findall(text))
    passive_pct = round((passive_count / max(sent_count, 1)) * 100, 1)

    # Reading time (avg 200 wpm)
    reading_minutes = word_count / 200
    if reading_minutes < 1:
        reading_time = f"{round(reading_minutes * 60)} seconds"
    else:
        reading_time = f"{round(reading_minutes, 1)} minutes"

    # Level
    if flesch >= 70:
        level = "Easy"
    elif flesch >= 50:
        level = "Medium"
    else:
        level = "Difficult"

    return {
        "flesch_score": flesch,
        "level": level,
        "avg_sentence_length": avg_sent_len,
        "difficult_words": difficult_unique,
        "difficult_word_count": len(difficult_words),
        "passive_voice_pct": passive_pct,
        "reading_time": reading_time,
        "word_count": word_count,
        "sentence_count": sent_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

RESUME_SIGNALS   = {"experience","education","skills","resume","cv","objective","references","employment","internship","certification","linkedin","github","portfolio"}
ACADEMIC_SIGNALS = {"abstract","introduction","methodology","conclusion","references","hypothesis","literature","research","study","findings","analysis","journal","thesis","dissertation"}
REPORT_SIGNALS   = {"executive","summary","report","findings","recommendation","overview","appendix","objective","scope","methodology","conclusion","analysis","department"}

def detect_document_type(text: str, words: list) -> dict:
    word_set = set(words)
    resume_score   = len(word_set & RESUME_SIGNALS)
    academic_score = len(word_set & ACADEMIC_SIGNALS)
    report_score   = len(word_set & REPORT_SIGNALS)

    scores = {"Resume": resume_score, "Academic": academic_score, "Report": report_score}
    doc_type = max(scores, key=scores.get)
    if scores[doc_type] == 0:
        doc_type = "General"

    return {"type": doc_type, "signals": scores}


# ══════════════════════════════════════════════════════════════════════════════
#  AI FEEDBACK SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_ai_feedback(text: str, sentences: list, words: list, errors: dict, readability: dict, doc_type: dict) -> list:
    feedback = []
    dtype = doc_type.get("type", "General")

    # Introduction check
    first_200 = text[:200].lower()
    if not any(w in first_200 for w in ["this document","this report","this paper","introduction","overview","purpose","objective","aim"]):
        feedback.append({"icon": "✍️", "type": "warning", "title": "Improve Introduction", "detail": "The opening section lacks a clear purpose statement. Start with a strong introduction that states the document's objective."})

    # Conclusion check
    last_200 = text[-200:].lower()
    if not any(w in last_200 for w in ["conclusion","summary","therefore","thus","in summary","to conclude","finally","overall"]):
        feedback.append({"icon": "🏁", "type": "warning", "title": "Improve Conclusion", "detail": "No clear conclusion detected. Add a concluding section that summarizes key points and takeaways."})

    # Examples check
    if not any(w in text.lower() for w in ["for example","for instance","such as","e.g.","i.e.","to illustrate"]):
        feedback.append({"icon": "💡", "type": "info", "title": "Add More Examples", "detail": "The document lacks concrete examples. Adding examples improves clarity and reader engagement."})

    # Repetition
    if errors.get("summary", {}).get("repeated_sentences_count", 0) > 0:
        feedback.append({"icon": "🔁", "type": "warning", "title": "Reduce Repetition", "detail": f"Found {errors['summary']['repeated_sentences_count']} repeated sentence(s). Remove or rephrase duplicates to improve flow."})

    # Passive voice
    if readability.get("passive_voice_pct", 0) > 30:
        feedback.append({"icon": "🔄", "type": "warning", "title": "Reduce Passive Voice", "detail": f"Passive voice used in ~{readability['passive_voice_pct']}% of sentences. Prefer active voice for stronger, clearer writing."})

    # Readability
    if readability.get("level") == "Difficult":
        feedback.append({"icon": "📖", "type": "danger", "title": "Enhance Clarity", "detail": f"Readability score is {readability['flesch_score']}/100 (Difficult). Shorten sentences and replace complex words with simpler alternatives."})
    elif readability.get("level") == "Medium":
        feedback.append({"icon": "📖", "type": "info", "title": "Improve Readability", "detail": f"Readability score is {readability['flesch_score']}/100 (Medium). Consider simplifying some sentences for a broader audience."})

    # Weak words
    if errors.get("summary", {}).get("weak_words_count", 0) >= 3:
        feedback.append({"icon": "💪", "type": "info", "title": "Strengthen Word Choice", "detail": "Multiple weak/filler words detected (e.g. 'very', 'really', 'basically'). Replace them with precise, impactful language."})

    # Long sentences
    if errors.get("summary", {}).get("long_sentences_count", 0) > 2:
        feedback.append({"icon": "✂️", "type": "warning", "title": "Break Long Sentences", "detail": f"{errors['summary']['long_sentences_count']} sentences exceed 40 words. Split them for better readability."})

    # Document-type specific feedback
    if dtype == "Resume":
        feedback.append({"icon": "🎯", "type": "ats", "title": "ATS Optimization", "detail": "Use standard section headings (Experience, Education, Skills). Include measurable achievements (e.g. 'Increased sales by 30%'). Avoid tables and graphics that ATS systems cannot parse."})
        feedback.append({"icon": "🔑", "type": "ats", "title": "Keyword Density", "detail": "Mirror keywords from the job description in your resume. ATS systems rank resumes by keyword match rate."})
    elif dtype == "Academic":
        feedback.append({"icon": "📚", "type": "academic", "title": "Citation & References", "detail": "Ensure all claims are backed by citations. Use a consistent citation style (APA, MLA, Chicago)."})
        feedback.append({"icon": "🧪", "type": "academic", "title": "Methodology Clarity", "detail": "Clearly describe your research methodology so results are reproducible."})
    elif dtype == "Report":
        feedback.append({"icon": "📊", "type": "report", "title": "Structure & Professionalism", "detail": "Ensure the report follows: Executive Summary → Introduction → Findings → Recommendations → Conclusion."})
        feedback.append({"icon": "📈", "type": "report", "title": "Data & Evidence", "detail": "Support recommendations with data, charts, or statistics to increase credibility."})

    # Spelling
    if errors.get("summary", {}).get("spelling_count", 0) > 0:
        feedback.append({"icon": "🔤", "type": "danger", "title": "Fix Spelling Errors", "detail": f"{errors['summary']['spelling_count']} spelling mistake(s) found. Proofread carefully before submission."})

    # Professionalism (short document)
    if readability.get("word_count", 0) < 100:
        feedback.append({"icon": "📝", "type": "info", "title": "Expand Content", "detail": "The document is very short. Consider adding more detail, context, and supporting information."})

    return feedback


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(errors: dict, readability: dict) -> dict:
    err_summary = errors.get("summary", {})
    total_errors = err_summary.get("total", 0)
    word_count = max(readability.get("word_count", 1), 1)

    # Grammar Score: penalize per error relative to document size
    error_rate = total_errors / (word_count / 100)
    grammar_score = max(0, round(100 - (error_rate * 8)))
    grammar_score = min(grammar_score, 100)

    # Readability Score: based on Flesch
    readability_score = min(100, max(0, round(readability.get("flesch_score", 50))))

    # Content Quality: based on word count, passive voice, weak words
    passive_penalty = min(30, readability.get("passive_voice_pct", 0) * 0.5)
    weak_penalty = min(10, err_summary.get("weak_words_count", 0) * 2)
    repeat_penalty = min(10, err_summary.get("repeated_sentences_count", 0) * 3)
    content_score = max(0, round(90 - passive_penalty - weak_penalty - repeat_penalty))
    if word_count > 300:
        content_score = min(100, content_score + 10)

    overall = round((grammar_score * 0.35) + (readability_score * 0.30) + (content_score * 0.35))

    return {
        "grammar": grammar_score,
        "readability": readability_score,
        "content_quality": content_score,
        "overall": overall,
    }
