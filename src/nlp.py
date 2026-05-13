import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import CodedSegment, Segment

# Common English stopwords plus interview filler words
_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it","its",
    "itself","they","them","their","theirs","themselves","what","which","who","whom",
    "this","that","these","those","am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing","a","an","the","and","but",
    "if","or","because","as","until","while","of","at","by","for","with","about",
    "against","between","into","through","during","before","after","above","below",
    "to","from","up","down","in","out","on","off","over","under","again","further",
    "then","once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own","same",
    "so","than","too","very","can","will","just","should","now","d","ll","m","re",
    "ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
    # interview filler
    "yeah","yes","okay","ok","like","know","think","really","well","um","uh","ah",
    "actually","kind","sort","thing","things","something","anything","everything",
    "lot","going","get","got","go","goes","went","say","said","says","saying",
    "also","even","way","back","one","two","three","make","made","need","want",
    "would","could","maybe","might","much","many","little","time","every","quite",
    "people","person","someone","everyone","somebody","anyone","s","t","n",
}


def _clean(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in _STOPWORDS and len(w) > 2)


def word_frequencies(segments: List[Segment], top_n: int = 50) -> Dict[str, int]:
    combined = " ".join(_clean(s.text) for s in segments)
    return dict(Counter(combined.split()).most_common(top_n))


def tfidf_keywords(
    segments: List[Segment], top_n: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """Top TF-IDF keywords per source file."""
    file_texts: Dict[str, List[str]] = {}
    for seg in segments:
        file_texts.setdefault(seg.source_file, []).append(_clean(seg.text))

    if len(file_texts) < 2:
        return {}

    sources = list(file_texts.keys())
    docs = [" ".join(file_texts[s]) for s in sources]

    try:
        vec = TfidfVectorizer(max_features=500, min_df=1, sublinear_tf=True)
        mat = vec.fit_transform(docs)
        names = vec.get_feature_names_out()
    except Exception:
        return {}

    result: Dict[str, List[Tuple[str, float]]] = {}
    for i, src in enumerate(sources):
        scores = mat[i].toarray()[0]
        top = np.argsort(scores)[::-1][:top_n]
        result[src] = [
            (names[j], round(float(scores[j]), 4)) for j in top if scores[j] > 0
        ]
    return result


def code_frequencies(coded_segments: List[CodedSegment]) -> Dict[str, int]:
    counter: Counter = Counter()
    for cs in coded_segments:
        for code in cs.codes:
            counter[code.strip().lower()] += 1
    return dict(counter.most_common())


def run_nlp(
    segments: List[Segment], coded_segments: Optional[List[CodedSegment]] = None
) -> Dict:
    result: Dict = {
        "word_frequencies": word_frequencies(segments),
        "tfidf_keywords": tfidf_keywords(segments),
        "segment_count": len(segments),
        "source_files": sorted({s.source_file for s in segments}),
        "participants": sorted({s.participant for s in segments if s.participant}),
        "total_words": sum(s.word_count for s in segments),
    }
    if coded_segments:
        result["code_frequencies"] = code_frequencies(coded_segments)
    return result
