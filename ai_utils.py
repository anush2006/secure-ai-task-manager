import re
import numpy as np
import spacy #type:ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests #type:ignore

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

def normalize_text(text):
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

_token_regex = re.compile(r"[A-Za-z0-9]+")

def tokenize_words(text, use_spacy=False):
    cleaned_text = normalize_text(text)
    if use_spacy:
        tokens = []
        doc = nlp(cleaned_text)
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            if tok.lemma_ is not None:
                token_text = tok.lemma_.lower()
            else:
                token_text = tok.text.lower()
            if _token_regex.fullmatch(token_text):
                tokens.append(token_text)
        return tokens
    simple_tokens = _token_regex.findall(cleaned_text)
    simple_tokens = [tok.lower() for tok in simple_tokens]
    return simple_tokens

def summarize_text(text, max_sentences=2):
    cleaned_text = normalize_text(text)
    if cleaned_text is None or cleaned_text.strip() == "":
        return ""
    doc = nlp(cleaned_text)
    raw_sentences = []
    for sent in doc.sents:
        line = sent.text.strip()
        if line != "":
            raw_sentences.append(line)
    if len(raw_sentences) == 0:
        parts = re.split(r"[.!?]\s+", cleaned_text)
        for s in parts:
            if s.strip() != "":
                raw_sentences.append(s.strip())
    if len(raw_sentences) <= max_sentences:
        return " ".join(raw_sentences)
    vectorizer = TfidfVectorizer(
        tokenizer=lambda s: tokenize_words(s, use_spacy=True),
        lowercase=True,
        min_df=1
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(raw_sentences)
    except ValueError:
        return " ".join(raw_sentences[:max_sentences])
    importance_scores = []
    for i in range(tfidf_matrix.shape[0]):
        importance_scores.append(float(tfidf_matrix[i].sum()))
    importance_scores = np.array(importance_scores)
    max_imp = importance_scores.max() + 1e-12
    norm_imp = importance_scores / max_imp
    centroid = tfidf_matrix.mean(axis=0)
    centrality_scores = cosine_similarity(tfidf_matrix, centroid).flatten()
    max_cen = centrality_scores.max() + 1e-12
    norm_cen = centrality_scores / max_cen
    final_scores = []
    for imp, cen in zip(norm_imp, norm_cen):
        final_scores.append(float(0.6 * imp + 0.4 * cen))
    final_scores = np.array(final_scores)
    sorted_idx = final_scores.argsort()[::-1]
    selected_idx = sorted(sorted_idx[:max_sentences])
    return " ".join(raw_sentences[i] for i in selected_idx)

def build_tfidf_index(task_list):
    docs = []
    ids = []
    for task in task_list:
        combined = ""
        for field in ["title", "description", "category", "status"]:
            value = task.get(field)
            if value:
                combined += " " + str(value)
        docs.append(combined.strip())
        ids.append(task["id"])
    vectorizer = TfidfVectorizer(
        tokenizer=lambda t: tokenize_words(t, use_spacy=False),
        lowercase=True,
        min_df=1,
        ngram_range=(1, 2)
    )
    if len(docs) > 0:
        matrix = vectorizer.fit_transform(docs)
    else:
        matrix = None
    return {
        "vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "ids": ids,
        "docs": docs
    }

def semantic_search(query_text, index, top_k=5):
    if index["tfidf_matrix"] is None:
        return []
    ids = index["ids"]
    vectorizer = index["vectorizer"]
    matrix = index["tfidf_matrix"]
    cleaned_query = normalize_text(query_text)
    if cleaned_query == "":
        return []
    query_vector = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vector, matrix)[0]
    sorted_idx = scores.argsort()[::-1]
    top_idx = sorted_idx[:top_k]
    results = []
    for i in top_idx:
        score = float(scores[i])
        if score > 0:
            results.append((ids[i], score))
    return results

def parse_text_spacy(text):
    doc = nlp(text)
    verbs = []
    nouns = []
    noun_chunks = []
    direct_objects = []
    entities = []
    for token in doc:
        if token.pos_ == "VERB" or token.tag_.startswith("VB"):
            lemma = token.lemma_ if token.lemma_ else token.text
            verbs.append(lemma.lower())
        if token.pos_ in ("NOUN", "PROPN"):
            nouns.append(token.text.lower())
        if token.dep_ in ("dobj", "obj", "iobj"):
            direct_objects.append(token.text.lower())
    if hasattr(doc, "noun_chunks"):
        for chunk in doc.noun_chunks:
            noun_chunks.append(chunk.text.lower())
    if hasattr(doc, "ents"):
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
    return {
        "doc": doc,
        "verbs": verbs,
        "nouns": nouns,
        "noun_chunks": noun_chunks,
        "direct_objects": direct_objects,
        "entities": entities
    }

def generate_subtasks(task):
    text = (task.get("title") or "") + " " + (task.get("description") or "")
    
    prompt = f"""
You MUST return a JSON array. No prose. No commentary.
Output format EXACTLY:

[
  {{"title":"...", "description":"...", "estimate_hours":number}},
  {{"title":"...", "description":"...", "estimate_hours":number}}
]

Generate 6 subtasks for the task.
TASK: "{text}"
"""
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt}, stream=True)

    merged = ""
    for line in response.iter_lines():
        if not line:
            continue

        try:
            js = json.loads(line.decode())
            if "response" in js:
                merged += js["response"]
        except:
            continue

    print("\n========= RAW MERGED RESPONSE =========")
    print(merged[:600])
    print("=======================================\n")
    try:
        return json.loads(merged)
    except:
        pass
    match = re.search(r"\[.*\]", merged, re.DOTALL)
    if match:
        block = match.group(0)
        try:
            return json.loads(block)
        except:
            block = block.replace(",]", "]").replace(", }", "}")
            return json.loads(block)

    return [{"error": "Still not valid JSON", "raw": merged[:500]}]