# ───────────────────────── app.py ─────────────────────────
from __future__ import annotations

# 3-party libs
import streamlit as st
import pdfminer.high_level as pdf
import docx
import spacy, wordfreq
from spacy.matcher import PhraseMatcher
from spacy.language import Language
import os
from agent_avatar import record_audio, transcribe_audio, get_ai_response, speak

# helpers
from llm_utils     import summarise_resume, summarise_jd, rank_skills
from interview_llm import next_interview_question

# stdlib
from pathlib import Path
import re, json

# ─── UI basics ──────────────────────────────────────────────────
st.set_page_config(page_title="Résumé ↔ JD parser", layout="centered")
st.markdown("""
<style>
  .main>div:first-child {max-width:860px;margin:auto;}
  textarea{font-family:'Fira Code',monospace}
  .stTabbed,.stContainer{box-shadow:0 3px 12px rgba(0,0,0,.08);
                          border-radius:.75rem;padding:1rem;}
  .token{background:#eee;border-radius:6px;padding:2px 6px;margin:2px;
         display:inline-block;font-size:0.85rem}
  .token-hit{background:#dff0d8}
</style>
""", unsafe_allow_html=True)
st.title("📄 Résumé & Job-Description Parser")

MAX_QUESTIONS = 5

# ─── uploads ────────────────────────────────────────────────────
resume_file = st.file_uploader("⇧ Upload Résumé", ["pdf", "txt"], key="resume")
jd_file     = st.file_uploader("⇧ Upload JD",     ["pdf", "txt", "docx"], key="jd")

# ─── text extraction ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_text(file) -> str:
    if not file: return ""
    suf = Path(file.name).suffix.lower()
    if suf == ".txt":  return file.read().decode("utf-8", errors="ignore")
    if suf == ".pdf":  return pdf.extract_text(file)
    if suf == ".docx":
        doc = docx.Document(file); return "\n".join(p.text for p in doc.paragraphs)
    return ""

# ─── spaCy bootstrap ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_nlp() -> Language:
    try: return spacy.load("en_core_web_sm")
    except OSError:
        with st.spinner("Downloading spaCy model …"):
            from spacy.cli import download; download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# ─── matcher setup ──────────────────────────────────────────────
STOP_WORDS = {*wordfreq.top_n_list("en", 2000),
              *"january february march april … december".split(),
              "responsibilities","summary","objective"}

ACTION_VERBS = {"achieve","administer","advise","analyze","architect","build",
    "collaborate","create","debug","define","design","develop","drive","enhance",
    "establish","evaluate","execute","facilitate","implement","improve","lead",
    "manage","optimize","organize","own","plan","prioritize","research","resolve",
    "scale","support","test","troubleshoot","upgrade","validate","deliver",
    "monitor","ensure","provide","partner","review"}

TECH_GAZETTE = ["Python","Java","JavaScript","SQL","PostgreSQL","MySQL","AWS",
    "GCP","Docker","Kubernetes","Terraform","Pandas","NumPy","React","FastAPI",
    "Django"]

CLEAN_SUFFIXES = (" programming"," development"," developer"," engineer",
                  " language"," framework"," library")

@st.cache_resource(show_spinner=False)
def get_matchers():
    nlp = get_nlp()
    pm  = PhraseMatcher(nlp.vocab, attr="LOWER")
    pm.add("TECH", [nlp.make_doc(t) for t in TECH_GAZETTE])
    return nlp, pm

# ─── extraction helpers ─────────────────────────────────────────
def _norm(skill:str)->str:
    s = skill.rstrip(".,;:- ").lower()
    for suf in CLEAN_SUFFIXES:
        if s.endswith(suf): s = s[:-len(suf)]
    return s.strip()

def _pretty(s:str)->str: return s.replace("_"," ").replace("-"," ").title()

def extract_skills(text:str)->set[str]:
    nlp,pm = get_matchers(); doc = nlp(text)
    skills = {doc[s:e].text for _,s,e in pm(doc)}
    sec = re.search(r"(?i)skills?\s*[:-–]\s*(.+)", text.replace("\n"," "))
    if sec: skills.update(re.split(r"[·•,;/]", sec.group(1)))
    if doc.has_annotation("DEP"):
        for ch in doc.noun_chunks:
            if 1<=len(ch)<=3 and (ch.text.istitle() or
               re.fullmatch(r"[A-Z0-9+#\-.]{2,}", ch.text)):
                skills.add(ch.text)
    return {_pretty(s) for s in (_norm(t) for t in skills)
            if 2<len(s)<30 and s not in STOP_WORDS}

def extract_bullets(text:str)->list[str]:
    bullets=[]; clean=text.replace("\xa0"," ")
    rx = re.compile(r"^[ \t]*(?:[-–—•*]|[\u2022-\u2024])[ \t]*(.+?)(?=\n|$)", re.M)
    bullets+= [l.strip() for l in rx.findall(clean)]
    if len(bullets)<10:
        sent_rx=re.compile(r"^[A-Z][^.\n]{10,150}$", re.M)
        for line in sent_rx.findall(clean):
            if line.split()[0].lower().rstrip("s,.;") in ACTION_VERBS:
                bullets.append(line);  
            if len(bullets)>=60: break
    if len(bullets)<10:
        head_rx=re.compile(r"^[ \t]*(?:[-–—•*]|[\u2022-\u2024])?[ \t]*"
                           r"[A-Z][A-Za-z0-9 &/()+\-]{2,50}:\s+.+", re.M)
        for m in head_rx.findall(clean):
            line=m.strip(); 
            if line not in bullets: bullets.append(line)
            if len(bullets)>=60: break
    return bullets[:60]

# ─── LLM summary wrappers ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def _sum_resume(t:str)->str: return summarise_resume(t)
@st.cache_data(show_spinner=False)
def _sum_jd(t:str)->str:     return summarise_jd(t)

# ═════════════════════════ main logic ══════════════════════════
if resume_file and jd_file:
    st.success("Files received — parsing …")
    res_txt = extract_text(resume_file)
    jd_txt  = extract_text(jd_file)

    # summaries (used for prompt!)
    res_summary = _sum_resume(res_txt)
    jd_summary  = _sum_jd(jd_txt)

    with st.spinner("Ranking top-5 skills …"):
        res_sk = set(rank_skills(extract_skills(res_txt), res_txt))
        jd_sk  = set(rank_skills(extract_skills(jd_txt), jd_txt))
    overlap = sorted(res_sk & jd_sk, key=str.lower)

    # ─── interview state ────────────────────────────────────────
    if "chat" not in st.session_state:   st.session_state.chat=[]
    if "q_count" not in st.session_state:st.session_state.q_count=0

    with st.sidebar.expander("🎙 Live interview", expanded=False):
     if st.button("Click to Speak"):
        with st.spinner("Listening..."):
            try:
                filename = record_audio()
                if not filename or not os.path.exists(filename):
                    st.error("Recording failed. Please try again.")
                else:
                    user_input = transcribe_audio(filename)
                    st.markdown(f"**You said:** {user_input}")

                    with st.spinner("Thinking..."):
                        response = get_ai_response(user_input)
                    st.markdown(f"**Interviewer:** {response}")
                    speak(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # ─── tabs ───────────────────────────────────────────────────
    raw_tab, parsed_tab = st.tabs(["📑 Previews","🔎 Parsed"])

    with raw_tab:
        c1,c2 = st.columns(2)
        c1.subheader("Résumé (first 2000 chars)")
        c1.text_area("", res_txt[:2000], height=280, label_visibility="collapsed")
        c2.subheader("JD (first 2000 chars)")
        c2.text_area("", jd_txt[:2000], height=280, label_visibility="collapsed")

    with parsed_tab:
        c1,c2 = st.columns(2); c1.write(res_summary); c2.write(jd_summary); st.divider()

        l,r = st.columns(2)
        l.subheader("🛠 Résumé skills"); r.subheader("🛠 JD skills")
        for col,sk in ((l,res_sk),(r,jd_sk)):
            if sk:
                with col.expander(f"{len(sk)} skills"):
                    cols=st.columns(3)
                    for i,s in enumerate(sorted(sk,key=str.lower)):
                        cols[i%3].markdown(f"<span class='token'>{s}</span>", unsafe_allow_html=True)
            else: col.caption("— none detected —")

        l.subheader("📌 Résumé bullets")
        for b in extract_bullets(res_txt): l.markdown(f"- {b}")
        r.subheader("📌 JD responsibilities")
        for b in extract_bullets(jd_txt):  r.markdown(f"- {b}")

        st.subheader(f"🎯 Skill overlap ({len(overlap)})")
        if overlap:
            st.markdown(" ".join(f"<span class='token token-hit'>{s}</span>" for s in overlap),
                        unsafe_allow_html=True)
        else: st.caption("No overlap yet — maybe refine the gazetteer?")

        if jd_sk:
            st.metric("Match score", f"{int(100*len(overlap)/len(jd_sk))}%")

        if st.button("📥 Download summaries & skill-match", key="dl_btn"):
            fn="resume_jd_summary.json"
            with open(fn,"w",encoding="utf-8") as f:
                json.dump({
                    "resume_summary":res_summary,"jd_summary":jd_summary,
                    "resume_skills":sorted(res_sk),"jd_skills":sorted(jd_sk),
                    "overlap":overlap}, f, indent=2)
            with open(fn,"rb") as f: st.download_button("Download JSON", f, file_name=fn,
                                                        mime="application/json")
else:
    st.warning("⬆️ Please upload **both** files to continue.")