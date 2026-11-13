"""Phase 1 · Visual review of positives: tag each forged ID with its fraud modality.

Local Streamlit app. Shows front and back of one POSITIVE at a time; one click on a modality
saves it and moves to the next. Results go to data/<stage>/review/modalities.csv (one row per
person). It resumes where it left off, and "Undo last" takes back the last answer.

By default it goes through every positive of the manifest. With --queue it shows only the
people listed in a CSV with an `anon_id` column, in that order: for example the positives the
classifier did not catch (runs/<name>/missed_positives.csv, written by 33_combine_sides.py).
The scores of the model are not shown, so the answer does not lean on them.

Run:  streamlit run scripts/13_visual_review.py
      streamlit run scripts/13_visual_review.py -- --queue runs/ensemble/missed_positives.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from id_fraud_detector.common import load_config, output_dir

MODALITIES = {
    "genuine_doc_stolen_identity": "Looks genuine (real document, fraud was elsewhere)",
    "replaced_photo": "Photo replaced (real data, another face)",
    "pasted_photo_template": "Photo pasted on a template",
    "cloned_back": "Back copied from a template",
    "assembled_back": "Back assembled from pieces",
    "screen_or_print_recapture": "Photo of a screen or a printout",
    "other": "Forged, something else (say what in the note)",
    "unsure": "Cannot tell",
}
COLUMNS = ["anon_id", "modality", "notes"]

ap = argparse.ArgumentParser()
ap.add_argument("--queue", help="CSV with an anon_id column: review only these people, in this order")
ap.add_argument("--out", help="where the answers are saved (default data/<stage>/review/modalities.csv)")
args, _ = ap.parse_known_args(sys.argv[1:])

cfg = load_config()
out = output_dir(cfg)
review_path = Path(args.out) if args.out else out / "review" / "modalities.csv"
review_path.parent.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(out / "manifest.csv")
positives = manifest[(manifest.label == "POSITIVE") & manifest.front_path.notna()]
if args.queue:
    queue = pd.read_csv(args.queue)["anon_id"]
    positives = positives.set_index("anon_id").reindex(queue).dropna(subset=["front_path"]).reset_index()
done = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame(columns=COLUMNS)
pending = positives[~positives.anon_id.isin(done.anon_id)].reset_index(drop=True)
in_scope = done[done.anon_id.isin(positives.anon_id)]

st.set_page_config(layout="wide", page_title="Visual review")
st.title("Visual review of positives")
st.caption(f"{len(in_scope)} tagged · {len(pending)} pending" + (f" · queue: {args.queue}" if args.queue else ""))

if len(done) and st.button("Undo last"):
    done.iloc[:-1].to_csv(review_path, index=False)
    st.rerun()

if pending.empty:
    st.success("Everything in this list is tagged.")
    st.dataframe(in_scope.modality.value_counts())
    st.stop()

row = pending.iloc[0]
c1, c2 = st.columns(2)
c1.image(str(out / row.front_path), caption="front", width="stretch")
# A plain if/else: Streamlit cannot parse a bare expression that continues on the next line.
if isinstance(row.back_path, str):
    c2.image(str(out / row.back_path), caption="back", width="stretch")
else:
    c2.warning("no back image")
st.write(f"**{row.anon_id}** · {row.tenant} · registered {str(row.registered_at)[:10]} · {row.channel} · {row.label_source}")

notes = st.text_input("Note (optional, write it before choosing; no names or ID numbers)", key=f"notes_{row.anon_id}")
st.write("What do you see? One click saves and shows the next one.")
buttons = st.columns(4)
for i, (modality, text) in enumerate(MODALITIES.items()):
    if buttons[i % 4].button(text, key=f"{modality}_{row.anon_id}", width="stretch"):
        answer = pd.DataFrame([{"anon_id": row.anon_id, "modality": modality, "notes": notes}])
        pd.concat([done, answer]).to_csv(review_path, index=False)
        st.rerun()
