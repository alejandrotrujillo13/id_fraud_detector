"""Phase 1 · Visual review of positives: tag each forged ID with its fraud modality.

Local Streamlit app. Shows front and back of each POSITIVE in the manifest and saves one row
per customer to data/<stage>/review/modalities.csv. Resumes where it left off.

Run:  streamlit run scripts/13_visual_review.py
"""

import pandas as pd
import streamlit as st

from id_fraud_detector.common import load_config, output_dir

MODALITIES = [
    "cloned_back",            # whole back copied from a template (same code/serial as other IDs)
    "assembled_back",         # back built from segments of real IDs
    "replaced_photo",         # real data, photo of the fraudster
    "pasted_photo_template",  # photo pasted on a generic template
    "screen_or_print_recapture",
    "genuine_doc_stolen_identity",  # document looks real; fraud was elsewhere -> hard negative
    "other",
    "unsure",
]

cfg = load_config()
out = output_dir(cfg)
review_dir = out / "review"
review_dir.mkdir(parents=True, exist_ok=True)
review_path = review_dir / "modalities.csv"

manifest = pd.read_csv(out / "manifest.csv")
positives = manifest[(manifest.label == "POSITIVE") & manifest.front_path.notna()].reset_index(drop=True)
done = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame(columns=["anon_id", "modality", "notes"])
pending = positives[~positives.anon_id.isin(done.anon_id)].reset_index(drop=True)

st.set_page_config(layout="wide", page_title="Visual review")
st.title("Visual review of positives")
st.caption(f"{len(done)} tagged · {len(pending)} pending")

if pending.empty:
    st.success("All positives are tagged.")
    st.dataframe(done.modality.value_counts())
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

modality = st.radio("Modality", MODALITIES, horizontal=True)
notes = st.text_input("Notes (optional)")
if st.button("Save and next", type="primary"):
    pd.concat([done, pd.DataFrame([{"anon_id": row.anon_id, "modality": modality, "notes": notes}])]) \
        .to_csv(review_path, index=False)
    st.rerun()
