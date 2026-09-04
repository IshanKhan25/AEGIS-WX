def status_card(st, label, value, detail=""):
    st.metric(label, value, detail if detail else None)
