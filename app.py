if st.button("🔍 Predict"):

    proba = model.predict_proba(features)
    confidence = float(np.max(proba))
    pred = int(np.argmax(proba))

    # -------------------- SMART VALIDATION --------------------
    
    if confidence < 0.60:
        st.error("❌ Invalid Image: Not a brain scan")
        st.write(f"Confidence: {confidence:.2f}")

    elif 0.60 <= confidence < 0.80:
        st.warning("⚠️ Low confidence prediction (image may be unclear)")

        if pred == 0:
            st.error("⚠️ Hemorrhagic Stroke (Low Confidence)")
        else:
            st.success("✅ Ischemic Stroke (Low Confidence)")

        st.write(f"Confidence: {confidence:.2f}")

    else:
        if pred == 0:
            st.error("⚠️ Hemorrhagic Stroke Detected")
        else:
            st.success("✅ Ischemic Stroke Detected")

        st.write(f"Confidence: {confidence:.2f}")
