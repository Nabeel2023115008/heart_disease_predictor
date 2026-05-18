import streamlit as st
import json
from gpt_analysis import analyze_image_and_extract
from spatial_db import insert_spatial_data
from graph_query import query_related_organs, test_connection

st.set_page_config(page_title="Heart Disease Analyzer", page_icon="🫀", layout="centered")
st.title("Intelligent Heart Disease Analyzer")

# File uploader for medical image
uploaded_image = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Save uploaded image temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_image.read())

    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)
    st.info("Analyzing image with AI... (this may take a few seconds)")

    # Analyze image using GPT (API key loaded internally)
    result = analyze_image_and_extract("temp.jpg")

    # Robust JSON parsing: handle empty/whitespace, extract JSON substring if GPT wraps it in text
    import re
    if not result or not str(result).strip():
        st.error("⚠️ Empty response received from GPT. Check the analyze_image_and_extract() implementation.")
        st.code(str(result))
        st.stop()

    # Try to find the first JSON object in the response (handles extra text around the JSON)
    match = re.search(r"(\{[\s\S]*\})", result)
    json_candidate = match.group(1) if match else result

    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as e:
        st.error("⚠️ GPT returned unstructured data or failed to produce valid JSON.")
        st.error(f"JSON parsing error: {str(e)}")
        # Show full raw response to help debugging
        st.subheader("Raw GPT response")
        st.code(result, language="json")
        st.stop()

    # Display diagnosis
    st.subheader("Diagnosis")
    diagnosis = parsed.get("diagnosis", "No diagnosis found.")
    st.write(diagnosis)

    # Display affected organs
    st.subheader("Organs Affected")
    organs = parsed.get("organs_affected", [])
    if organs:
        st.write(", ".join(organs))
    else:
        st.write("No affected organs identified.")

    # Display spatial info
    st.subheader("📍 Spatial Information (from GPT)")
    regions = parsed.get("regions", [])
    st.json(regions)

    # Insert spatial data (store both GeoPackage + CSV)
    if regions:
        for region in regions:
            label = region.get("label")
            bbox = region.get("bbox")
            if label and bbox:
                insert_spatial_data(label, bbox, image_name="temp.jpg")
        st.success("✅ Spatial data saved successfully (CSV + GeoPackage)!")

    # Test Neo4j connection and query related organs
    st.subheader("Related Organs (from Neo4j Knowledge Graph)")
    if test_connection():
        try:
            related_organs = query_related_organs(disease_name=diagnosis)
            if related_organs:
                st.success("Neo4j query successful!")
                st.write(related_organs)
            else:
                st.info("No related organs found in the Neo4j graph.")
        except Exception as e:
            st.error(f"Neo4j query failed: {e}")
    else:
        st.warning("⚠️ Could not connect to Neo4j. Please check if Neo4j is running and credentials are correct.")
