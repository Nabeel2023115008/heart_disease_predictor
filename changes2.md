import streamlit as st
import json
from gpt_analysis import analyze_image_and_extract
from spatial_db import insert_spatial_data
from graph_query import query_related_organs, test_connection
from PIL import Image, ImageDraw, ImageFont  # added import

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
    try:
        result = analyze_image_and_extract("temp.jpg")
    except Exception as e:
        # handle networking / API connection failures gracefully
        st.error("⚠️ Could not contact GPT API. Check network / API key / proxy settings.")
        st.error(str(e))
        # Optional: use a local fallback so you can continue testing annotation/UI
        fallback = {
            "diagnosis": "simulated heart abnormality",
            "organs_affected": ["Heart", "Lungs"],
            "regions": [
                {"label": "heart", "bbox": [100, 150, 300, 350]},
                {"label": "lung_left", "bbox": [320, 180, 500, 400]}
            ]
        }
        # Continue using fallback JSON so UI still shows annotation
        result = json.dumps(fallback)

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

    # Annotate image with regions (draw rectangles + translucent fill) and display
    try:
        img = Image.open("temp.jpg").convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for region in regions:
            label = region.get("label", "region")
            bbox = region.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = map(int, bbox[:4])
            # semi-transparent fill + colored border
            fill_color = (255, 0, 0, 60)   # red translucent
            outline_color = (255, 0, 0, 200)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=3)

            # optional: label text (simple)
            try:
                font = ImageFont.load_default()
                text_pos = (x1 + 4, max(0, y1 - 12))
                draw.rectangle([text_pos, (text_pos[0] + 8 + len(label)*6, text_pos[1] + 12)], fill=(0,0,0,140))
                draw.text(text_pos, label, fill=(255,255,255,255), font=font)
            except Exception:
                pass

        annotated = Image.alpha_composite(img, overlay).convert("RGB")
        annotated_path = "temp_annotated.jpg"
        annotated.save(annotated_path, quality=90)
        st.image(annotated_path, caption="Annotated Image", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not annotate image: {e}")

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
