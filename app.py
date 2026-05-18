import streamlit as st
import json
import re
from gpt_analysis import analyze_image_and_extract
from spatial_db import insert_spatial_data, get_spatial_data, clear_spatial_data
from graph_query import query_related_organs, test_connection
from PIL import Image, ImageDraw, ImageFont

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

    # Analyze image using GPT
    try:
        result = analyze_image_and_extract("temp.jpg")
    except Exception as e:
        st.error("⚠️ Could not contact GPT API. Check network / API key / proxy settings.")
        st.error(str(e))
        st.stop()

    # Robust JSON parsing
    if not result or not str(result).strip():
        st.error("⚠️ Empty response received from GPT.")
        st.stop()

    # Extract JSON from response
    match = re.search(r"(\{[\s\S]*\})", result)
    json_candidate = match.group(1) if match else result

    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as e:
        st.error("⚠️ GPT returned invalid JSON.")
        st.error(f"JSON parsing error: {str(e)}")
        st.subheader("Raw GPT response")
        st.code(result, language="json")
        st.stop()

    # Display diagnosis
    st.subheader("🔍 Diagnosis")
    diagnosis = parsed.get("diagnosis", "No diagnosis found.")
    st.write(diagnosis)

    # Display affected organs
    st.subheader("🫁 Organs Affected")
    organs = parsed.get("organs_affected", [])
    if organs:
        st.write(", ".join(organs))
    else:
        st.success("No affected organs identified - appears normal.")

    # Get all regions and filter for abnormalities only
    all_regions = parsed.get("regions", [])
    
    # Filter abnormal regions - trust the 'abnormal' flag from GPT
    abnormal_regions = []
    for region in all_regions:
        # Only include if explicitly marked as abnormal
        if region.get("abnormal") == True:
            abnormal_regions.append(region)

    # Clear previous spatial DB data
    clear_spatial_data()

    # Save abnormalities to spatial DB and display them
    if abnormal_regions:
        st.subheader("⚠️ Abnormalities Detected")
        
        saved_regions = []
        for region in abnormal_regions:
            # Get abnormality details
            abnormality_name = region.get("abnormality") or region.get("label") or "unknown"
            bbox = region.get("bbox")
            confidence = region.get("confidence", 0.0)
            
            if bbox and len(bbox) == 4:
                try:
                    # Save to spatial DB
                    insert_spatial_data(abnormality_name, bbox, image_name="temp.jpg")
                    saved_regions.append(region)
                    
                    # Display in UI
                    st.write(f"• **{abnormality_name}** (confidence: {confidence:.0%})")
                except Exception as e:
                    st.warning(f"Could not save {abnormality_name}: {e}")
        
        st.success(f"✅ {len(saved_regions)} abnormality(ies) detected and saved.")
        
        # Create annotated image
        st.subheader("📊 Annotated Image")
        
        try:
            img = Image.open("temp.jpg").convert("RGBA")
            w, h = img.size
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # Load font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()

            # Draw each abnormality
            for region in saved_regions:
                bbox = region["bbox"]
                label = region.get("abnormality") or region.get("label") or "abnormality"
                confidence = region.get("confidence", 0.0)
                
                # Convert coordinates to pixels
                try:
                    nums = [float(x) for x in bbox]
                except:
                    continue
                
                # Check if normalized (0-1) or absolute coordinates
                if all(0.0 <= v <= 1.0 for v in nums):
                    x1 = int(nums[0] * w)
                    y1 = int(nums[1] * h)
                    x2 = int(nums[2] * w)
                    y2 = int(nums[3] * h)
                else:
                    x1, y1, x2, y2 = map(int, nums)
                
                # Clip to image bounds
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(x1+1, min(w, x2))
                y2 = max(y1+1, min(h, y2))
                
                # Ensure valid box
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Color based on confidence
                if confidence >= 0.7:
                    color = (255, 0, 0)  # Red - high confidence
                    alpha = 80
                elif confidence >= 0.4:
                    color = (255, 165, 0)  # Orange - medium confidence
                    alpha = 60
                else:
                    color = (255, 255, 0)  # Yellow - low confidence
                    alpha = 40
                
                # Draw rectangle
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=(*color, alpha),
                    outline=(*color, 255),
                    width=3
                )
                
                # Draw center marker
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                marker_size = 6
                draw.ellipse(
                    [cx-marker_size, cy-marker_size, cx+marker_size, cy+marker_size],
                    fill=(255, 255, 255, 255),
                    outline=(0, 0, 0, 255),
                    width=2
                )
                
                # Prepare label text
                label_text = f"{label} ({confidence:.0%})"
                
                # Get text dimensions
                try:
                    bbox_text = draw.textbbox((0, 0), label_text, font=font)
                    text_w = bbox_text[2] - bbox_text[0]
                    text_h = bbox_text[3] - bbox_text[1]
                except:
                    # Fallback for older Pillow versions
                    text_w, text_h = draw.textsize(label_text, font=font)
                
                # Position label above bounding box
                label_x = x1
                label_y = max(5, y1 - text_h - 10)
                
                # Draw label background
                padding = 4
                draw.rectangle(
                    [label_x - padding, label_y - padding,
                     label_x + text_w + padding, label_y + text_h + padding],
                    fill=(0, 0, 0, 200)
                )
                
                # Draw label text
                draw.text((label_x, label_y), label_text, fill=(255, 255, 255, 255), font=font)

            # Composite and save
            annotated = Image.alpha_composite(img, overlay).convert("RGB")
            annotated_path = "temp_annotated.jpg"
            annotated.save(annotated_path, quality=95)
            
            st.image(annotated_path, caption="Detected Abnormalities Highlighted", use_column_width=True)
            
        except Exception as e:
            st.error(f"Could not create annotated image: {e}")
    else:
        st.success("✅ No abnormalities detected - image appears normal.")

    # Query Neo4j for related information
    st.subheader("🔗 Related Organs & Conditions")
    if test_connection():
        try:
            related_info = query_related_organs(disease_name=diagnosis)
            if related_info:
                st.success("Connected to medical knowledge graph")
                st.write(related_info)
            else:
                st.info("No additional related information found in knowledge graph.")
        except Exception as e:
            st.warning(f"Knowledge graph query failed: {e}")
    else:
        st.warning("⚠️ Could not connect to Neo4j knowledge graph.")

# Sidebar controls
st.sidebar.subheader("🛠️ Maintenance")
if st.sidebar.button("Clear All Spatial Data"):
    confirm = st.sidebar.checkbox("I confirm I want to clear all data")
    if confirm:
        if clear_spatial_data():
            st.sidebar.success("✅ Spatial database cleared.")
            st.rerun()
        else:
            st.sidebar.info("No data to clear.")

# Add helpful information in sidebar
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    "This tool uses AI to analyze medical images for potential cardiac abnormalities. "
    "Upload chest X-rays or cardiac imaging for automated analysis."
)

st.sidebar.subheader("📋 Detection Confidence")
st.sidebar.markdown("🔴 **Red**: High confidence (≥70%)")
st.sidebar.markdown("🟠 **Orange**: Medium confidence (40-70%)")
st.sidebar.markdown("🟡 **Yellow**: Low confidence (<40%)")