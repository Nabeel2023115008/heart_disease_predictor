import os
import json
import re
import base64
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Use a more capable model for better medical image analysis
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEBUG_DUMP = "last_model_response.txt"

def _get_client():
    api_key = os.getenv("INSERT API KEY HERE")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set the env var or add it to .env, then restart Streamlit.\n"
            "Example (PowerShell): $env:OPENAI_API_KEY='sk-...'\n"
        )
    return OpenAI(api_key=api_key)

# Enhanced few-shot examples covering multiple cardiac abnormality scenarios
_EXAMPLES = [
    {
        "scenario": "Cardiomegaly (enlarged heart)",
        "json": {
            "diagnosis": "cardiomegaly with possible left ventricular enlargement",
            "organs_affected": ["Heart"],
            "regions": [
                {
                    "label": "enlarged_heart",
                    "abnormality": "cardiomegaly",
                    "abnormal": True,
                    "bbox": [0.35, 0.25, 0.65, 0.75],
                    "confidence": 0.88
                }
            ]
        }
    },
    {
        "scenario": "Pulmonary edema",
        "json": {
            "diagnosis": "pulmonary edema bilateral lung fields",
            "organs_affected": ["Lungs"],
            "regions": [
                {
                    "label": "right_lung_edema",
                    "abnormality": "pulmonary_edema",
                    "abnormal": True,
                    "bbox": [0.55, 0.20, 0.85, 0.70],
                    "confidence": 0.82
                },
                {
                    "label": "left_lung_edema",
                    "abnormality": "pulmonary_edema",
                    "abnormal": True,
                    "bbox": [0.15, 0.20, 0.45, 0.70],
                    "confidence": 0.80
                }
            ]
        }
    },
    {
        "scenario": "Pericardial effusion",
        "json": {
            "diagnosis": "pericardial effusion moderate",
            "organs_affected": ["Heart"],
            "regions": [
                {
                    "label": "heart_pericardial_region",
                    "abnormality": "pericardial_effusion",
                    "abnormal": True,
                    "bbox": [0.38, 0.30, 0.62, 0.70],
                    "confidence": 0.91
                }
            ]
        }
    },
    {
        "scenario": "Normal chest X-ray",
        "json": {
            "diagnosis": "normal cardiac and pulmonary examination",
            "organs_affected": [],
            "regions": []
        }
    }
]

def analyze_image_and_extract(image_path: str, retries: int = 3, backoff: float = 1.5) -> str:
    """
    Analyzes medical image and returns JSON with accurate abnormality detection.
    Uses normalized coordinates (0-1) for consistency.
    
    Returns:
        JSON string with structure:
        {
            "diagnosis": str,
            "organs_affected": list[str],
            "regions": [
                {
                    "label": str,
                    "abnormality": str,
                    "abnormal": bool,
                    "bbox": [x1, y1, x2, y2],  # normalized 0-1
                    "confidence": float
                }
            ]
        }
    """
    # Read and encode image
    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    # Build examples text
    examples_text = "\n\n".join([
        f"EXAMPLE - {ex['scenario']}:\n{json.dumps(ex['json'], indent=2)}"
        for ex in _EXAMPLES
    ])

    system = (
        "You are an expert radiologist with extensive experience in cardiac imaging. "
        "Your task is to carefully analyze medical images and identify any abnormalities. "
        "You MUST return ONLY a valid JSON object with no additional text, explanations, markdown formatting, or code blocks. "
        "Be thorough and identify ALL visible abnormalities with appropriate confidence scores. "
        "Do not be overly conservative - if you see something that could be abnormal, include it with an appropriate confidence level."
    )
    
    user = (
        "Analyze this medical image and return EXACTLY ONE JSON object with this structure:\n\n"
        "{\n"
        '  "diagnosis": "<detailed clinical diagnosis or \'normal cardiac and pulmonary examination\'>",\n'
        '  "organs_affected": ["<organ1>", "<organ2>", ...],  // empty array if normal\n'
        '  "regions": [\n'
        '    {\n'
        '      "label": "<descriptive_label>",\n'
        '      "abnormality": "<medical_term>",  // e.g., cardiomegaly, pleural_effusion\n'
        '      "abnormal": true,\n'
        '      "bbox": [x1, y1, x2, y2],  // MUST use normalized 0-1 coordinates\n'
        '      "confidence": 0.XX  // 0.0 to 1.0\n'
        '    }\n'
        "  ]  // empty array if no abnormalities\n"
        "}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. **Coordinates**: Use ONLY normalized coordinates (0.0 to 1.0) for bbox\n"
        "   - Format: [left, top, right, bottom] where each value is between 0 and 1\n"
        "   - Example: [0.3, 0.2, 0.7, 0.8] represents a box from 30% to 70% horizontal, 20% to 80% vertical\n\n"
        "2. **Abnormality Detection**: \n"
        "   - Set 'abnormal': true for ANY visible abnormality\n"
        "   - Set 'abnormal': false ONLY for normal anatomical structures\n"
        "   - If uncertain, include with lower confidence (0.3-0.6) rather than omitting\n\n"
        "3. **Confidence Scoring**:\n"
        "   - 0.8-1.0: Very clear, obvious abnormality\n"
        "   - 0.6-0.8: Likely abnormality with clear features\n"
        "   - 0.4-0.6: Possible abnormality, some uncertainty\n"
        "   - 0.2-0.4: Subtle finding, high uncertainty\n\n"
        "4. **Common Cardiac Abnormalities to Look For**:\n"
        "   - Cardiomegaly (enlarged heart)\n"
        "   - Pericardial effusion (fluid around heart)\n"
        "   - Pulmonary edema (fluid in lungs)\n"
        "   - Pleural effusion (fluid around lungs)\n"
        "   - Atelectasis (collapsed lung tissue)\n"
        "   - Pneumonia or infiltrates\n"
        "   - Heart failure signs\n\n"
        "5. **Normal Images**: If the image appears normal:\n"
        "   - Set diagnosis to 'normal cardiac and pulmonary examination'\n"
        "   - Set organs_affected to empty array []\n"
        "   - Set regions to empty array []\n\n"
        "6. **Output Format**: Return ONLY the JSON object\n"
        "   - NO markdown code blocks (no ```json)\n"
        "   - NO explanatory text before or after\n"
        "   - NO comments in the JSON\n\n"
        + examples_text
        + "\n\nNow analyze this medical image:"
    )

    attempt = 0
    last_exc = None
    
    while attempt <= retries:
        try:
            client = _get_client()
            
            # Use proper vision API format with image_url
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user},
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                    "detail": "high"  # Request high detail analysis
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000,
            )

            # Extract response text
            raw_text = resp.choices[0].message.content.strip()

            # Save raw response for debugging
            try:
                with open(DEBUG_DUMP, "w", encoding="utf-8") as fh:
                    fh.write(f"=== Raw Response ===\n{raw_text}\n")
            except Exception:
                pass

            # Clean response: remove markdown code blocks
            if raw_text.startswith("```"):
                # Remove opening ```json or ```
                raw_text = re.sub(r'^```(?:json)?\s*\n?', '', raw_text)
                # Remove closing ```
                raw_text = re.sub(r'\n?```\s*$', '', raw_text)
                raw_text = raw_text.strip()
            
            # Extract JSON object (handles any remaining text around it)
            match = re.search(r'(\{[\s\S]*\})', raw_text)
            if not match:
                raise ValueError(f"No JSON object found in response: {raw_text[:200]}")
            
            json_text = match.group(1)
            
            # Parse JSON
            parsed = json.loads(json_text)

            # Validate required top-level fields
            if "diagnosis" not in parsed:
                raise ValueError("Response missing required field: 'diagnosis'")
            if "regions" not in parsed:
                parsed["regions"] = []  # Allow missing regions field, default to empty
            if "organs_affected" not in parsed:
                parsed["organs_affected"] = []

            # Validate and normalize each region
            validated_regions = []
            for idx, r in enumerate(parsed.get("regions", [])):
                try:
                    # Check required fields
                    if "bbox" not in r:
                        continue  # Skip region without bbox
                    
                    # Get abnormal flag (default to False if missing)
                    abnormal = r.get("abnormal", False)
                    
                    # Skip regions that are explicitly marked as normal
                    if abnormal is False:
                        continue
                    
                    # Validate bbox
                    bbox = r["bbox"]
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    
                    # Convert and validate bbox values
                    try:
                        bbox = [float(x) for x in bbox]
                    except (ValueError, TypeError):
                        continue
                    
                    # Ensure bbox values are valid (should be 0-1 for normalized)
                    if not all(isinstance(x, (int, float)) for x in bbox):
                        continue
                    
                    # Get confidence (default to 0.5 if missing)
                    confidence = r.get("confidence", 0.5)
                    if not isinstance(confidence, (int, float)):
                        confidence = 0.5
                    confidence = max(0.0, min(1.0, float(confidence)))
                    
                    # Build validated region
                    validated_region = {
                        "label": str(r.get("label", f"abnormality_{idx+1}")),
                        "abnormality": str(r.get("abnormality", r.get("label", "unknown"))),
                        "abnormal": True,  # Only include abnormal regions
                        "bbox": bbox,
                        "confidence": confidence
                    }
                    
                    validated_regions.append(validated_region)
                    
                except Exception as e:
                    # Log but don't fail - skip invalid region
                    try:
                        with open(DEBUG_DUMP, "a", encoding="utf-8") as fh:
                            fh.write(f"\nSkipped region {idx}: {e}\n")
                    except:
                        pass
                    continue
            
            # Update with validated regions
            parsed["regions"] = validated_regions
            
            # Log final parsed result
            try:
                with open(DEBUG_DUMP, "a", encoding="utf-8") as fh:
                    fh.write(f"\n=== Final Parsed JSON ===\n{json.dumps(parsed, indent=2)}\n")
            except:
                pass
            
            return json.dumps(parsed)
            
        except Exception as exc:
            last_exc = exc
            attempt += 1
            
            # Log error
            try:
                with open(DEBUG_DUMP, "a", encoding="utf-8") as fh:
                    fh.write(f"\n=== Attempt {attempt} Failed ===\n{str(exc)}\n")
            except:
                pass
            
            if attempt <= retries:
                time.sleep(backoff * attempt)

    raise RuntimeError(
        f"Failed to get valid JSON from model after {retries+1} attempts. "
        f"Last error: {last_exc}. Check {DEBUG_DUMP} for details."
    )