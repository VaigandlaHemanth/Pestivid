"""
Simple AI Agent for PestVid Project
A lightweight agricultural assistant that provides crop advice, pest identification, and farming recommendations
"""

import os
import logging
import requests
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class SimplePestVidAgent:
    """Simple AI Agent for agricultural assistance in PestVid"""
    
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        # Groq retires model ids roughly yearly - keep in config, never inline.
        self.groq_model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Agricultural knowledge base
        self.crop_diseases = {
            "tomato": ["blight", "mosaic virus", "bacterial spot", "fusarium wilt"],
            "potato": ["late blight", "early blight", "bacterial soft rot", "virus diseases"],
            "corn": ["corn borer", "rust", "smut", "leaf spot"],
            "wheat": ["rust", "powdery mildew", "septoria", "fusarium head blight"]
        }
        
        self.pest_treatments = {
            "aphids": "Use neem oil spray or introduce ladybugs. Apply insecticidal soap.",
            "caterpillars": "Use Bt (Bacillus thuringiensis) spray or hand-pick larger ones.",
            "spider_mites": "Increase humidity, use predatory mites, or miticide spray.",
            "whiteflies": "Use yellow sticky traps and neem oil spray.",
            "thrips": "Use blue sticky traps and beneficial insects like minute pirate bugs."
        }
        
        self.farming_tips = {
            "watering": "Water early morning or evening. Deep, less frequent watering is better.",
            "fertilizing": "Use balanced NPK fertilizer. Organic compost improves soil health.",
            "pruning": "Remove dead/diseased parts. Prune for air circulation.",
            "soil": "Test soil pH regularly. Most crops prefer 6.0-7.0 pH range."
        }
    
    def get_crop_advice(self, query: str) -> Dict[str, str]:
        """Get agricultural advice using Groq API"""
        if not self.groq_api_key:
            return self._get_local_advice(query, reason='GROQ_API_KEY is not set')
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            system_prompt = """You are an expert agricultural advisor for the PestVid platform. 
            Provide practical, actionable advice for farmers about:
            - Crop diseases and pest management
            - Planting and harvesting schedules
            - Soil health and fertilization
            - Irrigation and water management
            - Organic and sustainable farming practices
            
            Keep responses concise but informative. Always prioritize farmer safety and environmental sustainability."""
            
            data = {
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 1200,
                "temperature": 0.7
            }
            
            response = requests.post(self.groq_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                advice = (result['choices'][0]['message'].get('content') or '').strip()
                if not advice:
                    # Reasoning models can return HTTP 200 with empty content.
                    return self._get_local_advice(
                        query, reason='Groq returned an empty completion')
                return {
                    "status": "success",
                    "advice": advice,
                    "source": "Groq AI",
                    "query": query
                }
            else:
                # Groq returns a machine-readable error.code here, e.g.
                # "model_decommissioned" - surface it instead of discarding it.
                body = response.text[:300]
                logger.error("Groq HTTP %s for model %s: %s",
                             response.status_code, self.groq_model, body)
                return self._get_local_advice(
                    query, reason=f'Groq HTTP {response.status_code}: {body}')

        except Exception as e:
            logger.exception("Groq request failed for model %s", self.groq_model)
            return self._get_local_advice(query, reason=f'{type(e).__name__}: {e}'[:200])
    
    def _get_local_advice(self, query: str, reason: str = 'unspecified') -> Dict[str, str]:
        """Provide local advice when AI API is not available"""
        query_lower = query.lower()
        advice = "I'm here to help with your farming questions! "
        
        # Check for specific topics
        if any(pest in query_lower for pest in self.pest_treatments.keys()):
            for pest, treatment in self.pest_treatments.items():
                if pest in query_lower:
                    advice += f"For {pest}: {treatment}"
                    break
        
        elif any(crop in query_lower for crop in self.crop_diseases.keys()):
            for crop, diseases in self.crop_diseases.items():
                if crop in query_lower:
                    advice += f"Common {crop} issues include: {', '.join(diseases)}. "
                    advice += "Ensure proper spacing, good drainage, and regular monitoring."
                    break
        
        elif any(tip in query_lower for tip in ["water", "irrigat"]):
            advice += self.farming_tips["watering"]
        
        elif any(tip in query_lower for tip in ["fertil", "nutri"]):
            advice += self.farming_tips["fertilizing"]
        
        elif any(tip in query_lower for tip in ["prun", "trim"]):
            advice += self.farming_tips["pruning"]
        
        elif any(tip in query_lower for tip in ["soil", "ph"]):
            advice += self.farming_tips["soil"]
        
        else:
            advice += """Here are some general farming tips:
            1. Monitor your crops daily for early problem detection
            2. Maintain proper plant spacing for air circulation
            3. Use integrated pest management (IPM) approaches
            4. Keep detailed records of planting, treatments, and harvests
            5. Consider companion planting for natural pest control"""
        
        # NOT "success": this is the canned keyword dictionary, not the LLM.
        # Reporting success here is what hid a 12-month Groq outage.
        logger.warning("Serving canned local advice instead of the LLM: %s", reason)
        return {
            "status": "degraded",
            "advice": advice,
            "source": "Local Knowledge Base (fallback)",
            "reason": reason,
            "query": query
        }
    
    def analyze_crop_image_description(self, description: str) -> Dict[str, str]:
        """Analyze crop issues based on text description"""
        description_lower = description.lower()
        
        # Disease indicators
        disease_indicators = {
            "yellow leaves": "Possible nutrient deficiency (nitrogen) or overwatering",
            "brown spots": "Likely fungal disease - improve air circulation, reduce humidity",
            "wilting": "Check for root rot, watering issues, or pest damage",
            "holes in leaves": "Insect damage - look for caterpillars, beetles, or slugs",
            "white powder": "Powdery mildew - increase air circulation, apply fungicide",
            "curled leaves": "Possible virus, aphid damage, or environmental stress"
        }
        
        diagnosis = "Based on your description: "
        recommendations = []
        
        for symptom, advice in disease_indicators.items():
            if symptom in description_lower:
                diagnosis += f"{advice}. "
                
                if "fungal" in advice:
                    recommendations.append("Apply organic fungicide or neem oil")
                elif "insect" in advice:
                    recommendations.append("Inspect plants closely and use appropriate pest control")
                elif "nutrient" in advice:
                    recommendations.append("Test soil and apply balanced fertilizer")
        
        if not recommendations:
            recommendations = [
                "Take clear photos of affected plants",
                "Monitor environmental conditions (temperature, humidity)",
                "Check soil moisture levels",
                "Look for signs of pests or diseases"
            ]
        
        return {
            "status": "success",
            "source": "Rule-based keyword matcher (no LLM, no image analysis)",
            "diagnosis": diagnosis,
            "recommendations": recommendations,
            "next_steps": "Consider uploading images to PestVid for more detailed analysis"
        }
    
    def get_seasonal_advice(self, season: str, crop: str = None) -> Dict[str, str]:
        """Get seasonal farming advice"""
        season_lower = season.lower()
        
        seasonal_advice = {
            "spring": {
                "general": "Prepare soil, start seeds indoors, plan crop rotation",
                "tasks": ["Soil testing", "Seed starting", "Tool maintenance", "Irrigation setup"]
            },
            "summer": {
                "general": "Focus on watering, pest monitoring, and harvesting early crops",
                "tasks": ["Regular watering", "Pest control", "Harvesting", "Pruning"]
            },
            "fall": {
                "general": "Harvest main crops, prepare for winter, plant cover crops",
                "tasks": ["Harvesting", "Seed saving", "Soil preparation", "Equipment storage"]
            },
            "winter": {
                "general": "Plan next year, maintain equipment, study and learn",
                "tasks": ["Planning", "Equipment maintenance", "Learning", "Greenhouse management"]
            }
        }
        
        if season_lower in seasonal_advice:
            advice_data = seasonal_advice[season_lower]
            advice = f"For {season}: {advice_data['general']}"
            
            if crop:
                advice += f" For {crop} specifically, focus on appropriate seasonal care."
            
            return {
                "status": "success",
                "source": "Rule-based lookup table (no LLM)",
                "season": season,
                "advice": advice,
                "key_tasks": advice_data["tasks"],
                "crop": crop or "general"
            }
        
        return {
            "status": "error",
            "message": "Please specify a valid season: spring, summer, fall, or winter"
        }
    
    def get_quick_tips(self) -> List[str]:
        """Get quick farming tips"""
        return [
            "🌱 Check soil moisture before watering - stick your finger 2 inches deep",
            "🐛 Inspect plants weekly for early pest detection",
            "🌿 Rotate crops annually to prevent soil depletion and disease buildup",
            "💧 Water in early morning to reduce evaporation and disease risk",
            "🌾 Keep detailed records of planting dates, treatments, and harvests",
            "🦋 Plant flowers nearby to attract beneficial insects",
            "🍃 Mulch around plants to retain moisture and suppress weeds",
            "📱 Use PestVid to document and share your farming progress!"
        ]

# Global instance for easy import
pestivid_agent = SimplePestVidAgent()
