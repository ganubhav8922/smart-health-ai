import requests
from langchain_core.tools import tool

@tool
def get_weather_precautions(lat: float, lon: float) -> str:
    """
    Fetches live weather data using Open-Meteo API and generates
    immediate health risks based on temperature and humidity.
    Inputs are latitude (lat) and longitude (lon).
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
    
    try:
        response = requests.get(url, timeout=10).json()
        current_data = response.get("current", {})
        temp = current_data.get("temperature_2m")
        humidity = current_data.get("relative_humidity_2m")
        
        if temp is None or humidity is None:
            return "Unable to parse weather data at the moment."
            
        report = f"The current outdoor temperature is {temp}°C with {humidity}% humidity. "
        
        if temp > 38:
            report += "🚨 WARNING: Extreme heat detected! High risk of heat exhaustion or heatstroke. Advise staying indoors and drinking plenty of fluids."
        elif temp < 10:
            report += "🥶 NOTICE: Cold weather detected. Advise dressing in layers to avoid hypothermia."
        elif humidity > 80 and temp > 25:
            report += "🦟 NOTICE: High humidity and warm weather detected. High risk for mosquito breeding (Dengue/Malaria)."
        else:
            report += "☀️ Weather conditions look moderate. No immediate environmental health risks detected."
            
        return report

    except Exception as e:
        return f"Could not connect to the weather service: {str(e)}"


@tool
def emergency_hospital_locator_and_router(lat: float, lon: float) -> str:
    """
    Finds nearby hospitals using the free OpenStreetMap Overpass API
    and returns their names along with Google Maps navigation route links.
    Inputs are latitude (lat) and longitude (lon).
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    radius = 5000 
    
    query = f"""[out:json];(node["amenity"="hospital"](around:{radius},{lat},{lon});way["amenity"="hospital"](around:{radius},{lat},{lon}););out center;"""
    
    headers = {
        'User-Agent': 'HealthBotProject/1.0 (anubhavgupta; contact: developer)',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(overpass_url, params={'data': query}, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return f"🚨 Locator service returned status code: {response.status_code}"
            
        data = response.json()
        elements = data.get("elements", [])
        
        if not elements:
            return f"⚠️ No hospitals found within a {radius/1000}km radius of your location. If this is an emergency, call local emergency services immediately."
        
        report = "🚨 **EMERGENCY MEDICAL FACILITIES FOUND NEAR YOU:**\n\n"
        
        for idx, element in enumerate(elements[:3], 1):
            tags = element.get("tags", {})
            name = tags.get("name", "Un-named Emergency Facility")
            
            hosp_lat = element.get("lat") or element.get("center", {}).get("lat")
            hosp_lon = element.get("lon") or element.get("center", {}).get("lon")
            
            maps_url = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={hosp_lat},{hosp_lon}"
            report += f"{idx}️⃣ **{name}**\n 🗺️ Route Link: {maps_url}\n\n"
            
        return report
        
    except Exception as e:
        return f"🚨 Error connecting to emergency locator services: {str(e)}"

# Alias so both function names work if called elsewhere
find_emergency_hospital = emergency_hospital_locator_and_router


if __name__ == "__main__":
    sample_lat = 26.4499
    sample_lon = 80.3319
    
    print("🌤️ Testing Weather Tool:")
    print(get_weather_precautions.invoke({"lat": sample_lat, "lon": sample_lon}))
    print("-" * 40)
    print("🏥 Testing Hospital Finder Tool:")
    print(emergency_hospital_locator_and_router.invoke({"lat": sample_lat, "lon": sample_lon}))
