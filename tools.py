import requests

def get_weather_precautions(lat: float, lon: float) -> str:
    """
    Fetches live weather data using Open-Meteo API and generates
    immediate health risks based on temperature and humidity.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
    
    try:
        response = requests.get(url).json()
        current_data = response.get("current", {})
        temp = current_data.get("temperature_2m")
        humidity = current_data.get("relative_humidity_2m")
        
        if temp is None or humidity is None:
            return "Unable to parse weather data at the moment."
            
        report = f"The current outdoor temperature is {temp}°C with {humidity}% humidity. "
        
        if temp > 38:
            report += "🚨 WARNING: Extreme heat detected! High risk of heat exhaustion or heatstroke. Advise the user to stay indoors, avoid heavy exertion, and drink plenty of fluids."
        elif temp < 10:
            report += "🥶 NOTICE: Cold weather detected. Advise the user to dress in layers to avoid hypothermia, especially if they have pre-existing respiratory issues."
        elif humidity > 80 and temp > 25:
            report += "🦟 NOTICE: High humidity and warm weather detected. This creates prime breeding conditions for mosquitoes. Remind the user of Dengue/Malaria prevention guidelines."
        else:
            report += "☀️ The weather conditions look moderate. No immediate environmental health risks detected."
            
        return report

    except Exception as e:
        return f"Could not connect to the weather service: {str(e)}"


def find_emergency_hospital(lat: float, lon: float) -> str:
    """
    Finds nearby hospitals using the free OpenStreetMap Overpass API
    and returns their names along with a Google Maps navigation route link.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    radius = 5000 
    
    # Cleaning up the string format to ensure the server processes it correctly
    query = f"""[out:json];(node["amenity"="hospital"](around:{radius},{lat},{lon});way["amenity"="hospital"](around:{radius},{lat},{lon}););out center;"""
    
    # Adding a custom User-Agent to satisfy the server's security blocks
    headers = {
        'User-Agent': 'HealthBotProject/1.0 (anubhavgupta; contact: developer)',
        'Accept': 'application/json'
    }
    
    try:
        # Added headers=headers to the request
        response = requests.get(overpass_url, params={'data': query}, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return f"🚨 Error locator service returned status code: {response.status_code}"
            
        data = response.json()
        elements = data.get("elements", [])
        
        if not elements:
            return f"⚠️ No hospitals found within a {radius/1000}km radius of your location. If this is a life-threatening emergency, please call your local emergency hotline immediately."
        
        report = "🚨 **EMERGENCY MEDICAL FACILITIES FOUND NEAR YOU:**\n\n"
        
        for idx, element in enumerate(elements[:3], 1):
            tags = element.get("tags", {})
            name = tags.get("name", "Un-named Emergency Facility")
            
            hosp_lat = element.get("lat") or element.get("center", {}).get("lat")
            hosp_lon = element.get("lon") or element.get("center", {}).get("lon")
            
            maps_url = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={hosp_lat},{hosp_lon}"
            report += f"{idx}️⃣ **{name}**\n   🗺️ Route Link: {maps_url}\n\n"
            
        return report
        
    except Exception as e:
        return f"🚨 Error connecting to emergency locator services: {str(e)}"

if __name__ == "__main__":
    sample_lat = 26.4499
    sample_lon = 80.3319
    
    print("🌤️ Testing Weather Tool Output:")
    print(get_weather_precautions(sample_lat, sample_lon))
    print("-" * 40)
    print("🏥 Testing Hospital Finder & Router Output:")
    print(find_emergency_hospital(sample_lat, sample_lon))