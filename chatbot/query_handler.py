def parse_query(user_input):
    import re
    keywords = user_input.lower()
    date_match = re.findall(r'\b(20\d{2})\b', keywords)
    return {
        "court": "supreme" if "supreme court" in keywords else "federal",
        "topic": keywords,
        "year": date_match[0] if date_match else None
    }
