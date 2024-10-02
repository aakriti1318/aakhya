# src/looker_url_generator.py
class LookerURLGenerator:
    def __init__(self):
        self.urls = {
            "other": "https://lookerstudio.google.com/reporting/a3d4e601-f065-43ba-b311-6f3f15e2ead5/page/DONBE/",
            "cosmetic": "https://lookerstudio.google.com/reporting/d9c0cbf4-0d34-4be2-8c7f-ed399ea8d645/page/DONBE/"
        }

    def get_looker_url(self, classification):
        """
        Returns the appropriate Looker Studio URL based on the classification.
        
        Args:
        classification (str): The classification of the data ('cosmetic' or 'other')
        
        Returns:
        str: The corresponding Looker Studio URL
        """
        classification = classification.lower()
        if classification not in self.urls:
            raise ValueError(f"Invalid classification: {classification}. Expected 'cosmetic' or 'other'.")
        
        return self.urls[classification]


# Usage example:
if __name__ == "__main__":
    url_generator = LookerURLGenerator()
    
    # Test with 'cosmetic' classification
    cosmetic_url = url_generator.get_looker_url("cosmetic")
    print(f"Cosmetic URL: {cosmetic_url}")
    
    # Test with 'other' classification
    other_url = url_generator.get_looker_url("other")
    print(f"Other URL: {other_url}")
    
    # Uncomment to test with invalid classification
    # invalid_url = url_generator.get_looker_url("invalid")