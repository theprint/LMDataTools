# datacore/personas/generator.py

import random
from typing import Optional, Tuple, List
from ..llm.client import LLMClient


class PersonaGenerator:
    """Generate personas procedurally."""
    
    def __init__(self, client: Optional[LLMClient] = None):
        """
        Initialize persona generator.
        
        Args:
            client: LLM client (creates default if None)
        """
        if client is None:
            from ..llm.client import LLMClient
            from ..config.settings import config
            client = LLMClient(base_url=config.LLM_BASE_URL)
        
        self.client = client
    
    def generate_asker_persona(
        self, 
        asker_type: str, 
        topic: str,
        temperature: float = 0.8
    ) -> str:
        """
        Generate a persona for asking questions (questions.py style).
        
        Args:
            asker_type: Base type (e.g., "someone being introduced to")
            topic: Topic area
            temperature: LLM temperature
            
        Returns:
            Persona description string
        """
        prompt = (
            "You are developing a realistic persona as part of synthesizing a data set. "
            "Your precision and creativity is required. "
            "Please create and return a partial string (for later concatenation) that briefly "
            "describes 1 specific type of person (use a job title or similar descriptor) "
            f"doing something. Both the character and the action must in some way be related to {topic}, "
            "but keep it believable and a little open-ended. The character should be someone with a "
            "related question or problem to solve. "
            "It's crucial that the string you create follows the format shown in the 3 examples here "
            "(just the words, not the 'example x:' part), otherwise the result may not concatenate properly:\n"
            "\nexample 1: a seasoned TTRPG game master who brainstorming a campaign"
            "\nexample 2: an amateur home cook who is struggling to learn a new technique"
            "\nexample 3: a budding author working out an issue with their novel"
            f"\n\nFor this particular string, think of {asker_type} the topic of {topic} as basis for your response. "
            "Please respond with your partial string and no other text."
        )
        
        system_prompt = "You are a helpful assistant that generates realistic personas."
        
        persona = self.client.call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=1024
        )
        
        # Strip quotation marks if present
        persona = persona.strip()
        if persona.startswith('"') and persona.endswith('"'):
            persona = persona[1:-1]
        if persona.startswith("'") and persona.endswith("'"):
            persona = persona[1:-1]
        
        return persona
    
    def generate_writer_persona(
        self,
        age_range: Tuple[int, int] = (19, 64)
    ) -> Tuple[str, str]:
        """
        Generate a writer persona (writer.py style).
        
        Args:
            age_range: Min and max age
            
        Returns:
            Tuple of (system_prompt, doc_type)
        """
        age = f"{random.randint(age_range[0], age_range[1])}-year-old"
        name = self._generate_name()
        
        things_to_like = [
            "cats",
            "wine",
            "board games",
            "cinema",
            "the arts",
            "what-if-scenarios",
            "all things nature",
            "tech gadgets",
            "genre fiction",
            "word puzzles",
            "late nights",
            "early mornings",
            "physical exercise",
            "pushing boundaries",
            "social justice",
            "self-reflection",
            "trying new things",
            "meeting new people",
            "alone time",
            "anecdotal embellishments",
            "satirical humor",
            "everyday observations",
            "relatable metaphors",
            "historical references",
            "technology and innovation",
            "social dynamics",
            "studying human behavior",
            "family traditions",
            "strategic thinking",
            "science and discovery",
        ]
        things_to_dislike = [
            "mob mentality",
            "jumping to conclusions",
            "tough love",
            "injustice",
            "inequality",
            "most people",
            "short-term thinking",
            "opinion-based arguments",
            "clutter",
            "restrictions",
            "government overreach",
            "corporate influence",
            "consumerism",
            "procrastination",
            "monotony",
            "moralizing",
            "organized religion",
            "outrage culture",
            "selfishness",
            "bad food",
            "banal small talk",
            "clichés",
            "self-deprecation",
            "political correctness",
            "stress and rushing",
            "hustle culture",
            "materialism",
            "apathy",
            "willful ignorance",
            "preachiness",
        ]
        likedislike = f"who {random.choice(['values', 'likes', 'is into', 'is passionate about'])} {random.choice(things_to_like)} and {random.choice(['has a distaste for', 'strongly dislikes', 'is not a fan of', 'hates'])} {random.choice(things_to_dislike)}"

        locations = [
            "Seattle, Washington",
            "Austin, Texas",
            "New York, New York",
            "San Francisco, California",
            "Miami, Florida",
            "Chicago, Illinois",
            "Vancouver, Canada",
            "London, UK",
            "Edinburgh, Scotland",
            "Dublin, Ireland",
            "Sydney, Australia",
            "Melbourne, Australia",
            "Toronto, Canada",
            "Dallas, Texas",
            "Los Angeles, California",
            "Las Vegas, Nevada",
            "Boston, Massachusetts",
            "San Diego, California",
            "Portland, Oregon",
            "Denver, Colorado",
            "Witchita, Kansas",
            "Atlanta, Georgia",
            "Washington, D.C.",
            "New Orleans, Louisiana",
            "Phoenix, Arizona",
        ]
        location = f"{random.choice(['from', 'located in', 'living in', 'based out of'])} {random.choice(locations)}"

        personas = [
            (f"You are a {age} blogger named {name}, {location}, {likedislike}.", "blog post"),
            (f"You are a {age} news reporter {location}, {likedislike}. Your name is {name}.", "news article"),
            (f"You are a {age} deep-diving reporter named {name} {location}, {likedislike}.", "investigative report"),
            (f"You are a {age} technical writer {likedislike}, named {name}. You are {location}.", "technical document"),
            (f"You are a {age} essay writer {location}. You are {name}, someone {likedislike}.", "essay"),
            (f"You are a {age} influencer {location} {likedislike}, named {name}.", "social media post"),
            (f"You are a {age} marketer named {name} {location}, {likedislike}.", "marketing copy"),
            (f"You are a {age} novelist named {name} {location}, {likedislike}.", "short story"),
            (f"You are a {age} academic researcher {location}, {likedislike}, and is named {name}.", "research paper"),
            (f"You are a {age} freelance journalist {likedislike}, named {name} and {location}.", "feature article"),
            (f"You are a {age} editor named {name} {location}, {likedislike}.", "editorial"),
            (f"You are a {age} self-help author {location} named {name}, {likedislike}.", "self-help article"),
            (f"You are a {age} concerned citizen {location} {likedislike}, whose name is {name}.", "open letter"),
            (f"You are a {age} armchair opinion writer {likedislike}, is {location} and whose name is {name}.", "opinion piece"),
            (f"You are a {age} analyst named {name} {location}, {likedislike}.", "analytical report"),
            (f"You are a {age} copy writer named {name} {location}, {likedislike}.", "advertisement"),
            (f"You are a {age} media pundit {likedislike}, named {name}, {location}.", "opinion column"),
            (f"You are a {age} historian named {name}, {likedislike} and is {location}.", "historical article"),
            (f"You are a {age} scholar named {name}, {likedislike}.", "academic article"),
            (f"You are a {age} senior journalist {location}, named {name}, and {likedislike}.", "in-depth article"),
            (f"You are a {age} encycopedia writer {likedislike}, is {location} and named {name}.", "encyclopedia entry"),
            (f"You are a {age} tutor {likedislike}, {location}, named {name}.", "educational article"),
            (f"You are a {age} coach named {name} {location}, {likedislike}.", "motivational article"),
            (f"You are a {age} social activist {location}, {likedislike}. Your name is {name}.", "advocacy article"),
            (f"You are a {age} tabloid journalist {location} named {name}, {likedislike}.", "tabloid story"),
            (f"You are a {age} local reporter {location} named {name}, {likedislike}.", "local news story"),
            (f"You are a {age} upset citizen {likedislike}, named {name}. You are {location}.", "complaint letter"),
            (f"You are a {age} non-fiction author named {name} {location}, {likedislike}.", "non-fiction document"),
        ]
        
        return random.choice(personas)
    
    def generate_writing_style(self) -> str:
        """
        Generate a writing style descriptor.
        
        Returns:
            Style description string
        """
        styles = [
            "formal and academic",
            "casual and conversational",
            "persuasive and motivational",
            "descriptive and vivid",
            "analytical and critical",
            "explanatory and informative",
            "fun and light-hearted",
            "snarky and witty",
            "wordy and elaborate",
            "minimalist and to-the-point",
            "authoritative and confident",
            "empathetic and compassionate",
            "speculative and imaginative",
            "argumentative and logical",
            "investigative and thorough",
            "complaining and sardonic",
            "optimistic and uplifting",
            "sarcastic and dry",
            "heartfelt and sincere",
            "humorous and entertaining",
            "poetic and lyrical",
            "straightforward and concise",
            "investigative and detailed",
            "reflective and introspective",
            "dramatic and intense",
            "playful and quirky",
            "technical and precise",
            "formal and matter-of-fact",
            "explanatory and formal",
            "unimaginitive and plain",
            "procedural and clear",
            "amusing and light-hearted",
            "critical and evaluative",  
            "narrative and engaging",
        ]
        return random.choice(styles)
    
    def _generate_name(self, gender: Optional[str] = None) -> str:
        """
        Generate a random name.
        
        Args:
            gender: "male", "female", or None for random
            
        Returns:
            Full name string
        """
        if gender is None:
            gender = random.choice(["male", "female"])
        
        if gender == "male":
            first_names = [
                "John", "Michael", "David", "James", "Robert", "William", "Richard", 
                "Joseph", "Thomas", "Charles", "Neil", "Daniel", "Paul", "Mark", "Donald", 
                "Eugene", "Kevin", "Brian", "George", "Edward", "Ronald", "Kenneth", 
                "Steven", "Anthony", "Eric", "Scott", "Andrew", "Raymond", "Gregory", 
                "Joshua", "Jerry", "Dennis", "Walter", "Patrick", "Peter", "Harold", 
                "Douglas", "Henry", "Carl", "Arthur", "Ryan", "Roger", "Joe", "Juan", 
                "Jack", "Albert", "Jonathan", "Justin", "Terry", "Gerald", "Keith", 
                "Samuel", "Willie", "Ralph", "Lawrence", "Nicholas", "Roy", "Benjamin", 
                "Bruce", "Brandon", "Adam", "Harry", "Fred", "Wayne", "Billy", "Steve", 
                "Louis", "Jeremy", "Aaron", "Randy", "Howard", "Carlos", "Russell", 
                "Bobby", "Mohammed", "Vincent", "Johnny", "Phillip", "Logan", "Brett", 
                "Craig", "Quentin", "Devin", "Maxwell", "Caleb", "Gavin", "Zachary", 
                "Cameron", "Leonard", "Rahul", "Miguel", "Martin", "Dylan", "Phil"
            ]
        else:
            first_names = [
                "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", 
                "Jessica", "Sarah", "Karen", "Emily", "Laura", "Megan", "Rachel", "Angela", 
                "Samantha", "Nicole", "Stephanie", "Cynthia", "Amy", "Sharon", "Michelle", 
                "Elaine", "Donna", "Carol", "Amanda", "Melissa", "Deborah", "Rebecca", 
                "Helen", "Shirley", "Catherine", "Christine", "Sophie", "Janet", "Ruth", 
                "Maria", "Diane", "Virginia", "Julie", "Joyce", "Victoria", "Olivia", 
                "Kelly", "Christina", "Lauren", "Joan", "Evelyn", "Judith", "Martha", 
                "Cheryl", "Mildred", "Katherine", "Andrea", "Ann", "Jean", "Alice", 
                "Julia", "Judy", "Hannah", "Grace", "Denise", "Amber", "Marilyn", 
                "Beverly", "Danielle", "Theresa", "Sophia", "Marie", "Doris", "Madison", 
                "Ayesha", "Frances", "Kathleen", "Janice", "Jeanette", "Rose", "Brittany", 
                "Diana", "Abigail", "Natalie", "Jane", "Lori", "Debbie", "Serena", 
                "Tiffany", "Lillian", "Tammy", "Irene", "Tonya", "Jada", "Reba"
            ]
        
        last_names = [
            "Smith", "Johnson", "Brown", "Williams", "Jones", "Garcia", "Miller", "Davis", 
            "Rodriguez", "Martinez", "Kowalski", "Anderson", "Hansen", "Gonzalez", "Wilson", 
            "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez", "Schmidt", 
            "Sanchez", "Ramirez", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", 
            "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", 
            "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", 
            "Roberts", "Rasmussen", "Cline", "Goldman", "Alexander", "Phillips", "Hoffman", 
            "Johnston", "Woods", "Coleman", "West", "Jordan", "Owens", "Reed", "Foster", 
            "Graham", "Kim", "Simmons", "Butler", "Barnes", "Ross", "Henderson", "Cole", 
            "Jenkins", "Perry", "Powell", "Long", "Patterson", "Hughes", "Washington", 
            "Sullivan", "Fisher", "Myers", "Ford", "Hamilton", "Gonzales", "Garza", "Burke", 
            "Ellis", "Harrison", "Fernandez", "Mcdonald", "Wayne", "Jefferson", "Bryant", 
            "Russell", "Griffin", "Diaz", "Hayes", "Chen", "Cox", "Carlson", "Pope", "Lynn", 
            "Curry", "Marshall", "Gilbert", "Reynolds", "Greene", "Burton", "Santos", "Mason", 
            "Clayton", "Poole", "Calhoun", "Vasquez", "Morales", "Richards", "Willis", 
            "Peterson", "Woodard"
        ]
        
        first_name = random.choice(first_names)
        
        # Add middle initial sometimes
        if random.random() < 0.34:
            middle_initial = chr(random.randint(65, 90))  # ASCII A-Z
            first_name = f"{first_name} {middle_initial}."
        
        # Hyphenated last name sometimes
        if random.random() < 0.08:
            hyphenated = random.sample(last_names, 2)
            last_name = f"{hyphenated[0]}-{hyphenated[1]}"
        else:
            last_name = random.choice(last_names)
        
        return f"{first_name} {last_name}"


# Convenience functions
def generate_asker(asker_type: str, topic: str, client: Optional[LLMClient] = None) -> str:
    """Generate an asker persona quickly."""
    generator = PersonaGenerator(client)
    return generator.generate_asker_persona(asker_type, topic)


def generate_writer(client: Optional[LLMClient] = None) -> Tuple[str, str]:
    """Generate a writer persona quickly."""
    generator = PersonaGenerator(client)
    return generator.generate_writer_persona()