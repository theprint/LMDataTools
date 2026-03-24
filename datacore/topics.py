# datacore/topics/loader.py

import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from .config.settings import config


def load_topics(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load topics structure from JSON file.
    
    Args:
        file_path: Path to topics file (uses config default if None)
        
    Returns:
        Complete topics structure with tiers
    """
    if file_path is None:
        # Construct path to topics.json in the project root, assuming it's one level up from datacore
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, "topics.json")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_topics_by_tier(tier_num: int, file_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Get all topics for a specific tier.
    
    Args:
        tier_num: Tier number (1-6)
        file_path: Path to topics file
        
    Returns:
        List of topics with their descriptions
        
    Raises:
        ValueError: If tier not found
    """
    data = load_topics(file_path)
    
    for tier in data["tiers"]:
        if tier["tier"] == tier_num:
            return tier["topics"]
    
    raise ValueError(f"Tier {tier_num} not found")


def get_tier_info(tier_num: int, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get complete information about a tier.
    
    Args:
        tier_num: Tier number (1-6)
        file_path: Path to topics file
        
    Returns:
        Tier information including name and topics
    """
    data = load_topics(file_path)
    
    for tier in data["tiers"]:
        if tier["tier"] == tier_num:
            return tier
    
    raise ValueError(f"Tier {tier_num} not found")


def get_random_topic(
    tier_weights: Optional[List[Tuple[int, float]]] = None,
    file_path: Optional[str] = None
) -> Tuple[str, int]:
    """
    Get a random topic with weighted tier selection.
    
    Args:
        tier_weights: List of (tier_number, weight) tuples
                     Default weights match writer.py pattern
        file_path: Path to topics file
        
    Returns:
        Tuple of (topic_string, tier_number) where topic_string is
        formatted as "Topic: description"
    """
    if tier_weights is None:
        # Default weights from writer.py
        tier_weights = [
            (1, 0.20),
            (2, 0.35),
            (3, 0.10),
            (4, 0.15),
            (5, 0.10),
            (6, 0.10)
        ]
    
    data = load_topics(file_path)
    
    # Choose tier based on weights
    chosen_tier_num = random.choices(
        [t for t, _ in tier_weights],
        weights=[w for _, w in tier_weights],
        k=1
    )[0]
    
    # Get tier data
    chosen_tier = next(t for t in data["tiers"] if t["tier"] == chosen_tier_num)
    
    # Choose random topic from tier
    topic = random.choice(chosen_tier["topics"])
    
    # Format as "topic: description"
    topic_string = f"{topic['topic']}: {topic['description']}"
    
    return topic_string, chosen_tier_num


def list_all_topics(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get a flat list of all topics across all tiers.
    
    Args:
        file_path: Path to topics file
        
    Returns:
        List of all topics with tier information added
    """
    data = load_topics(file_path)
    all_topics = []
    
    for tier in data["tiers"]:
        tier_num = tier["tier"]
        tier_name = tier["name"]
        
        for topic in tier["topics"]:
            topic_with_tier = topic.copy()
            topic_with_tier["tier"] = tier_num
            topic_with_tier["tier_name"] = tier_name
            all_topics.append(topic_with_tier)
    
    return all_topics


def get_tier_count(file_path: Optional[str] = None) -> int:
    """Get the number of tiers."""
    data = load_topics(file_path)
    return len(data["tiers"])


def get_topic_count(tier_num: Optional[int] = None, file_path: Optional[str] = None) -> int:
    """
    Get count of topics.
    
    Args:
        tier_num: If specified, count topics in that tier only
        file_path: Path to topics file
        
    Returns:
        Topic count
    """
    if tier_num is None:
        all_topics = list_all_topics(file_path)
        return len(all_topics)
    else:
        topics = get_topics_by_tier(tier_num, file_path)
        return len(topics)