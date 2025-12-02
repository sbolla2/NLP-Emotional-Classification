class EmotionExample: 
    """
    Represents a single piece of training or testing data
    """
    def __init__(self, tokens: list, transformer_tokens: dict[str, list[int]] | None, emotional_intensity: float, emotional_polarity: float, empathy: float):
        self.tokens = tokens
        self.transformer_tokens = transformer_tokens
        self.emotional_intensity = emotional_intensity
        self.emotional_polarity = emotional_polarity
        self.empathy = empathy